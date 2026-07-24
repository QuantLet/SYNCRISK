#!/usr/bin/env python3
"""
PROVENANCE / BUILD SCRIPT (run once by the package authors; users do NOT run this).

Consolidates the SYNCRISK main run (runs/main_run_20260612/) into the publishable,
tidy data shipped under data/published/. It deliberately ships ONLY derived data and
our own LLM outputs:

  - per-arm tidy elicited scores (scores_P0.parquet, scores_P3.parquet) consolidated
    from runs/.../scores/checkpoints/*.parquet  [OUR LLM output -> ships in full]
  - panel_features.parquet = derived numeric features ONLY (the raw `headlines`
    column, which contains licensed EODHD news text, is DROPPED)
  - scm_panel.parquet : Eq.6 SCM series at h in {4,8,13} weeks, precomputed from the
    licensed returns_daily so users never need the raw returns
  - macro_weekly.parquet : the W_t controls used by Eq.7 (subset of columns)
  - main_vix.csv : weekly VIX (for the 90th-pct stress labelling)
  - residuals_{P0,P3}_{C2x,C3x}.parquet : expanding-window cleaned residuals (phase3x)
  - Amendment 8 (frontier tier): scores_frontier_{P0,P3}.parquet (tidy frontier-arm
    elicited scores; P0=3 frontier models, P3=2 — gemini-frontier P3 dropped on cost
    discipline, Amendment 8a), residuals9_{P0,P3}_{C1x,C2x,C3x}.parquet (9-model
    expanding-window residuals), phase3x9_results.json (authoritative tier numbers)
  - results: phase3x_results.json, phase3_results.json, eq7_results_v2.csv,
    excess_afsi_*.csv, retest_correlation.json
  - coverage CSVs, realized_panel.json, stress_weeks.json
  - design_change_log.md (Amendments 1-8a)
  - raw_io_manifest.json : counts + a small response-only sample (raw prompts, which
    embed licensed news, are NOT shipped)

NEVER ships: prices_daily, news_articles, returns_daily, sentiment, fundamentals,
or any data/panel_main_v2_20260612/ raw file.
"""
from __future__ import annotations
import json, sys, glob, shutil
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("/home/jovyan/syncrisk_pilot")
RUN = REPO / "runs/main_run_20260612"
HERE = Path(__file__).resolve().parent       # data/published
PKG = HERE.parents[1]                         # package/syncrisk
sys.path.insert(0, str(PKG / "src"))
from syncrisk.outcomes import scm_panel

ISET = "prices_plus_news"
SCORE_COLS = ["date", "asset_ticker", "model", "info_set",
              "score_relative_risk_score", "score_de_risking_signal",
              "cost_usd", "tokens_in", "tokens_out"]
HORIZONS_W = [4, 8, 13]


def consolidate_scores():
    info = json.load(open(RUN / "realized_panel.json"))
    for arm in ["P0", "P3"]:
        frames = []
        for f in sorted((RUN / "scores" / "checkpoints").glob(f"{arm}__*.parquet")):
            d = pd.read_parquet(f)
            d = d[d["score_relative_risk_score"].notna()]
            if len(d):
                d["info_set"] = ISET
                frames.append(d[SCORE_COLS])
        s = pd.concat(frames, ignore_index=True)
        s["date"] = pd.to_datetime(s["date"])
        s = s.sort_values(["date", "model", "asset_ticker"]).reset_index(drop=True)
        s.to_parquet(HERE / f"scores_{arm}.parquet", index=False)
        print(f"scores_{arm}.parquet: {len(s)} rows, {s['model'].nunique()} models, "
              f"{s['date'].nunique()} weeks, {s['asset_ticker'].nunique()} assets")
    return info


def consolidate_frontier_scores():
    """Amendment 8 frontier-tier elicited scores -> tidy per-arm parquets.

    Same schema as scores_{P0,P3}.parquet. Asymmetric design (Amendment 8a):
    P0 has 3 frontier models (claude/gpt/gemini-frontier); P3 has 2
    (claude/gpt-frontier; gemini-frontier P3 dropped on cost discipline).
    """
    for arm in ["P0", "P3"]:
        frames = []
        for f in sorted((RUN / "scores_frontier" / "checkpoints").glob(f"{arm}__*.parquet")):
            d = pd.read_parquet(f)
            d = d[d["score_relative_risk_score"].notna()]
            if len(d):
                d["info_set"] = ISET
                frames.append(d[SCORE_COLS])
        s = pd.concat(frames, ignore_index=True)
        s["date"] = pd.to_datetime(s["date"])
        s = s.sort_values(["date", "model", "asset_ticker"]).reset_index(drop=True)
        s.to_parquet(HERE / f"scores_frontier_{arm}.parquet", index=False)
        print(f"scores_frontier_{arm}.parquet: {len(s)} rows, {s['model'].nunique()} models "
              f"({sorted(s['model'].unique())}), {s['date'].nunique()} weeks")


def copy_phase3x9():
    """9-model tier x family artifacts (Amendment 8)."""
    shutil.copy(RUN / "phase3x9/phase3x9_results.json", HERE / "phase3x9_results.json")
    for arm in ["P0", "P3"]:
        for sn in ["C1x", "C2x", "C3x"]:
            shutil.copy(RUN / f"phase3x9/residuals9_{arm}_{sn}.parquet",
                        HERE / f"residuals9_{arm}_{sn}.parquet")
    print("copied phase3x9_results.json + 6 residuals9 parquets")


def panel_features():
    p = pd.read_parquet(RUN / "main_pert_features.parquet")
    drop = [c for c in ["headlines"] if c in p.columns]   # raw licensed news text
    p = p.drop(columns=drop)
    p["date"] = pd.to_datetime(p["date"])
    p.to_parquet(HERE / "panel_features.parquet", index=False)
    print(f"panel_features.parquet: {p.shape}, dropped {drop} (licensed raw news)")


def scm_series(info):
    rd = pd.read_parquet(REPO / "data/panel_main_v2_20260612/returns_daily.parquet")
    rd["date"] = pd.to_datetime(rd["date"])
    rd = rd[rd["asset_ticker"].isin(info["assets"])]
    panel = pd.read_parquet(HERE / "panel_features.parquet")
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    out = None
    for h in HORIZONS_W:
        s = scm_panel(rd, dates, h_days=h * 5, direction="forward")[["date", "scm"]]
        s = s.rename(columns={"scm": f"scm_h{h}"})
        out = s if out is None else out.merge(s, on="date", how="outer")
    out = out.sort_values("date").reset_index(drop=True)
    out.to_parquet(HERE / "scm_panel.parquet", index=False)
    print(f"scm_panel.parquet: {out.shape} (h in {HORIZONS_W} weeks)")


def macro():
    m = pd.read_parquet(REPO / "data/panel_20260602/macro_weekly.parquet")
    cols = [c for c in ["date", "vix", "hy_spread", "term_slope", "market_return"] if c in m.columns]
    m = m[cols].copy()
    m["date"] = pd.to_datetime(m["date"])
    m.to_parquet(HERE / "macro_weekly.parquet", index=False)
    print(f"macro_weekly.parquet: {m.shape} cols={cols}")


def copy_artifacts():
    files = [
        (RUN / "main_vix.csv", "main_vix.csv"),
        (RUN / "realized_panel.json", "realized_panel.json"),
        (RUN / "stress_weeks.json", "stress_weeks.json"),
        (RUN / "design_change_log.md", "design_change_log.md"),
        (RUN / "phase3x/phase3x_results.json", "phase3x_results.json"),
        (RUN / "phase3/phase3_results.json", "phase3_results.json"),
        (RUN / "eq7_results_v2.csv", "eq7_results_v2.csv"),
        (RUN / "scores/retest_correlation.json", "retest_correlation.json"),
        (RUN / "coverage_by_year_class_v2.csv", "coverage_by_year_class_v2.csv"),
        (RUN / "training_coverage_per_asset_v2.csv", "training_coverage_per_asset_v2.csv"),
        (RUN / "panel_definitions.csv", "panel_definitions.csv"),
        (RUN / "realized_weekly_N_v2.csv", "realized_weekly_N_v2.csv"),
        (RUN / "usable_cross_section_per_week.csv", "usable_cross_section_per_week.csv"),
    ]
    for src, dst in files:
        shutil.copy(src, HERE / dst)
    # residuals (expanding-window) and per-arm excess-AFSI series
    for arm in ["P0", "P3"]:
        for sn in ["C2x", "C3x"]:
            shutil.copy(RUN / f"phase3x/residuals_{arm}_{sn}.parquet",
                        HERE / f"residuals_{arm}_{sn}.parquet")
            shutil.copy(RUN / f"phase3x/excess_afsi_{arm}_{sn}.csv",
                        HERE / f"excess_afsi_{arm}_{sn}.csv")
    print(f"copied {len(files)} artifacts + residuals + excess-AFSI series")


def raw_io_manifest():
    jsonls = sorted(glob.glob(str(RUN / "scores" / "raw_io_*.jsonl")))
    entries, total = [], 0
    sample = None
    for jl in jsonls:
        n = sum(1 for _ in open(jl))
        total += n
        entries.append({"file": Path(jl).name, "n_records": n})
        if sample is None:
            with open(jl) as f:
                rec = json.loads(f.readline())
            # response-only sample: raw prompt withheld (embeds licensed news text)
            sample = {"arm": rec.get("arm"), "date": rec.get("date"),
                      "asset_ticker": rec.get("asset_ticker"), "model": rec.get("model"),
                      "call_kind": rec.get("call_kind"), "response": rec.get("response"),
                      "usage": rec.get("usage"),
                      "template_idx": rec.get("template_idx"),
                      "price_window": rec.get("price_window"),
                      "lag_weeks": rec.get("lag_weeks"),
                      "n_headlines_kept": rec.get("n_headlines_kept")}
    manifest = {
        "description": "Raw LLM input/output logs for the SYNCRISK main run. The full "
                       "raw_io_*.jsonl files (87,845 records) are NOT shipped because the "
                       "request `messages` embed licensed EODHD news headlines. This "
                       "manifest gives per-file record counts and a single response-only "
                       "sample (prompt withheld). The parsed model outputs themselves ship "
                       "in full in scores_P0.parquet / scores_P3.parquet.",
        "total_records": total,
        "files": entries,
        "sample_record_response_only": sample,
    }
    json.dump(manifest, open(HERE / "raw_io_manifest.json", "w"), indent=2)
    print(f"raw_io_manifest.json: {total} records across {len(entries)} files")


if __name__ == "__main__":
    info = consolidate_scores()
    consolidate_frontier_scores()
    copy_phase3x9()
    panel_features()
    scm_series(info)
    macro()
    copy_artifacts()
    raw_io_manifest()
    print("DONE building data/published/")
