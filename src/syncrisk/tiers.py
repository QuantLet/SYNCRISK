"""
Tier x family decomposition for the 9-model SYNCRISK evaluation (Amendment 8).

The frontier-tier arm adds three frontier-tier commercial twins on the IDENTICAL
realized panel and the IDENTICAL logged P3 perturbation draws as their efficiency
twins (perturb.PERT_KEY draw-reuse), so a tier contrast is not confounded with
perturbation variance. The design is ASYMMETRIC (Amendment 8a):

  - P0 (identical inputs):   9 models = efficiency-3 + frontier-3 + open-3
  - P3 (perturbed):          8 models = efficiency-3 + frontier-2 {claude,gpt} + open-3
                             (gemini-frontier P3 dropped on cost discipline)

Every consumer must label its model set; P0-9 and P3-8 figures are never mixed.

This module computes Excess-AFSI over explicit, named pair-sets (tier blocks and
same-provider lineage pairs) so the within-frontier / within-efficiency / within-open
contrasts and the lineage channel are reproduced exactly as in phase3x9_results.json.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

# Tier rosters ---------------------------------------------------------------
EFFICIENCY = ["claude", "gpt", "gemini"]                       # existing commercial tier
FRONTIER = ["claude-frontier", "gpt-frontier", "gemini-frontier"]
OPEN = ["qwen3-32b", "mistral-24b", "gemma-3-27b"]
FRONTIER_2 = ["claude-frontier", "gpt-frontier"]               # pre-specified robustness set

TIER = {**{m: "efficiency" for m in EFFICIENCY},
        **{m: "frontier" for m in FRONTIER},
        **{m: "open" for m in OPEN}}
FAMILY = {**{m: "commercial" for m in EFFICIENCY},
          **{m: "commercial" for m in FRONTIER},
          **{m: "open" for m in OPEN}}

# Frontier twins reuse their efficiency twin's perturbation draws (perturb.PERT_KEY).
PERT_KEY = {"claude-frontier": "claude", "gpt-frontier": "gpt", "gemini-frontier": "gemini"}
# Same-provider cross-tier (lineage) twin pairs.
LINEAGE_PAIRS = [("claude-frontier", "claude"), ("gpt-frontier", "gpt"),
                 ("gemini-frontier", "gemini")]

# Pair-set layout (order matters for table rows). Tier blocks + frontier-2 robustness
# + same-provider lineage pairs.
PAIRSET_LABELS = {
    "all9": "all-9",
    "within_efficiency": "within-efficiency",
    "within_frontier": "within-frontier",
    "within_open": "within-open",
    "frontier2": "frontier-2 (claude+gpt)",
    "lineage_same_provider_cross_tier": "lineage (same-provider cross-tier)",
}


def pairs_within(models):
    """All unordered pairs within a roster."""
    return list(itertools.combinations(models, 2))


def pairsets(models_present=None):
    """The named pair-sets. If `models_present` is given (e.g. the 8 models of arm
    P3), pairs referencing absent models are dropped so the same dict works for the
    asymmetric P0-9 / P3-8 design."""
    ps = {
        "all9": pairs_within(EFFICIENCY + FRONTIER + OPEN),
        "within_efficiency": pairs_within(EFFICIENCY),
        "within_frontier": pairs_within(FRONTIER),
        "within_open": pairs_within(OPEN),
        "frontier2": pairs_within(FRONTIER_2),
        "lineage_same_provider_cross_tier": list(LINEAGE_PAIRS),
    }
    if models_present is not None:
        present = set(models_present)
        ps = {k: [(a, b) for (a, b) in v if a in present and b in present] for k, v in ps.items()}
    return ps


def afsi_over_pairs(long_df, value_col, name_pairs):
    """Per-date mean cross-sectional correlation over an explicit list of
    (modelA, modelB) name pairs. Returns a [date, afsi] frame.

    Mirrors the main-run phase3x9 computation: assets with any NaN across the
    paired models are dropped per date; a pair needs >= 5 overlapping assets."""
    recs = []
    for date, g in long_df.dropna(subset=[value_col]).groupby("date"):
        wide = g.pivot_table(index="asset_ticker", columns="model", values=value_col).dropna(axis=0, how="any")
        cols = set(wide.columns)
        cs = []
        for a, b in name_pairs:
            if a in cols and b in cols and wide.shape[0] >= 5:
                c = np.corrcoef(wide[a].to_numpy(), wide[b].to_numpy())[0, 1]
                if np.isfinite(c):
                    cs.append(c)
        recs.append({"date": date, "afsi": float(np.mean(cs)) if cs else np.nan})
    return pd.DataFrame(recs).sort_values("date")


def level_by_regime(series_df, stress):
    """Mean AFSI on normal vs stress weeks for a [date, afsi] series."""
    mm = series_df.merge(stress, on="date", how="left")
    mm["stress"] = mm["stress"].fillna(False).astype(bool)
    return {"normal": round(float(mm.loc[~mm["stress"], "afsi"].mean()), 4),
            "stress": round(float(mm.loc[mm["stress"], "afsi"].mean()), 4)}


def levels_over_pairsets(long_df, value_col, stress, models_present=None):
    """Excess/Raw-AFSI normal/stress level for every named pair-set."""
    ps = pairsets(models_present)
    return {name: level_by_regime(afsi_over_pairs(long_df, value_col, pl), stress)
            for name, pl in ps.items()}
