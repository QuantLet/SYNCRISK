"""
Perturbation arm for the SYNCRISK pilot (concept note v0.8).

Factorial on the SAME 20 assets x 50 weeks x 6 models:
  P0 = baseline (reuse existing scores, do NOT re-query)
  P1 = prompt paraphrase only      (3 paraphrase templates, family-rotated)
  P2 = information perturbation only (70% news subsample per model-cell;
       per-model price window in {60,90} days; per-model 0/1-week feature lag)
  P3 = P1 + P2

Template rotation: 6 models / 3 templates cannot be globally distinct in a cell,
but the 3 models WITHIN each family (commercial / open) get 3 DISTINCT templates
per cell, rotating across cells so wording is decorrelated from model identity.

"fundamentals lag offset": the pilot information set has no fundamentals (prices +
news only). The lag is therefore applied as a per-model {0,1}-week offset on the
numeric price-feature snapshot — a faithful stand-in; true fundamentals-lag is a
main-run item. Logged in the report.

This module is stdlib + the prompts package only, so it imports in BOTH the
`syncrisk` (commercial) and `syncrisk-vllm` (open-weight) environments.
"""
from __future__ import annotations

import hashlib
from .prompts import SYSTEM_PROMPT, FEWSHOT_USER, FEWSHOT_ASSISTANT

COMMERCIAL = ["claude", "gpt", "gemini"]
OPEN = ["qwen3-32b", "mistral-24b", "gemma-3-27b"]
# Amendment 8: frontier-tier twins of the commercial models. They REUSE their
# efficiency twin's perturbation draws (template/window/lag/news subsample) via
# PERT_KEY, so the tier contrast is on identical perturbed information (no
# perturbation variance confounded with tier).
FRONTIER_COMMERCIAL = ["claude-frontier", "gpt-frontier", "gemini-frontier"]
PERT_KEY = {"claude-frontier": "claude", "gpt-frontier": "gpt", "gemini-frontier": "gemini"}
TIER = {**{m: "efficiency" for m in COMMERCIAL},
        **{m: "frontier" for m in FRONTIER_COMMERCIAL},
        **{m: "open" for m in OPEN}}
FAMILY = {**{m: "commercial" for m in COMMERCIAL},
          **{m: "commercial" for m in FRONTIER_COMMERCIAL},
          **{m: "open" for m in OPEN}}
FAMILY_RANK = {**{m: i for i, m in enumerate(COMMERCIAL)},
               **{m: i for i, m in enumerate(OPEN)}}

# Per-model draws for P2/P3, balanced 3/3 across the panel (documented in report).
MODEL_WINDOW = {"claude": 60, "gpt": 90, "gemini": 60,
                "qwen3-32b": 90, "mistral-24b": 60, "gemma-3-27b": 90}
MODEL_LAG = {"claude": 0, "gpt": 1, "gemini": 1,
             "qwen3-32b": 0, "mistral-24b": 1, "gemma-3-27b": 0}
NEWS_KEEP_FRAC = 0.70

# ---------------------------------------------------------------------------
# Prompt body templates. All share the same placeholders and present identical
# information; only the wording differs. {pw}/{rv_pw}/{dd_pw} carry the (possibly
# perturbed) price-window number and its volatility / drawdown values.
# ---------------------------------------------------------------------------
_FEATURE_BLOCK_BASELINE = """\
DATE: {date}
ASSET: {asset_ticker} ({asset_name})

PRICE-BASED FEATURES (most recent observations, computed at the close before DATE):
- 1-day return:                {ret_1d:+.4f}
- 5-day cumulative return:     {ret_5d:+.4f}
- 20-day cumulative return:    {ret_20d:+.4f}
- 20-day realized volatility:  {rv_20d:.4f}  (annualized)
- {pw}-day realized volatility:  {rv_pw:.4f}  (annualized)
- {pw}-day max drawdown:         {dd_pw:.4f}  (most negative cumulative loss)
- 20-day min return:           {min_ret_20d:+.4f}
- Beta to market (60d):        {beta_60d:+.3f}
- Z-score of 20d RV vs 1y:     {rv_zscore:+.3f}
"""

_NEWS_BLOCK_BASELINE = """\
RECENT FINANCIAL NEWS (last 7 days, top {n_headlines} headlines by relevance):
{headlines_block}

NEWS TONE (GDELT-derived, range approximately -10 to +10):
- Mean tone, last 7 days:  {tone_7d:+.3f}
- Tone change vs prior 7d: {tone_delta:+.3f}
"""

_TASK_BASELINE = """\
TASK
Produce a RELATIVE risk assessment for this asset, on the panel-wide scale used in \
this batch. Return strict JSON matching the schema below. No commentary.

SCHEMA
{{
  "relative_risk_score": <int 0-100, higher = riskier>,
  "tail_risk_rank": <int 1-10, higher = more tail risk>,
  "de_risking_signal": <"increase" | "hold" | "reduce">,
  "confidence": <float 0-1>,
  "brief_reason": <string, max 30 words>
}}
"""

# --- Paraphrase 1 ---
_FEATURE_BLOCK_P1 = """\
Assessment date: {date}
Instrument: {asset_ticker} — {asset_name}

QUANTITATIVE PRICE SIGNALS (as of the close prior to the assessment date):
- Return over the last day:        {ret_1d:+.4f}
- Cumulative return, trailing 5d:  {ret_5d:+.4f}
- Cumulative return, trailing 20d: {ret_20d:+.4f}
- Annualized realized vol (20d):   {rv_20d:.4f}
- Annualized realized vol ({pw}d):   {rv_pw:.4f}
- Worst drawdown over {pw}d:         {dd_pw:.4f}
- Lowest single-day return (20d):  {min_ret_20d:+.4f}
- 60-day market beta:              {beta_60d:+.3f}
- 20d-vol z-score vs trailing 1y:  {rv_zscore:+.3f}
"""
_NEWS_BLOCK_P1 = """\
HEADLINES FROM THE PAST WEEK (up to {n_headlines}, most relevant first):
{headlines_block}

SENTIMENT (GDELT tone, roughly -10..+10):
- Average tone over 7 days:        {tone_7d:+.3f}
- Shift versus the previous week:  {tone_delta:+.3f}
"""
_TASK_P1 = """\
YOUR TASK
Judge this asset's risk RELATIVE to the others scored in this batch, on the shared \
panel scale. Respond with strict JSON only — no prose.

REQUIRED JSON SCHEMA
{{
  "relative_risk_score": <int 0-100, higher = riskier>,
  "tail_risk_rank": <int 1-10, higher = more tail risk>,
  "de_risking_signal": <"increase" | "hold" | "reduce">,
  "confidence": <float 0-1>,
  "brief_reason": <string, max 30 words>
}}
"""

# --- Paraphrase 2 ---
_FEATURE_BLOCK_P2 = """\
[{date}] Risk review for {asset_ticker} ({asset_name}).

Recent price-derived metrics (measured at the prior close):
  * 1d return ......... {ret_1d:+.4f}
  * 5d return ......... {ret_5d:+.4f}
  * 20d return ........ {ret_20d:+.4f}
  * realized vol 20d .. {rv_20d:.4f} (ann.)
  * realized vol {pw}d .. {rv_pw:.4f} (ann.)
  * max drawdown {pw}d .. {dd_pw:.4f}
  * min 20d daily ret . {min_ret_20d:+.4f}
  * beta (60d) ........ {beta_60d:+.3f}
  * vol z-score (1y) .. {rv_zscore:+.3f}
"""
_NEWS_BLOCK_P2 = """\
News flow, trailing 7 days (top {n_headlines}):
{headlines_block}

Tone metrics (GDELT, ~-10..+10):
  * mean tone (7d) .... {tone_7d:+.3f}
  * tone delta (vs 7d) {tone_delta:+.3f}
"""
_TASK_P2 = """\
TASK.
Give a cross-sectional RELATIVE risk read for this asset on the batch's common \
scale. Output must be strict JSON and nothing else.

SCHEMA
{{
  "relative_risk_score": <int 0-100, higher = riskier>,
  "tail_risk_rank": <int 1-10, higher = more tail risk>,
  "de_risking_signal": <"increase" | "hold" | "reduce">,
  "confidence": <float 0-1>,
  "brief_reason": <string, max 30 words>
}}
"""

# --- Paraphrase 3 ---
_FEATURE_BLOCK_P3 = """\
Date of analysis: {date}
Security under review: {asset_ticker} ({asset_name})

The following price statistics were computed as of the close before the analysis date:
- one-day return is {ret_1d:+.4f};
- five-day cumulative return is {ret_5d:+.4f};
- twenty-day cumulative return is {ret_20d:+.4f};
- twenty-day annualized realized volatility is {rv_20d:.4f};
- {pw}-day annualized realized volatility is {rv_pw:.4f};
- the {pw}-day maximum drawdown is {dd_pw:.4f};
- the minimum twenty-day daily return is {min_ret_20d:+.4f};
- the sixty-day beta to the market is {beta_60d:+.3f};
- the z-score of twenty-day volatility against the past year is {rv_zscore:+.3f}.
"""
_NEWS_BLOCK_P3 = """\
Over the last seven days, the most relevant headlines (up to {n_headlines}) were:
{headlines_block}

GDELT-based sentiment (approximately -10 to +10):
- the mean tone over seven days is {tone_7d:+.3f};
- the change in tone relative to the prior week is {tone_delta:+.3f}.
"""
_TASK_P3 = """\
What follows is your task: produce a RELATIVE risk assessment for this security on \
the panel-wide scale shared by this batch. Reply with strict JSON only, no commentary.

The JSON must follow this schema:
{{
  "relative_risk_score": <int 0-100, higher = riskier>,
  "tail_risk_rank": <int 1-10, higher = more tail risk>,
  "de_risking_signal": <"increase" | "hold" | "reduce">,
  "confidence": <float 0-1>,
  "brief_reason": <string, max 30 words>
}}
"""

# template index 0 == baseline wording (used by P0/P2); 1..3 == paraphrases (P1/P3)
TEMPLATES = [
    (_FEATURE_BLOCK_BASELINE, _NEWS_BLOCK_BASELINE, _TASK_BASELINE),
    (_FEATURE_BLOCK_P1, _NEWS_BLOCK_P1, _TASK_P1),
    (_FEATURE_BLOCK_P2, _NEWS_BLOCK_P2, _TASK_P2),
    (_FEATURE_BLOCK_P3, _NEWS_BLOCK_P3, _TASK_P3),
]


def _rng_keep(headlines, model, date, asset, frac=NEWS_KEEP_FRAC):
    """Deterministically keep a `frac` subsample of headlines for (model,date,asset)."""
    if not headlines:
        return headlines
    h = hashlib.sha256(f"{model}|{date}|{asset}".encode()).digest()
    seed = int.from_bytes(h[:8], "big")
    n = len(headlines)
    k = max(1, int(round(n * frac)))
    # deterministic shuffle via seeded indices
    order = sorted(range(n), key=lambda i: hashlib.sha256(f"{seed}|{i}".encode()).hexdigest())
    keep = sorted(order[:k])
    return [headlines[i] for i in keep]


def template_index(arm, model, week_index, asset_index):
    """0 = baseline wording (P0/P2); 1..3 = paraphrase (P1/P3, family-rotated)."""
    if arm in ("P0", "P2"):
        return 0
    # within-family rotation across the 3 paraphrases (templates 1..3)
    return 1 + ((FAMILY_RANK[model] + week_index + asset_index) % 3)


def build_messages(arm, model, cur, prev, week_index, asset_index):
    """Return (messages, meta).

    cur/prev are dicts for the current / previous-week feature row of this asset.
    Required keys: date, asset_ticker, asset_name, ret_1d, ret_5d, ret_20d,
    rv_20d, rv_60d, rv_90d, dd_60d, dd_90d, min_ret_20d, beta_60d, rv_zscore,
    headlines (list|None), tone_7d, tone_delta.
    """
    # Frontier twins reuse their efficiency twin's perturbation draws (Amendment 8).
    base = PERT_KEY.get(model, model)
    info_pert = arm in ("P2", "P3")
    lag = MODEL_LAG[base] if info_pert else 0
    src = prev if (lag == 1 and prev is not None) else cur
    pw = MODEL_WINDOW[base] if info_pert else 60
    rv_pw = src["rv_90d"] if pw == 90 else src["rv_60d"]
    dd_pw = src["dd_90d"] if pw == 90 else src["dd_60d"]

    headlines = cur.get("headlines")
    if headlines is None:
        headlines = []
    elif not isinstance(headlines, list):
        headlines = list(headlines)
    if info_pert:
        headlines = _rng_keep(headlines, base, cur["date"], cur["asset_ticker"])
    hl = headlines[:10]
    headlines_block = ("\n".join(f"  - {str(h).strip()}" for h in hl)
                       if hl else "  (no headlines available)")

    tidx = template_index(arm, base, week_index, asset_index)
    feat_t, news_t, task_t = TEMPLATES[tidx]

    body = feat_t.format(
        date=cur["date"], asset_ticker=cur["asset_ticker"], asset_name=cur["asset_name"],
        ret_1d=src["ret_1d"], ret_5d=src["ret_5d"], ret_20d=src["ret_20d"],
        rv_20d=src["rv_20d"], pw=pw, rv_pw=rv_pw, dd_pw=dd_pw,
        min_ret_20d=src["min_ret_20d"], beta_60d=src["beta_60d"], rv_zscore=src["rv_zscore"],
    )
    news = news_t.format(
        n_headlines=len(hl), headlines_block=headlines_block,
        tone_7d=cur["tone_7d"] if cur.get("tone_7d") is not None else 0.0,
        tone_delta=cur["tone_delta"] if cur.get("tone_delta") is not None else 0.0,
    )
    user_prompt = body + "\n" + news + "\n" + task_t

    messages = [
        {"role": "user", "content": FEWSHOT_USER},
        {"role": "assistant", "content": FEWSHOT_ASSISTANT},
        {"role": "user", "content": user_prompt},
    ]
    meta = {"template_idx": tidx, "price_window": pw, "lag_weeks": lag,
            "n_headlines_kept": len(hl)}
    return messages, meta
