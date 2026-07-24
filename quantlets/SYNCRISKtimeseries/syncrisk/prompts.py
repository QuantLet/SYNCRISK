"""
SYNCRISK pilot prompts — neutral risk-assessment prompt with two information sets.

Design principles
-----------------
- Single neutral prompt regime in pilot (prompt variations only in robustness).
- Two information sets: prices_only and prices_plus_news.
- LLM outputs are RELATIVE RISK BELIEFS, not calibrated probabilities. The schema
  enforces this by asking for a 0-100 score and a 1-10 rank, not probabilities.
- Strict JSON output. A repair prompt is provided for retries.
- Deterministic generation settings (temperature=0) set in client code, not here.
- Same prompt body across all LLMs to ensure the dependence structure of outputs
  reflects model differences, not prompt differences.

Schema
------
{
  "relative_risk_score": int 0-100,
  "tail_risk_rank": int 1-10,
  "de_risking_signal": "increase" | "hold" | "reduce",
  "confidence": float 0-1,
  "brief_reason": str (<= 30 words)
}
"""

from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Literal


SYSTEM_PROMPT = """You are a quantitative financial risk analyst. \
You assess RELATIVE risk across a panel of assets on a given date. \
You produce structured JSON outputs only. You never produce calibrated probability \
forecasts; you produce relative risk rankings within a panel.

Rules:
1. Score every asset on the same fixed scale, not relative to itself over time.
2. Use the full 0-100 range when justified; do not cluster scores near the middle.
3. Return strict JSON with no commentary, no markdown, no code fences.
4. If information is sparse for an asset, lower the confidence field; do not refuse.
5. "brief_reason" is at most 30 words. No exceptions.
"""


PROMPT_PRICES_ONLY = """\
DATE: {date}
ASSET: {asset_ticker} ({asset_name})

PRICE-BASED FEATURES (most recent observations, computed at the close before DATE):
- 1-day return:                {ret_1d:+.4f}
- 5-day cumulative return:     {ret_5d:+.4f}
- 20-day cumulative return:    {ret_20d:+.4f}
- 20-day realized volatility:  {rv_20d:.4f}  (annualized)
- 60-day realized volatility:  {rv_60d:.4f}  (annualized)
- 60-day max drawdown:         {dd_60d:.4f}  (most negative cumulative loss)
- 20-day min return:           {min_ret_20d:+.4f}
- Beta to market (60d):        {beta_60d:+.3f}
- Z-score of 20d RV vs 1y:     {rv_zscore:+.3f}

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


PROMPT_PRICES_PLUS_NEWS = PROMPT_PRICES_ONLY.replace(
    "TASK",
    """RECENT FINANCIAL NEWS (last 7 days, top {n_headlines} headlines by relevance):
{headlines_block}

NEWS TONE (GDELT-derived, range approximately -10 to +10):
- Mean tone, last 7 days:  {tone_7d:+.3f}
- Tone change vs prior 7d: {tone_delta:+.3f}

TASK""",
)


REPAIR_PROMPT = """\
Your previous response was not valid JSON or did not match the required schema. \
The schema is:
{{
  "relative_risk_score": <int 0-100>,
  "tail_risk_rank": <int 1-10>,
  "de_risking_signal": <"increase" | "hold" | "reduce">,
  "confidence": <float 0-1>,
  "brief_reason": <string, max 30 words>
}}
Return ONLY the corrected JSON object, with no surrounding text, no markdown, \
and no code fences. Do not explain. Do not apologize.

Your previous response was:
{previous_response}
"""


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

InformationSet = Literal["prices_only", "prices_plus_news"]


@dataclass
class AssetFeatures:
    """Feature bundle for one (asset, date) pair.

    All return/volatility values are decimal (e.g. 0.0123 for 1.23%), not percentages.
    Volatility values are annualized using sqrt(252).
    """
    date: str                       # ISO date string, e.g. "2024-05-10"
    asset_ticker: str               # e.g. "AAPL"
    asset_name: str                 # e.g. "Apple Inc."
    ret_1d: float
    ret_5d: float
    ret_20d: float
    rv_20d: float                   # annualized
    rv_60d: float                   # annualized
    dd_60d: float                   # most negative cumulative return over 60d window
    min_ret_20d: float
    beta_60d: float
    rv_zscore: float                # z-score of 20d RV vs trailing 1y distribution
    # News-set additions (optional; required only for prices_plus_news)
    headlines: list[str] | None = None
    tone_7d: float | None = None
    tone_delta: float | None = None


def build_user_prompt(
    features: AssetFeatures,
    info_set: InformationSet,
    max_headlines: int = 10,
) -> str:
    """Render the user-facing prompt for one (asset, date, info_set) tuple."""
    if info_set == "prices_only":
        return PROMPT_PRICES_ONLY.format(
            date=features.date,
            asset_ticker=features.asset_ticker,
            asset_name=features.asset_name,
            ret_1d=features.ret_1d,
            ret_5d=features.ret_5d,
            ret_20d=features.ret_20d,
            rv_20d=features.rv_20d,
            rv_60d=features.rv_60d,
            dd_60d=features.dd_60d,
            min_ret_20d=features.min_ret_20d,
            beta_60d=features.beta_60d,
            rv_zscore=features.rv_zscore,
        )

    if info_set == "prices_plus_news":
        if features.headlines is None or features.tone_7d is None:
            raise ValueError(
                f"prices_plus_news requires headlines and tone_7d; got "
                f"headlines={features.headlines!r}, tone_7d={features.tone_7d!r}"
            )
        hl = features.headlines[:max_headlines]
        headlines_block = (
            "\n".join(f"  - {h.strip()}" for h in hl) if hl else "  (no headlines available)"
        )
        return PROMPT_PRICES_PLUS_NEWS.format(
            date=features.date,
            asset_ticker=features.asset_ticker,
            asset_name=features.asset_name,
            ret_1d=features.ret_1d,
            ret_5d=features.ret_5d,
            ret_20d=features.ret_20d,
            rv_20d=features.rv_20d,
            rv_60d=features.rv_60d,
            dd_60d=features.dd_60d,
            min_ret_20d=features.min_ret_20d,
            beta_60d=features.beta_60d,
            rv_zscore=features.rv_zscore,
            n_headlines=len(hl),
            headlines_block=headlines_block,
            tone_7d=features.tone_7d,
            tone_delta=features.tone_delta if features.tone_delta is not None else 0.0,
        )

    raise ValueError(f"unknown info_set: {info_set!r}")


# ---------------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------------

VALID_SIGNALS = {"increase", "hold", "reduce"}


def parse_and_validate(response_text: str) -> dict:
    """Parse model output and validate against schema. Raises ValueError on failure.

    Tolerates:
    - leading/trailing whitespace
    - a stray code fence ```json ... ```
    Does NOT tolerate extra prose around the JSON; that triggers a repair retry.
    """
    text = response_text.strip()
    # Strip code fences if the model added them despite instructions.
    if text.startswith("```"):
        # remove first fence line and last fence line
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"not valid JSON: {e}") from e

    required = {"relative_risk_score", "tail_risk_rank", "de_risking_signal",
                "confidence", "brief_reason"}
    missing = required - obj.keys()
    if missing:
        raise ValueError(f"missing fields: {sorted(missing)}")

    rrs = obj["relative_risk_score"]
    if not isinstance(rrs, (int, float)) or not (0 <= rrs <= 100):
        raise ValueError(f"relative_risk_score out of range: {rrs!r}")
    obj["relative_risk_score"] = int(round(rrs))

    trr = obj["tail_risk_rank"]
    if not isinstance(trr, (int, float)) or not (1 <= trr <= 10):
        raise ValueError(f"tail_risk_rank out of range: {trr!r}")
    obj["tail_risk_rank"] = int(round(trr))

    sig = str(obj["de_risking_signal"]).strip().lower()
    if sig not in VALID_SIGNALS:
        raise ValueError(f"de_risking_signal invalid: {sig!r}")
    obj["de_risking_signal"] = sig

    conf = obj["confidence"]
    if not isinstance(conf, (int, float)) or not (0 <= conf <= 1):
        raise ValueError(f"confidence out of range: {conf!r}")
    obj["confidence"] = float(conf)

    reason = str(obj["brief_reason"]).strip()
    if len(reason.split()) > 40:    # tolerate a small overshoot before rejecting
        raise ValueError(f"brief_reason too long: {len(reason.split())} words")
    obj["brief_reason"] = reason

    return obj


# ---------------------------------------------------------------------------
# Few-shot example (used internally to anchor cross-model consistency;
# included as an assistant turn before the real ask).
# ---------------------------------------------------------------------------

FEWSHOT_USER = """\
DATE: 2020-03-12
ASSET: SPY (SPDR S&P 500 ETF)

PRICE-BASED FEATURES (most recent observations, computed at the close before DATE):
- 1-day return:                -0.0750
- 5-day cumulative return:     -0.1430
- 20-day cumulative return:    -0.1920
- 20-day realized volatility:  0.6500  (annualized)
- 60-day realized volatility:  0.4200  (annualized)
- 60-day max drawdown:         -0.2700
- 20-day min return:           -0.0750
- Beta to market (60d):        +1.000
- Z-score of 20d RV vs 1y:     +4.500

TASK
Produce a RELATIVE risk assessment for this asset, on the panel-wide scale used in this batch. Return strict JSON matching the schema below. No commentary.

SCHEMA
{
  "relative_risk_score": <int 0-100, higher = riskier>,
  "tail_risk_rank": <int 1-10, higher = more tail risk>,
  "de_risking_signal": <"increase" | "hold" | "reduce">,
  "confidence": <float 0-1>,
  "brief_reason": <string, max 30 words>
}"""

FEWSHOT_ASSISTANT = (
    '{"relative_risk_score": 92, "tail_risk_rank": 10, '
    '"de_risking_signal": "reduce", "confidence": 0.85, '
    '"brief_reason": "Extreme realized volatility (4.5 sigma above 1y), '
    'cascading drawdown, large single-day loss; classic crash regime."}'
)
