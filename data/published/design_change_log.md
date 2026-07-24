# SYNCRISK Main Run — Design Change Log (v0.9 amendments)

## Amendment 1 — validation rule (2026-06-12, approved Phase-0 response)
The frozen rule "drop asset-weeks with missing news; **abort if >10% of pairs are
lost**" was calibrated on the news-rich pilot panel (20 US large-caps, 2022,
≥98% coverage) and does NOT transfer to the full 80/74-asset panel spanning
2018–2026 (which includes FX/crypto/commodities/bonds and the sparse-news
2018–2020 period). The abort threshold is therefore **replaced by a measurement
gate (Phase 1.5)**: build the panel, measure coverage, and STOP for an explicit
panel-definition decision (D1/D2/D3) before any scoring. No abort on coverage.

## Note — panel size (logged)
Frozen design v0.9 states "80 assets". `config/tickers.yaml` (the full panel
config) enumerates **74 assets** (30 US single + 11 sector ETF + 10 global index
+ 8 commodity + 8 FX + 5 bond + 2 crypto); its own notes say "to reach 80, add 6
more — never added". The main run uses the 74 assets actually in the config; the
74-vs-80 gap is recorded here as a spec/config discrepancy, not a silent change.

## Amendment 2 — panel-selection rule (2026-06-12, fixed BEFORE seeing v2 coverage)
Approved at the Phase 1.5 decision. Applied to the year-chunked-news rebuilt data.
- **Asset filter:** keep assets with **≥60% usable training weeks (2019-02..2021-12)
  AND ≥70% usable weeks overall**. **US equities only.**
  - Scope interpretation (logged): "US equities" = categories `us_single` (30
    single-name US stocks) + `us_sector_etf` (11 US sector/broad equity ETFs).
    EXCLUDED: `global_index` (non-US-underlying equity ETFs, e.g. EWJ/EZU/EWG),
    `commodity`, `bond_etf`, `fx`, `crypto`.
  - Reframed justification: the frozen feature set (beta-to-SPY, equity RV/drawdown
    on the US trading calendar) is equity-calibrated; restricting to US equities
    preserves cross-sectional comparability and the validity of the frozen
    standardization moments and cleaning fits. FX/crypto are out of scope (calendar
    mismatch breaks beta-to-SPY); commodities/bonds/non-US index ETFs are excluded
    for feature comparability.
- **Week filter:** keep weeks with usable cross-section **N ≥ 30** *after* the asset
  filter (usable = passes frozen essentials AND news present).
- This rule is FIXED prior to computing v2 numbers.

## Amendment 3 — year-chunked news fetch (2026-06-12)
`news_all_pages` previously paginated newest-first to a 50k cap over the whole
2018-2026 range, truncating early-period news for the highest-volume names. Replaced
with **per-calendar-year fetching** (each year fetched as its own from/to window,
paginated within-year), so full history is recovered. Data-pipeline fix only; no
equation/feature/criterion change.

## Amendment 4 — threshold amendment (2026-06-12, "Amendment 3" in the decision text)
Justification (ex ante, not result-driven): the rule fixed in Amendment 2 proved
**internally inconsistent** on the year-chunked rebuilt data (the asset filter
admitted only 13 assets, so the N≥30 week rule could never be met → empty panel).
**No LLM score exists yet**, so revising the sample definition here cannot be driven
by results — this is the distinction from p-hacking. Revised BEFORE any elicitation:
- **LOCKED thresholds:** assets `pct_train ≥ 40%` AND `pct_overall ≥ 70%`;
  weeks `N ≥ 30` (usable cross-section after asset filter).
- **Absolute floor:** every included asset must have `≥ 50 usable TRAINING weeks`
  (2019-02..2021-12), not merely the 40% share — Eq. 1 standardization moments and
  the Eq. 5 tail thresholds c_k need a minimum absolute sample.
- US equities only (us_single + us_sector_etf), as in Amendment 2.

## Amendment 5 — pre-registered robustness subsample (2026-06-12)
All headline results will ALSO be recomputed on the **13-asset high-coverage core**
(`pct_train ≥ 60` AND `pct_overall ≥ 70`) as an appendix subsample check. The core
is too small for weekly AFSI *levels*, but asset-level residual correlations are
computable and should agree with the full realized panel. Pre-registered here
before scoring.

## Note — realized panel scope (for the eventual v0.9 → v1.0 design update)
The realized panel is "~30 US large/mid-caps with dense news coverage, ~2021–2026",
NOT the v0.9 "80 assets, 2018–2026". A consolidated v1.0 design update (realized
panel + Amendments 1–5) will be made before manuscript drafting so the design
document matches what actually ran. Numbers first.

## Amendment 6 — cleaning revision (expanding window), post-scoring (2026-06-12)
Trigger: the **pre-specified orthogonality diagnostic FAILED**. C2 was defined in the
pilot by residual-on-observables R² ≈ 0.00; on the main run it is **0.17–0.30** per
model — the under-cleaning pathology the cleaning ladder was meant to remove is back.
Root cause: the frozen cleaning/standardization/tail fits used **only 48 training
weeks, all from the calm 2021 regime** (the realized panel's kept weeks in 2019-2021),
so fits do not generalize to 2022–2026 and OOS residuals stay loaded on observables.

**GATE BREACH (logged honestly):** the conditional-approval condition "≥50 usable
TRAINING weeks per asset" was checked against *news-usable* training weeks (min 65 →
reported PASS), but the quantity that actually binds Eq.1 moments and the cleaning
fit is the number of *scored/kept* training weeks = **48 < 50**. The gate should have
stopped there. No elicitation is redone (scores are valid); the fix is post-processing.

Revision (recomputation only, no new elicitation):
- **C2x / C3x = expanding-window** versions of C2 / C3 (same feature sets). For each
  week t, fit on ALL weeks < t (minimum 48 weeks of history before the first OOS
  residual), **refit quarterly (~13 weeks)**. Strictly no contemporaneous/future data
  in any fit. Standardization moments and tail thresholds c_k use the same expanding
  scheme. Uses the full sample for estimation, stays strictly OOS, covers both regimes.
- Frozen-2021 C2/C3 retained as an APPENDIX column for transparency.
- New diagnostics: C2x residual R² full-OOS AND split normal/stress (target <0.05 in
  both); mechanical-premium diagnostic (per-week cleaning-error magnitude vs per-week
  Excess-AFSI across stress weeks). Output: main_run_report_v2.md.
This is triggered by a failed pre-specified diagnostic, not by dislike of results —
the distinction that keeps the post-scoring revision honest.

## Amendment 7 — realized-design reconciliation (2026-06-13, recorded before the frontier arm)
Records, as a permanent correction, the discrepancies between concept-note v10's
framing and what the main run actually executed (flagged as \todo{RECONCILE} in the
draft). These are corrections to the design record, not to results:
- **Commercial models were EFFICIENCY-tier, not frontier.** Realized strings:
  `claude-haiku-4-5-20251001`, `gpt-4o-mini`, `gemini-2.5-flash`. v10 §2.1's
  "Claude Sonnet 4.6 / GPT-5.4 / Gemini 3.5 Flash, frontier-tier" was an aspirational
  framing error (author's, to be owned in v1.1). The existing 3 commercial models are
  hereafter labelled the **efficiency tier**.
- **Open-weight Mistral substitution:** realized `Mistral-Small-24B-Instruct-2501-AWQ`
  (no ungated AWQ of 3.1/2503 without an HF token); Qwen3-32B-AWQ and
  gemma-3-27b-it (W4A16) as logged.
- **Panel:** realized 30 US equities × 244 weeks (2021-02..2026-06), not 80×436/2018-26.
- **Eq.7 controls:** realized W_t = {vix, hy_spread, term_slope, market_return}
  (v10 listed NFCI+WTI; not used).
The v1.1 design record (author) will carry these on the record.

## Amendment 8 — frontier-tier arm (2026-06-13)
Extends the main run with 3 frontier-tier commercial models on the IDENTICAL realized
panel and the IDENTICAL logged P3 perturbation draws per cell (clean tier contrast, no
perturbation variance confounded with tier). No other change to the frozen design.
Phase 0 cost is a mandatory pause; conditional pre-approval if projected total <= $300.

## Amendment 8 addendum — gemini-2.5-pro thinking (approved 2026-06-13, $249.49 / guard $375)
gemini-2.5-pro MANDATES thinking (rejects thinking_budget=0). Run in its native thinking
mode (max_output_tokens raised to 2048; temperature 0). This is a model property, not a
config choice: Google's frontier endpoint reasons before answering and cannot be run
otherwise; substituting the older non-thinking gemini-1.5-pro would put a model weaker
than the efficiency tier in the frontier slot. Revised projection $249.49; cost guard
raised to $375 (1.5x). Logged deviations and caveats:
1. **Google cross-tier pair caveat.** Sonnet–Haiku and gpt-4.1–gpt-4o-mini isolate
   scale/lineage at EQUAL configuration. The Gemini pro–flash pair CONFOUNDS scale with
   thinking (efficiency 2.5-flash ran with thinking disabled — the pilot fix; frontier
   2.5-pro runs with thinking on). The Gemini cross-tier contrast is reported WITH this
   note, not hidden.
2. **Pre-specified robustness (free).** Every frontier-block statistic is reported on
   frontier-2 (Claude+GPT, no Gemini) ALONGSIDE frontier-3, so any dependence of the
   tier conclusions on the single thinking-mode model is visible immediately.
Recorded prediction (4th, ex ante): if thinking yields more idiosyncratic reasoning,
gemini-2.5-pro should be LESS correlated with the others than claude/gpt frontier are
with each other (a crude proxy for "does reasoning break monoculture?"); the opposite
result — thinking yet still convergent — would be more telling still.

## Amendment 8a — frontier projection error + asymmetric design (2026-06-13)
During the frontier run the per-model cost revealed a Phase-0 projection ERROR:
gemini-2.5-pro was projected over ONE arm (7,320 cells) instead of both (14,640),
understating it by ~$135; true full-arm cost ≈ $383, breaching the $300 ceiling and
the $375 guard. The run was STOPPED at $238.93 (before gemini-2.5-pro P3) rather than
spending into the guard. Decision (approved): finish gpt-frontier P3 only (~$8), leave
gemini-frontier at P0-complete. Resulting **asymmetric design, declared plainly (not
cherry-picked — gemini-frontier P3 dropped on cost discipline)**:
- **P0 (identical inputs): all 9 models** — efficiency {claude,gpt,gemini} + frontier
  {claude,gpt,gemini}-frontier + open {qwen3-32b,mistral-24b,gemma-3-27b}. Carries the
  primary tier contrast, the same-provider lineage pairs (incl. gemini pro–flash), and
  the reasoning-vs-monoculture test.
- **P3 (perturbed): 8 models** — efficiency-3 + frontier-2 {claude,gpt}-frontier +
  open-3. Frontier-2 was already the pre-specified Amendment-8 robustness set.
- Every report table states its model set in the header; P0-9 and P3-8 figures are
  never mixed silently. gemini-frontier P3 (the thinking-confounded, asterisked cell)
  is the dropped piece — the least interpretable cell, by cost discipline.
Total frontier spend ≈ $247. v1.1 design record (author) will carry both projection
errors and the asymmetric design.
