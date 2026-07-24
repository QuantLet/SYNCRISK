"""
SYNCRISK — Measuring Excess Forecast Synchronization Across Large Language Models.

Replication package for: "Measuring Excess Forecast Synchronization Across Large
Language Models in Financial Risk Assessment."

Subpackages / modules
----------------------
indices      : Eq.1 standardization, Eq.2 Raw-AFSI, Eq.3/4 cleaning + Excess-AFSI,
               Eq.5 Tail-AFSI, permutation test.
cleaning     : MAIN expanding-window cleaning (C2x / C3x, Amendment 6) + feature sets.
outcomes     : Eq.6 Systemic Co-Movement (SCM).
families     : 6-model family decomposition + per-date AFSI series helpers.
regression   : Newey-West HAC OLS for the descriptive Eq.7 projection.
perturbation : prompt/information perturbation arms (P0..P3).
prompts      : risk-elicitation prompt templates.
loaders      : raw-data loaders (EODHD, FRED) — only for reconstruction.
"""
__version__ = "1.0.0"
