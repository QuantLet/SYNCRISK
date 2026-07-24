# SYNCRISK Quantlets

Replication code for **"SYNCRISK: A Label-Free Audit of Effective Diversity in AI Risk-Scoring Panels"** (Daniel Traian Pele).

SYNCRISK measures **residual synchronization (Excess-AFSI)** among LLM-based financial-risk scores — the cross-model agreement that survives after shared reaction to common observable information is removed out of sample. Each Quantlet below reproduces one reported table or figure from the released, B-stabilized results files. Every Quantlet ships a `.py` script, a `.ipynb` notebook, a `.csv` of its output, and (where the deliverable is a figure) a `.png`.

| Quantlet | Reproduces | Output |
|---|---|---|
| `SYNCRISKdecay` | Decay chain (Table 3, Figure 2): Raw → C1x → C2x → C3x by arm | csv, png |
| `SYNCRISKladder` | Cleaning-ladder specification and orthogonality summary | csv |
| `SYNCRISKfamily` | Family-block decomposition (within-commercial / within-open / cross-family) | csv |
| `SYNCRISKstress` | Stress diagnostic and stress-minus-normal premium (upper bounds) | csv |
| `SYNCRISKcore` | High-coverage 13-asset core robustness check | csv |
| `SYNCRISKtier` | Exploratory frontier-tier extension (Appendix G) | csv |
| `SYNCRISKeq7` | Descriptive co-movement projection — the market-link null | csv |
| `SYNCRISKoverlap` | Effective diversity as a curve: K_eff(J) from the matched-overlap sweep, with the empirical panel (J=0.57, K_eff=2.1) marked on it | csv, png |
| `SYNCRISKtimeseries` | Per-week Excess-AFSI time series | csv, png |
| `SYNCRISKsynthetic` | Full pipeline end-to-end on a **synthetic panel — no licensed data** | csv |
| `SYNCRISKpanel` | Realized panel construction and coverage (30 US equities × 244 weeks) | csv, png |

## Layout (self-contained)

**Every Quantlet folder is independently self-contained**: it ships its own input data, code, and results side by side, and runs on its own even if copied out of this bundle. Each `quantlets/SYNCRISK*/` directory holds:

- the `.py` script and matching `.ipynb` notebook (**code**),
- the input data file(s) it reads (**data**) — bundled locally, so no `../../data` reach-out,
- the `.csv`/`.png` it produces (**results**),
- the two pipeline Quantlets (`SYNCRISKtimeseries`, `SYNCRISKsynthetic`) also carry a local copy of the `syncrisk/` library they import.

```
Quantlets/
├── README.md, Metainfo.txt
├── src/syncrisk/          canonical index/cleaning library (dev source of truth)
├── data/published/        canonical released, B-stabilized result files (provenance)
├── data/synthetic/        canonical synthetic-panel generator (no licensed inputs)
└── quantlets/SYNCRISK*/   the eleven Quantlets, each self-contained:
                           script + notebook + its input data + csv/png output
```

The top-level `src/` and `data/` remain the canonical source of truth; each Quantlet folder carries its own copy of exactly the files it needs, so a reviewer can download a single Quantlet folder and run it in place.

To run one: `cd quantlets/SYNCRISKdecay && python SYNCRISKdecay.py` (Python 3, pandas/numpy/scikit-learn/lightgbm). Each script reads its input data from its own folder and writes its `.csv`/`.png` output beside itself.

## Reproducibility

- All Quantlets render from the released results files bundled under `data/published/` (e.g. `upg_Bstab_results.json`); reproduction requires **no model calls and no licensed data**.
- `SYNCRISKsynthetic` runs the entire standardization → cleaning → Excess-AFSI pipeline on synthetic scores, so it is fully self-contained and is the recommended entry point for reviewers.
- The underlying price/news inputs are from a single commercial feed (EOD Historical Data) and are licensed, so they are not redistributed; the package ships the elicited and derived files needed to reproduce every reported number.
- Fixed seeds throughout (`random_state = 0`).

## Status

The repository is withheld during double-blind review. On acceptance the collection is deposited on the QuantLet platform with persistent identifiers, with accompanying slides on Quantinar.
