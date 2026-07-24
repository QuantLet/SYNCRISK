"""Expanding-window cleaning (Amendment 6) — C2x / C3x.

This is the MAIN cleaning specification of the SYNCRISK main run. It is the exact
logic used in the original `phase3x_compute.py`; reproduced here verbatim (only the
imports are made package-relative) so the replication does not re-derive it.

For each week t: fit the cleaning regression on ALL weeks < t (>= MIN_HISTORY weeks
of history before the first OOS residual), refitting every REFIT_EVERY weeks
(~quarterly). Standardization moments and tail thresholds use the same expanding
scheme. No contemporaneous/future data enters any fit; residuals are strictly OOS.

  C2x = expanding-window linear cleaning  (spec="lin")
  C3x = expanding-window LightGBM cleaning (spec="gbm")
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_HISTORY = 48
REFIT_EVERY = 13
RANDOM_STATE = 0


def expanding_residuals(z, panel, features, spec, *,
                        min_history=MIN_HISTORY, refit_every=REFIT_EVERY,
                        random_state=RANDOM_STATE):
    """Expanding-window residuals u with per-week fit-error magnitude.

    Parameters
    ----------
    z : DataFrame with columns [date, asset_ticker, model, info_set, Z]
        Standardized scores (Eq. 1).
    panel : DataFrame with [date, asset_ticker] + `features`
        Cross-sectional cleaning observables.
    features : list[str]
    spec : {"lin", "gbm"}
        "lin" -> C2x (OLS), "gbm" -> C3x (LightGBM).

    Returns
    -------
    DataFrame [date, asset_ticker, model, info_set, u, raw_resid] + features
        `u` is the within-date-demeaned residual (Eq. 3 cleaned score);
        `raw_resid` is the pre-demeaning residual (used for fit-error diagnostics).
    """
    m = z.merge(panel[["date", "asset_ticker"] + features],
                on=["date", "asset_ticker"], how="left")
    m = m.dropna(subset=["Z"] + features).copy()
    weeks = np.array(sorted(m["date"].unique()))
    refit_pts = list(range(min_history, len(weeks), refit_every))
    out = []
    for model, g in m.groupby("model"):
        g = g.copy()
        g["Zc"] = g["Z"] - g.groupby("date")["Z"].transform("mean")
        Xc = pd.DataFrame(
            {f: g[f] - g.groupby("date")[f].transform("mean") for f in features},
            index=g.index)
        raw = pd.Series(np.nan, index=g.index)
        for i, r in enumerate(refit_pts):
            train_dates = set(weeks[:r])
            hi = refit_pts[i + 1] if i + 1 < len(refit_pts) else len(weeks)
            apply_dates = set(weeks[r:hi])
            tr = g["date"].isin(train_dates)
            apm = g["date"].isin(apply_dates)
            if tr.sum() < 30 or apm.sum() == 0:
                continue
            if spec == "lin":
                beta, *_ = np.linalg.lstsq(Xc.loc[tr].to_numpy(),
                                           g.loc[tr, "Zc"].to_numpy(), rcond=None)
                pred = Xc.loc[apm].to_numpy() @ beta
            else:
                import lightgbm as lgb
                mdl = lgb.LGBMRegressor(
                    n_estimators=300, learning_rate=0.05, num_leaves=31,
                    min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                    random_state=random_state, verbosity=-1)
                mdl.fit(Xc.loc[tr].to_numpy(), g.loc[tr, "Zc"].to_numpy())
                pred = mdl.predict(Xc.loc[apm].to_numpy())
            raw.loc[apm] = g.loc[apm, "Zc"].to_numpy() - pred
        g["raw_resid"] = raw
        g = g.dropna(subset=["raw_resid"])
        g["u"] = g["raw_resid"] - g.groupby("date")["raw_resid"].transform("mean")
        out.append(g[["date", "asset_ticker", "model", "info_set", "u", "raw_resid"] + features])
    return pd.concat(out, ignore_index=True)


def resid_r2(resid, features, by_stress=None):
    """R2 of u on within-date-demeaned observables, per model; optionally per regime.

    The orthogonality diagnostic: under a clean residualization this should be ~0.
    `by_stress` is an optional [date, stress] frame to split into normal/stress.
    """
    def one(df):
        res = {}
        for mdl, g in df.groupby("model"):
            gg = g.dropna(subset=["u"] + features)
            if len(gg) < 50:
                res[mdl] = np.nan
                continue
            y = (gg["u"] - gg.groupby("date")["u"].transform("mean")).to_numpy()
            X = np.column_stack(
                [(gg[f] - gg.groupby("date")[f].transform("mean")).to_numpy() for f in features])
            Xi = np.column_stack([np.ones(len(y)), X])
            b, *_ = np.linalg.lstsq(Xi, y, rcond=None)
            r = y - Xi @ b
            sst = ((y - y.mean()) ** 2).sum()
            res[mdl] = round(float(1 - (r @ r) / sst), 4) if sst > 0 else np.nan
        return res
    if by_stress is None:
        return one(resid)
    rr = resid.merge(by_stress, on="date", how="left")
    rr["stress"] = rr["stress"].fillna(False).astype(bool)
    return {"normal": one(rr[~rr["stress"]]), "stress": one(rr[rr["stress"]])}


def expanding_tail_majority(z, families, groupings, *, quantile=0.90,
                            min_history=MIN_HISTORY, refit_every=REFIT_EVERY):
    """Tail-AFSI (Eq. 5, q=0.5 majority) with expanding-window per-model thresholds c_k.

    c_k = `quantile` of model k's standardized scores over weeks < t, refit quarterly.
    Returns a DataFrame with one row per date and a column per grouping.
    """
    z = z.dropna(subset=["Z"]).copy()
    weeks = np.array(sorted(z["date"].unique()))
    refit = list(range(min_history, len(weeks), refit_every))
    z["H"] = np.nan
    for i, r in enumerate(refit):
        train = set(weeks[:r])
        hi = refit[i + 1] if i + 1 < len(refit) else len(weeks)
        app = set(weeks[r:hi])
        ck = z[z["date"].isin(train)].groupby("model")["Z"].quantile(quantile)
        mask = z["date"].isin(app)
        z.loc[mask, "H"] = (
            z.loc[mask].apply(lambda r_: r_["Z"] > ck.get(r_["model"], np.inf), axis=1)
        ).astype(float)
    zz = z.dropna(subset=["H"])
    rows = []
    for date, g in zz.groupby("date"):
        wide = g.pivot_table(index="asset_ticker", columns="model", values="H", aggfunc="max")
        models = list(wide.columns)
        row = {"date": date}
        for grp in groupings:
            if grp == "cross_family":
                share = wide.mean(axis=1)
            else:
                idx = [mm for mm in models if (grp == "all") or
                       (grp == "within_comm" and families.get(mm) == "commercial") or
                       (grp == "within_open" and families.get(mm) == "open")]
                share = wide[idx].mean(axis=1) if idx else pd.Series(dtype=float)
            row[grp] = float((share >= 0.5).mean()) if len(share) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
