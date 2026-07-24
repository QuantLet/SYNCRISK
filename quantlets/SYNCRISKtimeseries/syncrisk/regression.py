"""Newey-West (HAC) OLS helper for the descriptive Eq.7 projection."""
from __future__ import annotations


def newey_west(y, X, maxlags):
    """OLS of y on [const, X] with HAC (Newey-West) standard errors."""
    import statsmodels.api as sm
    Xc = sm.add_constant(X)
    return sm.OLS(y, Xc, missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
