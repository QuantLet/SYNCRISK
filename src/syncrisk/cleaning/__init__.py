"""SYNCRISK cleaning specifications. Main spec = expanding-window C2x/C3x (Amendment 6)."""
from .features import C1_FEATURES, C2_FEATURES
from .expanding_window import (expanding_residuals, resid_r2, expanding_tail_majority,
                               MIN_HISTORY, REFIT_EVERY)

__all__ = ["C1_FEATURES", "C2_FEATURES", "expanding_residuals", "resid_r2",
           "expanding_tail_majority", "MIN_HISTORY", "REFIT_EVERY"]
