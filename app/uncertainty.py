import numpy as np

def conformal_expansion(y_true, lower, upper, nominal_coverage):
    """Finite-sample split-conformal additive expansion for central prediction intervals."""
    y = np.asarray(y_true, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    ok = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    scores = np.maximum.reduce([lo[ok] - y[ok], y[ok] - hi[ok], np.zeros(ok.sum())])
    if len(scores) == 0: return 0.0
    rank = min(len(scores), int(np.ceil((len(scores) + 1) * nominal_coverage)))
    return float(np.partition(scores, rank - 1)[rank - 1])

def apply_calibration(model, y_true, predictions):
    y = np.asarray(y_true, dtype=float)
    p = predictions
    model.calibration["50"] = conformal_expansion(y, p["lower_50"], p["upper_50"], 0.50)
    model.calibration["90"] = conformal_expansion(y, p["lower_90"], p["upper_90"], 0.90)
    return model
