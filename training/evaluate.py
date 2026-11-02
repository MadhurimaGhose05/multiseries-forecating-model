import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def metrics(y, pred):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    ok = np.isfinite(y) & np.isfinite(pred)
    y, pred = y[ok], pred[ok]
    if not len(y): return {"mae": None, "rmse": None, "smape": None}
    denom = np.abs(y) + np.abs(pred)
    smape = np.mean(np.where(denom == 0, 0, 2*np.abs(y-pred)/denom))*100
    return {"mae": float(mean_absolute_error(y,pred)),
            "rmse": float(np.sqrt(mean_squared_error(y,pred))),
            "smape": float(smape)}

def interval_metrics(y, p):
    y = np.asarray(y, float)
    out = {}
    for level in ["50","90"]:
        lo, hi = p[f"lower_{level}"], p[f"upper_{level}"]
        ok = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
        out[f"coverage_{level}"] = float(np.mean((y[ok]>=lo[ok]) & (y[ok]<=hi[ok]))) if ok.any() else None
        out[f"width_{level}"] = float(np.mean(hi[ok]-lo[ok])) if ok.any() else None
    return out

def evaluate_predictions(y, p):
    return {**metrics(y,p["prediction"]), **interval_metrics(y,p)}
