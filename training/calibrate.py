import numpy as np
import pandas as pd
from app.uncertainty import apply_calibration

def calibrate_global(model, validation_df, horizon=168):
    forecasts, actual = [], []
    for sid, g in validation_df.groupby("series_id"):
        g = g.sort_values("timestamp").reset_index(drop=True)
        if len(g) <= 168: continue
        # validation_df is the chronological calibration segment; context is preceding history
        # Caller should pass full series and calibration_start if using this helper directly.
        cut = max(168, int(len(g)*0.5))
        hist, truth = g.iloc[:cut], g.iloc[cut:cut+horizon]
        if truth.empty: continue
        p = model.predict(hist, len(truth))
        joined = truth[["timestamp","target"]].merge(p, on="timestamp")
        forecasts.extend(joined.to_dict("records"))
    if forecasts:
        f = pd.DataFrame(forecasts)
        apply_calibration(model, f.target, f)
    return model
