import numpy as np
import pandas as pd

NUMERIC_INPUTS = [f"aux_{i}" for i in range(1, 6)] + [f"cov_{i}" for i in range(1, 6)]

def regularize_series(df: pd.DataFrame) -> pd.DataFrame:
    """Create an hourly index; never fill target labels with fabricated zeroes."""
    chunks = []
    for sid, g in df.groupby("series_id"):
        g = g.sort_values("timestamp").set_index("timestamp")
        full = pd.date_range(g.index.min(), g.index.max(), freq="h")
        g = g.reindex(full)
        g.index.name = "timestamp"
        g["series_id"] = str(sid)
        g["cov_cat"] = g["cov_cat"].ffill().bfill().fillna("__MISSING__")
        chunks.append(g.reset_index())
    return pd.concat(chunks, ignore_index=True)

def past_only_fill(s: pd.Series) -> pd.Series:
    # Forward-fill only; leading gaps use expanding past-independent training median.
    med = s.median()
    if not np.isfinite(med):
        med = 0.0
    return s.ffill().fillna(float(med))

def future_exog_from_history(history: pd.DataFrame, future_times, use_exogenous=True):
    """Forecast future auxiliary/covariate values from historical hour-of-week medians."""
    out = pd.DataFrame({"timestamp": pd.to_datetime(future_times)})
    if not use_exogenous:
        for c in NUMERIC_INPUTS:
            out[c] = 0.0
        out["cov_cat"] = "__DISABLED__"
        return out
    h = history.copy()
    h["hour_of_week"] = h["timestamp"].dt.dayofweek * 24 + h["timestamp"].dt.hour
    out["hour_of_week"] = out["timestamp"].dt.dayofweek * 24 + out["timestamp"].dt.hour
    for c in NUMERIC_INPUTS:
        vals = h.groupby("hour_of_week")[c].median()
        global_med = h[c].median()
        if not np.isfinite(global_med): global_med = 0.0
        out[c] = out["hour_of_week"].map(vals).fillna(global_med)
    cat_mode = h.groupby("hour_of_week")["cov_cat"].agg(
        lambda x: x.dropna().mode().iloc[0] if not x.dropna().mode().empty else "__MISSING__")
    fallback = h["cov_cat"].dropna().mode()
    fallback = fallback.iloc[0] if not fallback.empty else "__MISSING__"
    out["cov_cat"] = out["hour_of_week"].map(cat_mode).fillna(fallback).astype(str)
    return out.drop(columns=["hour_of_week"])

def make_feature_row(history: pd.DataFrame, timestamp, exog: dict, use_exogenous=True):
    y = history["target"].astype(float)
    row = {}
    for lag in [1, 2, 3, 6, 12, 24, 48, 72, 168]:
        row[f"target_lag_{lag}"] = float(y.iloc[-lag]) if len(y) >= lag and pd.notna(y.iloc[-lag]) else np.nan
    for window in [6, 12, 24, 48, 168]:
        tail = y.iloc[-window:]
        row[f"target_mean_{window}"] = float(tail.mean()) if tail.notna().any() else np.nan
        row[f"target_std_{window}"] = float(tail.std()) if tail.notna().sum() > 1 else np.nan
    ts = pd.Timestamp(timestamp)
    row.update({
        "hour": ts.hour, "dayofweek": ts.dayofweek, "day": ts.day,
        "month": ts.month, "weekofyear": int(ts.isocalendar().week),
        "hour_sin": np.sin(2*np.pi*ts.hour/24), "hour_cos": np.cos(2*np.pi*ts.hour/24),
        "dow_sin": np.sin(2*np.pi*ts.dayofweek/7), "dow_cos": np.cos(2*np.pi*ts.dayofweek/7),
    })
    for c in NUMERIC_INPUTS:
        row[c] = float(exog.get(c, 0.0)) if use_exogenous else 0.0
        row[f"{c}_missing"] = int(pd.isna(exog.get(c, np.nan))) if use_exogenous else 0
    row["cov_cat"] = str(exog.get("cov_cat", "__DISABLED__")) if use_exogenous else "__DISABLED__"
    return row

def training_frame(df: pd.DataFrame, use_exogenous=True):
    """Supervised one-step rows; each feature uses observations strictly before its label."""
    rows = []
    for sid, g in df.groupby("series_id"):
        g = g.sort_values("timestamp").reset_index(drop=True)
        for i in range(168, len(g)):
            if pd.isna(g.loc[i, "target"]): continue
            hist = g.iloc[:i]
            # Match serving: future covariates are estimated from the history available
            # strictly before this label, never read from the label timestamp itself.
            future_exog = future_exog_from_history(
                hist, [g.loc[i, "timestamp"]], use_exogenous
            ).iloc[0].to_dict()
            rows.append((make_feature_row(hist, g.loc[i, "timestamp"], future_exog, use_exogenous),
                         float(g.loc[i, "target"])))
    if not rows: raise ValueError("No supervised rows available; need at least 169 hourly rows with targets")
    X = pd.DataFrame([x for x, _ in rows])
    X = pd.get_dummies(X, columns=["cov_cat"], dummy_na=True)
    return X, np.asarray([y for _, y in rows], dtype=float)

def align_features(X: pd.DataFrame, columns):
    return X.reindex(columns=columns, fill_value=0.0)
