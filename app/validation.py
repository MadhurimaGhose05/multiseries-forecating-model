import io
import pandas as pd

REQUIRED = ["timestamp", "series_id", "target"] + [f"aux_{i}" for i in range(1, 6)] + \
           [f"cov_{i}" for i in range(1, 6)] + ["cov_cat"]
NUMERIC = ["target"] + [f"aux_{i}" for i in range(1, 6)] + [f"cov_{i}" for i in range(1, 6)]

class DataValidationError(ValueError):
    pass

def validate_frame(df: pd.DataFrame, require_target=True) -> pd.DataFrame:
    missing = sorted(set(REQUIRED) - set(df.columns))
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")
    if df.empty:
        raise DataValidationError("CSV contains no rows")
    df = df[REQUIRED].copy()
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise", utc=True).dt.tz_convert(None)
    except Exception as e:
        raise DataValidationError(f"Invalid timestamp values: {e}") from e
    if df["series_id"].isna().any():
        raise DataValidationError("series_id cannot be null")
    df["series_id"] = df["series_id"].astype(str)
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if require_target and df["target"].notna().sum() == 0:
        raise DataValidationError("No observed target values found")
    if df.duplicated(["series_id", "timestamp"]).any():
        raise DataValidationError("Duplicate timestamp/series_id rows found")
    df = df.sort_values(["series_id", "timestamp"]).reset_index(drop=True)
    for sid, part in df.groupby("series_id"):
        if len(part) < 48:
            raise DataValidationError(f"Series {sid!r} has fewer than 48 observations")
        diffs = part["timestamp"].diff().dropna()
        if len(diffs) and (diffs <= pd.Timedelta(0)).any():
            raise DataValidationError(f"Series {sid!r} timestamps are not strictly increasing")
    df["cov_cat"] = df["cov_cat"].fillna("__MISSING__").astype(str)
    return df

def read_csv_upload(raw: bytes) -> pd.DataFrame:
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise DataValidationError(f"Could not parse CSV: {e}") from e
    return validate_frame(df)
