from pathlib import Path
import pandas as pd
from app.validation import validate_frame
from models.preprocessing import regularize_series

def load_dataset(data_dir="data"):
    paths = sorted(Path(data_dir).glob("*.csv"))
    if not paths: raise FileNotFoundError(f"No CSV files found in {data_dir}")
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        # Multiple files may each contain one series; validation occurs after concatenation.
        frames.append(df)
    return regularize_series(validate_frame(pd.concat(frames, ignore_index=True)))

if __name__ == "__main__":
    df = load_dataset()
    print(f"Loaded {len(df):,} hourly rows across {df.series_id.nunique()} series")
    print(df.isna().mean().sort_values(ascending=False).to_string())
