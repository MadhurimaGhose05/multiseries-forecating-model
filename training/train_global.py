from pathlib import Path
import json
import numpy as np
import pandas as pd
from models.global_model import GlobalQuantileForecaster
from app.uncertainty import apply_calibration
from training.evaluate import evaluate_predictions
from training.prepare_data import load_dataset

def _split_one(g):
    g = g.sort_values("timestamp").reset_index(drop=True)
    n = len(g)
    a, b = int(n * .70), int(n * .85)
    return g.iloc[:a].copy(), g.iloc[a:b].copy(), g.iloc[b:].copy()

def _forecast_window(model, context, window, max_horizon=168):
    # Evaluate the first max_horizon points after the split boundary.
    if window.empty: return None
    truth = window.iloc[:max_horizon].copy()
    pred = model.predict(context, len(truth), use_exogenous=True)
    merged = truth[["timestamp","target"]].merge(pred, on="timestamp", how="inner")
    return merged if not merged.empty else None

def train(data_dir="data", artifact_dir="artifacts"):
    df = load_dataset(data_dir)
    train_parts, val_parts, test_parts = [], {}, {}
    for sid, g in df.groupby("series_id"):
        tr, va, te = _split_one(g)
        if len(tr) < 200 or va.empty or te.empty:
            raise ValueError(f"Series {sid} is too short for 70/15/15 chronological split")
        train_parts.append(tr)
        val_parts[sid] = (tr, va)
        test_parts[sid] = (tr, va, te)

    # Train only on the earliest 70% from every training series.
    train_df = pd.concat(train_parts, ignore_index=True)
    model = GlobalQuantileForecaster().fit(train_df)

    # Calibration: pooled out-of-time residuals from the next 15%.
    cal_frames = []
    for sid, (tr, va) in val_parts.items():
        part = _forecast_window(model, tr, va)
        if part is not None: cal_frames.append(part)
    if cal_frames:
        cal = pd.concat(cal_frames, ignore_index=True)
        apply_calibration(model, cal["target"], cal)

    # Untouched test window: next 168 hours per series, after calibration is frozen.
    test_frames = []
    for sid, (tr, va, te) in test_parts.items():
        context = pd.concat([tr, va], ignore_index=True)
        part = _forecast_window(model, context, te)
        if part is not None: test_frames.append(part)
    report = {}
    if test_frames:
        test = pd.concat(test_frames, ignore_index=True)
        report = evaluate_predictions(test["target"], test)
    report["calibration_expansion"] = model.calibration
    report["protocol"] = "per-series chronological 70/15/15; first 168 test hours; pooled validation calibration"
    report["series_count"] = int(df.series_id.nunique())
    report["train_rows"] = int(len(train_df))
    artifact_path = Path(artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)
    model.save(artifact_path)
    (artifact_path / "global_metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return model, report

if __name__ == "__main__":
    train()
