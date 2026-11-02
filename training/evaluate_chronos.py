import json
from pathlib import Path
import pandas as pd
from models.chronos_model import Chronos2Forecaster
from app.uncertainty import apply_calibration
from training.evaluate import evaluate_predictions
from training.train_global import _split_one, _forecast_window
from training.prepare_data import load_dataset

def evaluate_chronos(data_dir="data", artifact_dir="artifacts"):
    df = load_dataset(data_dir)
    model = Chronos2Forecaster(device="cpu").load()
    splits = {sid: _split_one(g) for sid, g in df.groupby("series_id")}
    cal_frames=[]
    for sid,(tr,va,te) in splits.items():
        p=_forecast_window(model,tr,va)
        if p is not None: cal_frames.append(p)
    if cal_frames:
        cal=pd.concat(cal_frames,ignore_index=True)
        apply_calibration(model,cal["target"],cal)
    test_frames=[]
    for sid,(tr,va,te) in splits.items():
        p=_forecast_window(model,pd.concat([tr,va],ignore_index=True),te)
        if p is not None: test_frames.append(p)
    report={}
    if test_frames:
        test=pd.concat(test_frames,ignore_index=True)
        report=evaluate_predictions(test["target"],test)
    report["calibration_expansion"]=model.calibration
    report["protocol"]="same per-series chronological 70/15/15; first 168 test hours"
    Path(artifact_dir).mkdir(parents=True,exist_ok=True)
    (Path(artifact_dir)/"chronos_metrics.json").write_text(json.dumps(report,indent=2))
    model.save_calibration(Path(artifact_dir)/"chronos_calibration.json")
    print(json.dumps(report,indent=2))
    return report

if __name__=="__main__": evaluate_chronos()
