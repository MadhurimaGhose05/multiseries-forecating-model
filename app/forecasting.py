from pathlib import Path
import json
import pandas as pd
from models.global_model import GlobalQuantileForecaster
from models.chronos_model import Chronos2Forecaster
from models.preprocessing import regularize_series

class ForecastService:
    def __init__(self, artifact_dir="artifacts"):
        self.artifact_dir = Path(artifact_dir)
        self.global_model = None
        self.chronos_model = None

    def get_model(self, name):
        if name == "global":
            if self.global_model is None:
                self.global_model = GlobalQuantileForecaster.load(self.artifact_dir)
            return self.global_model
        if name == "chronos":
            if self.chronos_model is None:
                self.chronos_model = Chronos2Forecaster(device="cpu")
                cal = self.artifact_dir / "chronos_calibration.json"
                if cal.exists():
                    self.chronos_model.calibration = json.loads(cal.read_text())
            return self.chronos_model
        raise ValueError("model must be 'global' or 'chronos'")

    def forecast(self, df, horizon, model_name="global", use_exogenous=True):
        if df.series_id.nunique() != 1:
            raise ValueError("Upload exactly one series per forecast request")
        history = regularize_series(df).sort_values("timestamp").copy()
        model = self.get_model(model_name)
        pred = model.predict(history, horizon, use_exogenous=use_exogenous)
        return str(history.series_id.iloc[0]), pred
