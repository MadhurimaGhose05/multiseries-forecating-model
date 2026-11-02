import numpy as np
import pandas as pd

class Chronos2Forecaster:
    """CPU-capable pretrained Chronos-2 adapter. Model weights are downloaded on first use."""
    def __init__(self, model_id="amazon/chronos-2", device="cpu"):
        self.model_id = model_id
        self.device = device
        self.pipeline = None
        self.calibration = {"50": 0.0, "90": 0.0}

    def load(self):
        if self.pipeline is None:
            try:
                from chronos import Chronos2Pipeline
            except ImportError as e:
                raise RuntimeError("Chronos-2 dependency missing. Install requirements.txt and retry.") from e
            self.pipeline = Chronos2Pipeline.from_pretrained(self.model_id, device_map=self.device)
        return self

    def predict(self, history, horizon, use_exogenous=True):
        self.load()
        g = history.sort_values("timestamp").copy()
        g = g.dropna(subset=["target"])
        if len(g) < 32: raise ValueError("Chronos-2 requires at least 32 observed target values")
        # Chronos-2 dataframe interface: target context plus known-future covariates.
        context = g[["timestamp", "target"]].copy()
        context["id"] = "uploaded_series"
        future_times = pd.date_range(g.timestamp.max() + pd.Timedelta(hours=1), periods=horizon, freq="h")
        future = pd.DataFrame({"id": "uploaded_series", "timestamp": future_times})
        if use_exogenous:
            from models.preprocessing import future_exog_from_history
            ex = future_exog_from_history(g, future_times, True)
            for c in [f"cov_{i}" for i in range(1,6)]:
                context[c] = pd.to_numeric(g[c], errors="coerce").ffill().fillna(g[c].median())
                future[c] = ex[c].values
            # Chronos-2 numerical covariates are used here; categorical covariate is excluded.
        try:
            pred = self.pipeline.predict_df(
                context_df=context, future_df=future, prediction_length=horizon,
                quantile_levels=[0.05, 0.25, 0.5, 0.75, 0.95],
                id_column="id", timestamp_column="timestamp", target="target"
            )
        except TypeError:
            pred = self.pipeline.predict_df(context, future, prediction_length=horizon,
                                            quantile_levels=[0.05,0.25,0.5,0.75,0.95])
        pred = pred.sort_values("timestamp").reset_index(drop=True)
        # Chronos output columns are usually '0.05', '0.25', ...; normalize supported forms.
        def col(q):
            for name in [str(q), f"quantile_{q}", f"q{int(q*100):02d}"]:
                if name in pred.columns: return pred[name].astype(float).to_numpy()
            raise RuntimeError(f"Chronos output missing quantile {q}; received columns {list(pred.columns)}")
        a05,a25,a50,a75,a95 = [col(q) for q in [0.05,0.25,0.5,0.75,0.95]]
        a05,a25,a50,a75,a95 = [np.asarray(x) for x in [a05,a25,a50,a75,a95]]
        return pd.DataFrame({
            "timestamp": future_times, "prediction": a50,
            "lower_50": a25-self.calibration["50"], "upper_50": a75+self.calibration["50"],
            "lower_90": a05-self.calibration["90"], "upper_90": a95+self.calibration["90"]
        })

    def save_calibration(self, path):
        import json
        with open(path, "w") as f: json.dump(self.calibration, f)
