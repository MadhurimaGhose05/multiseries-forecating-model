import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from models.preprocessing import training_frame, make_feature_row, future_exog_from_history, align_features

QUANTILES = {"q05": 0.05, "q25": 0.25, "q50": 0.50, "q75": 0.75, "q95": 0.95}

class GlobalQuantileForecaster:
    def __init__(self, random_state=42, n_estimators=350, learning_rate=0.04, num_leaves=31):
        self.random_state = random_state
        self.params = dict(n_estimators=n_estimators, learning_rate=learning_rate,
                           num_leaves=num_leaves, random_state=random_state,
                           verbosity=-1, n_jobs=1, deterministic=True, force_col_wise=True)
        self.models = {}
        self.columns = None
        self.calibration = {"50": 0.0, "90": 0.0}

    def fit(self, df):
        X, y = training_frame(df, use_exogenous=True)
        self.columns = list(X.columns)
        for name, q in QUANTILES.items():
            model = LGBMRegressor(objective="quantile", alpha=q, **self.params)
            model.fit(X, y)
            self.models[name] = model
        return self

    def predict(self, history, horizon, use_exogenous=True):
        g = history.sort_values("timestamp").copy().reset_index(drop=True)
        if g.empty: raise ValueError("History is empty")
        last = pd.Timestamp(g.timestamp.iloc[-1])
        future_times = pd.date_range(last + pd.Timedelta(hours=1), periods=horizon, freq="h")
        exogs = future_exog_from_history(g, future_times, use_exogenous)
        out = []
        for i, ts in enumerate(future_times):
            exog = exogs.iloc[i].to_dict()
            row = make_feature_row(g, ts, exog, use_exogenous)
            X = pd.get_dummies(pd.DataFrame([row]), columns=["cov_cat"], dummy_na=True)
            X = align_features(X, self.columns)
            pred = {k: float(m.predict(X)[0]) for k, m in self.models.items()}
            # Enforce non-crossing quantiles.
            vals = np.maximum.accumulate([pred[k] for k in ["q05","q25","q50","q75","q95"]])
            pred = dict(zip(["q05","q25","q50","q75","q95"], vals))
            lo50 = pred["q25"] - self.calibration["50"]
            hi50 = pred["q75"] + self.calibration["50"]
            lo90 = pred["q05"] - self.calibration["90"]
            hi90 = pred["q95"] + self.calibration["90"]
            out.append({"timestamp": ts, "prediction": pred["q50"],
                        "lower_50": lo50, "upper_50": hi50,
                        "lower_90": lo90, "upper_90": hi90})
            # Recursive target feedback; future exogenous inputs are historical seasonal estimates.
            g = pd.concat([g, pd.DataFrame([{
                **{c: exog.get(c, np.nan) for c in [f"aux_{j}" for j in range(1,6)] +
                   [f"cov_{j}" for j in range(1,6)]},
                "timestamp": ts, "series_id": str(g.series_id.iloc[-1]),
                "target": pred["q50"], "cov_cat": exog.get("cov_cat", "__MISSING__")
            }])], ignore_index=True)
        return pd.DataFrame(out)

    def save(self, path):
        path = Path(path); path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path / "global_model.joblib")

    @classmethod
    def load(cls, path):
        return joblib.load(Path(path) / "global_model.joblib")
