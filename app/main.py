import os
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from app.validation import read_csv_upload, DataValidationError
from app.forecasting import ForecastService

app = FastAPI(title="Multi-series Forecasting API", version="1.0.0")
service = ForecastService(os.getenv("ARTIFACT_DIR", "artifacts"))

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/forecast")
async def forecast(
    file: UploadFile = File(...),
    horizon: int = Form(24),
    model: str = Form("global"),
    interval_width: float = Form(0.90),
    use_exogenous: bool = Form(True)
):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Upload a .csv file")
    if not 1 <= horizon <= 1680:
        raise HTTPException(422, "horizon must be between 1 and 1680")
    if model not in {"global", "chronos"}:
        raise HTTPException(422, "model must be global or chronos")
    if interval_width not in {0.5, 0.9}:
        raise HTTPException(422, "interval_width must be 0.5 or 0.9")
    try:
        df = read_csv_upload(await file.read())
        if df.series_id.nunique() != 1:
            raise DataValidationError("Upload one series per request")
        sid, pred = service.forecast(df, horizon, model, use_exogenous)
        # Both interval widths are returned regardless of selected display width.
        points = []
        for r in pred.to_dict("records"):
            points.append({k: (v.isoformat() if k == "timestamp" else float(v))
                           for k,v in r.items()})
        return {"series_id": sid, "model": model, "horizon": horizon,
                "interval_width": interval_width, "forecasts": points,
                "metrics": {"mae": None, "coverage_90": None},
                "warnings": ["Metrics are unavailable without future ground truth."]}
    except DataValidationError as e:
        raise HTTPException(422, str(e))
    except FileNotFoundError as e:
        raise HTTPException(503, "Global model artifact not found. Run python train.py first.")
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(500, f"Forecast failed: {type(e).__name__}: {e}")
