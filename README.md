# Multi-series forecasting

CPU-first implementation of a pooled LightGBM quantile forecaster and a pretrained Chronos-2
foundation-model adapter. The service accepts one previously unseen series at inference time.

## Quick start

1. Put the 10 authorized training CSVs in `data/`. Do not commit the private data.
2. Create an environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
   pip install -r requirements.txt
   ```
   Chronos-2 downloads pretrained weights on first use and can be slow on CPU.
3. Train the global model and initialize Chronos-2:
   ```bash
   python train.py
   ```
   To train only the global model: `python train.py --skip-chronos`.
4. Start the service and UI:
   ```bash
   docker compose up --build
   ```
   Open http://localhost:8501 and API docs at http://localhost:8000/docs.

## API

`POST /forecast` multipart form:
- `file`: one-series CSV with the required schema
- `horizon`: 1..1680 hourly steps
- `model`: `global` or `chronos`
- `interval_width`: `0.5` or `0.9` (both intervals are returned)
- `use_exogenous`: `true` or `false`

The API returns a point prediction and 50%/90% intervals for each timestamp. Metrics are
`null` at inference unless ground truth is separately supplied; the UI must not fabricate them.

## Exogenous feature availability

At forecast time the caller supplies historical observations only, unless it separately supplies
known-future covariates in a future extension of this API. This implementation assumes no
future observed aux/cov values are available. It estimates future numeric aux/cov values using
hour-of-week medians calculated from the uploaded history, and estimates `cov_cat` using the
historical hour-of-week mode. This is a deterministic seasonal-naive exogenous scenario, not
a claim that future covariates are known. With `use_exogenous=false`, future exogenous values
are neutralized. The API accepts the historical CSV, horizon, model, interval width and feature
toggle. A production extension should accept a separately validated future-covariate CSV.

For global-model training, target labels with missing values are excluded. Feature gaps are
forward-filled using only past observations, then initialized with a training-series median;
missingness flags are retained. Rows are regularized to hourly frequency. Missing target values
are never converted to zero. Categorical values are string encoded via one-hot encoding; unseen
categories are ignored through feature-column alignment.

## Validation and metrics

Training utilities and metric functions are provided. For a report-quality benchmark, create a
chronological train/calibration/test partition (e.g. 70%/15%/15%) independently within each
series. Fit preprocessing and models only on the training period; calibrate intervals on the
calibration period; report MAE, RMSE, sMAPE, empirical 50%/90% coverage and mean interval
width on the untouched test period. Use identical timestamps/horizons for both models.
Also run leave-one-series-out tests to approximate the debrief's unseen-series condition.

No benchmark numbers are hard-coded: populate the results table only after running evaluation
on all 10 authorized series. The two held-out series must not be used for model selection.

| Model | MAE | RMSE | sMAPE | 50% coverage | 90% coverage |
|---|---:|---:|---:|---:|---:|
| Global LightGBM | TBD | TBD | TBD | TBD | TBD |
| Chronos-2 | TBD | TBD | TBD | TBD | TBD |

## Determinism

Seeds are fixed, LightGBM uses one CPU thread and deterministic mode, and feature construction
is deterministic. Pin Python package versions. Chronos weights are identified by model ID;
for strict production reproducibility, pin a Hugging Face model revision and cache the weights.

## Tests

`pytest -q`

## What I would do next

Add a future-covariate upload contract, rolling-origin backtests, series-level calibration,
monitoring of missingness and interval coverage, and a true leave-one-series-out benchmark.
