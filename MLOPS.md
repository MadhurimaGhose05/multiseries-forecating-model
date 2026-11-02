# MLOps production plan

## Registry and versioning
Register the LightGBM artifact and Chronos-2 model reference separately in MLflow Model Registry.
Version each artifact with immutable model version, training-data snapshot/hash, feature-schema
version, code commit, dependency lockfile, random seed, calibration artifact and evaluation report.
For Chronos-2, record the Hugging Face repository revision and cached weight checksum.

## Online and batch serving
Use FastAPI for low-latency online forecasts, with model artifacts loaded once per worker and
CPU thread limits to avoid oversubscription. For large fleets or long horizons, use scheduled
batch inference with partitioned input/output in object storage. Keep the same preprocessing
library in both paths. Run API and UI separately; scale API workers independently.

## Data quality and observability
At ingestion, monitor schema violations, duplicate timestamps, frequency gaps, target missingness,
covariate missingness, category novelty, outlier rates and timestamp freshness. Reject structurally
invalid inputs with 4xx responses; log quality warnings for recoverable missing values. Do not log
raw uploaded data or identifiers unless approved.

## Drift monitoring
Track feature distributions, target level/variance, seasonal profile, residual MAE/RMSE/sMAPE,
interval coverage and interval width by series and horizon. Use PSI or Wasserstein distance for
numeric features and category-frequency/unknown-category rates for `cov_cat`. Monitor missingness
and covariate forecast-vs-observed error separately. Ground-truth metrics are delayed until labels
arrive.

## Retraining triggers
Trigger investigation/retraining when rolling MAE exceeds a baseline threshold, 90% interval
coverage falls materially below target, feature drift persists for multiple windows, missingness
changes sharply, or a new schema is approved. Use rolling-origin backtests and unseen-series
holdouts before promotion. Retrain LightGBM on pooled series; refresh the foundation-model
calibration independently and upgrade Chronos weights only after a controlled benchmark.

## Deployment, canary and rollback
Promote candidate versions through offline validation, shadow traffic, then a small canary.
Compare latency, failures, point accuracy and interval calibration against the incumbent.
Keep the previous model and calibration artifact immutable. Roll back by switching the registry
alias/deployment manifest to the last known-good version. Never overwrite a production artifact
in place.

## Security and governance
Use least-privilege service identities, encrypted object storage, secrets management, access
controls and audit logs. The private OneDrive source data must not be committed to GitHub.
Set retention limits for uploaded files and avoid returning training data in responses.
