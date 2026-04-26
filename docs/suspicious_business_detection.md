## Suspicious Business Detection

### Goal
Identify businesses that show **review spikes** and have activity from **suspicious (anomalous) users** during those spike periods.

### Inputs
- `outputs/review_spikes.parquet`
- `outputs/user_anomaly_scores.parquet`
- `data/processed/restaurant_reviews.parquet`

### Output
- `outputs/suspicious_businesses.parquet`

### Approach (high level)
- Filter to spike business-days (`is_spike = 1`)
- Filter to suspicious users (`is_anomaly = 1`)
- Join restaurant reviews to spike business-days within a configurable window (default: same day)
- Keep only reviews authored by suspicious users in that window
- Aggregate per `business_id` + `spike_date` to produce:
  - `total_reviews_in_window`
  - `suspicious_reviews_in_window`
  - `suspicious_user_count`
  - `max_anomaly_score`, `avg_anomaly_score`

### Run locally
From repo root:

```bash
python3 scripts/business_suspicious_activity.py \
  --review_spikes outputs/review_spikes.parquet \
  --user_anomaly_scores /path/to/outputs/user_anomaly_scores.parquet \
  --restaurant_reviews data/processed/restaurant_reviews.parquet \
  --output outputs/suspicious_businesses.parquet \
  --window_days 0
```

To widen the window around the spike day (e.g. \(\pm 3\) days), set `--window_days 3`.

