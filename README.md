## Person 4 — Time / Spike Detection

Detect sudden increases (“spikes”) in review volume per business using daily counts and a rolling baseline.

### Input
- `data/processed/restaurant_reviews.parquet`

### Output
- `outputs/review_spikes.parquet`

### Output columns
- **`business_id`**
- **`review_date`**
- **`review_count`**
- **`baseline_mean`**, **`baseline_std`** (rolling window over prior days)
- **`history_days`** (number of prior days used for baseline)
- **`abs_increase`**, **`z_score`**
- **`is_spike`** (1 = spike, 0 = not spike)

### Run locally
From repo root:

```bash
python3 scripts/detect_review_spikes.py \
  --input data/processed/restaurant_reviews.parquet \
  --output outputs/review_spikes.parquet
