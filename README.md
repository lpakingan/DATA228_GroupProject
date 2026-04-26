## Rating Impact Analysis

### Goal
For each suspicious spike event (`business_id`, `spike_date`), compute average star rating:
- **BEFORE** the spike window
- **DURING** the spike window
- **AFTER** the spike window

### Inputs
- `data/processed/restaurant_reviews.parquet`
- `outputs/suspicious_businesses.parquet`

### Output
- `outputs/rating_impact.parquet`

### Windows (defaults)
- **BEFORE**: \([spike\_date - 30, spike\_date - 1]\)
- **DURING**: spike day only (`during_days = 0`)
- **AFTER**: \([spike\_date + 1, spike\_date + 30]\)

### Run locally
From repo root:

```bash
python3 scripts/rating_impact.py \
  --restaurant_reviews data/processed/restaurant_reviews.parquet \
  --suspicious_businesses outputs/suspicious_businesses.parquet \
  --output outputs/rating_impact.parquet \
  --before_days 30 \
  --during_days 0 \
  --after_days 30
```

