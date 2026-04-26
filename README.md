## Person 3 — Review Features

This branch contains the **Person 3** deliverable: per-review feature engineering for restaurant reviews.

### Inputs
- `data/processed/restaurant_reviews.parquet`
- `data/processed/users_base.parquet`

### Output
- `data/processed/review_features.parquet` (partitioned by `review_year`)

### Features created (per review)
- **`review_length`**: character count of review text
- **`review_hour`**: hour of day (0–23) from review timestamp
- **`day_of_week`**: 1–7 (Spark convention)
- **`days_since_signup`**: days between `yelping_since` and review date
- **`user_review_sequence`**: per-user sequence number ordered by (`date`, `review_id`)

### Run locally
From repo root:

```bash
python3 scripts/build_review_features.py \
  --restaurant_reviews data/processed/restaurant_reviews.parquet \
  --users_base data/processed/users_base.parquet \
  --output data/processed/review_features.parquet
```
