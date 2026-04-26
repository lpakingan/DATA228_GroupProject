# DATA228_GroupProject

Reads data/processed/user_features.parquet
Trains IsolationForest on:
- avg_stars_given
- pct_5_star_reviews
- pct_1_star_reviews
- review_count
- reviews_per_day
- account_age_days
- num_friends
  
Outputs per-user results to outputs/user_anomaly_scores.parquet with:
- user_id
- anomaly_score (higher = more suspicious)
- is_anomaly (1/0)
- anomaly_rank
  
Tracks run in MLflow:
- logs params (paths, hyperparams, feature list, row count)
- logs metrics (n_anomalies, anomaly_rate, score stats, etc.)
- logs output parquet as an artifact
