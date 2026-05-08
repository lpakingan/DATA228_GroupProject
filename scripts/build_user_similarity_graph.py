#!/usr/bin/env python3
"""
Build user similarity graph. Find pairs of users who reviewed the same business within a 7-day window and build a weighted edge list where edge weight = number of shared businesses.

Reads:
  - data/processed/review_features.parquet
  - data/processed/user_anomaly_scores.parquet
Writes:
  - outputs/user_similarity_edges.parquet
"""

from pathlib import Path
import pandas as pd
import itertools

# Get the project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

REVIEW_INPUT = PROJECT_ROOT / "data" / "processed" / "review_features.parquet"
SCORE_INPUT = PROJECT_ROOT / "outputs" / "user_anomaly_scores.parquet"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "user_similarity_edges.parquet"

DEFAULT_TIME_WINDOW = 7

def main():
    input_review = REVIEW_INPUT
    input_score = SCORE_INPUT

    review_df = pd.read_parquet(input_review)
    score_df = pd.read_parquet(input_score)

    df = review_df.merge(score_df, on='user_id', how='left')

    anomalous_df = df[df['is_anomaly'] == 1].copy()

    anomalous_df['date'] = pd.to_datetime(anomalous_df['date'])

    edges = []
    for business_id, group in anomalous_df.groupby('business_id'):
        pairs = itertools.combinations(group.itertuples(), 2)
        for row_a, row_b in pairs:
            time_diff_days = abs(row_a.date - row_b.date).days
            if time_diff_days < DEFAULT_TIME_WINDOW:
                edges.append([row_a.user_id, row_b.user_id, row_a.business_id, time_diff_days])

    edges_df = pd.DataFrame(edges, columns=['user_id_a', 'user_id_b', 'business_id', 'time_diff_days'])
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    edges_df.to_parquet(DEFAULT_OUTPUT, index=False)
    print(f"Output saved to: {DEFAULT_OUTPUT}")

if __name__ == "__main__":
    main()