#!/usr/bin/env python3
"""
FastAPI app for Yelp Review Authenticity results.

Endpoints:
  /suspicious_users
  /suspicious_businesses
  /business/{business_id}/impact

Run:
  uvicorn app:app --reload
  open http://localhost:8000/docs
"""

from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query


# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
RESTAURANTS_PATH = PROJECT_ROOT / "data" / "processed" / "restaurants_base.parquet"

USER_ANOMALY_PATH = OUTPUT_DIR / "user_anomaly_scores.parquet"
SUSPICIOUS_BUSINESSES_PATH = OUTPUT_DIR / "suspicious_businesses.parquet"
VOLUME_IMPACT_PATH = OUTPUT_DIR / "volume_impact.parquet"
RATING_IMPACT_PATH = OUTPUT_DIR / "rating_impact.parquet"


app = FastAPI(title="Yelp Review Authenticity API", version="1.0")

# In-memory data, loaded once at startup
user_anomaly_df = None
suspicious_businesses_df = None
volume_impact_df = None
rating_impact_df = None
business_meta = {}  # business_id -> {name, city, state, categories}


def clean_value(v):
    """Make a single value JSON-safe."""
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d")
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return None if (np.isnan(v) or np.isinf(v)) else float(v)
    try:
        if pd.isna(v):
            return None
    except TypeError:
        pass
    return v


def df_to_records(df):
    """Convert DataFrame to JSON-safe dicts; attach business name when possible."""
    records = df.to_dict(orient="records")
    out = []
    for row in records:
        clean_row = {k: clean_value(v) for k, v in row.items()}
        bid = clean_row.get("business_id")
        if bid and bid in business_meta:
            clean_row["business"] = business_meta[bid]
        out.append(clean_row)
    return out


@app.on_event("startup")
def load_data():
    global user_anomaly_df, suspicious_businesses_df
    global volume_impact_df, rating_impact_df, business_meta

    user_anomaly_df = pd.read_parquet(USER_ANOMALY_PATH)
    suspicious_businesses_df = pd.read_parquet(SUSPICIOUS_BUSINESSES_PATH)
    volume_impact_df = pd.read_parquet(VOLUME_IMPACT_PATH)
    rating_impact_df = pd.read_parquet(RATING_IMPACT_PATH)

    # Build a business_id -> {name, city, state, categories} lookup so the
    # API returns human-readable info instead of just hex IDs.
    if RESTAURANTS_PATH.exists():
        meta = pd.read_parquet(RESTAURANTS_PATH)
        keep = [c for c in ["business_id", "name", "city", "state", "categories"] if c in meta.columns]
        meta = meta[keep].drop_duplicates("business_id")
        business_meta = {
            row["business_id"]: {k: clean_value(row[k]) for k in keep if k != "business_id"}
            for _, row in meta.iterrows()
        }


@app.get("/")
def home():
    return {
        "message": "Yelp Review Authenticity API",
        "endpoints": [
            "/suspicious_users",
            "/suspicious_businesses",
            "/business/{business_id}/impact",
        ],
    }


@app.get("/suspicious_users")
def get_suspicious_users(limit: int = Query(50, ge=1, le=500)):
    """Top suspicious users, sorted by anomaly_rank."""
    users = user_anomaly_df
    if "is_anomaly" in users.columns:
        users = users[users["is_anomaly"] == 1]
    if "anomaly_rank" in users.columns:
        users = users.sort_values("anomaly_rank")
    elif "anomaly_score" in users.columns:
        users = users.sort_values("anomaly_score", ascending=False)
    users = users.head(limit)
    return {"count": len(users), "results": df_to_records(users)}


@app.get("/suspicious_businesses")
def get_suspicious_businesses(limit: int = Query(50, ge=1, le=500)):
    """Top suspicious businesses (spike + suspicious-user signals)."""
    biz = suspicious_businesses_df
    sort_cols = [c for c in ["z_score", "suspicious_user_count"] if c in biz.columns]
    if sort_cols:
        biz = biz.sort_values(sort_cols, ascending=False)
    biz = biz.head(limit)
    return {"count": len(biz), "results": df_to_records(biz)}


@app.get("/business/{business_id}/impact")
def get_business_impact(business_id: str):
    """Volume + rating impact for a specific business around its spike."""
    volume = volume_impact_df[volume_impact_df["business_id"] == business_id]
    rating = rating_impact_df[rating_impact_df["business_id"] == business_id].copy()

    if volume.empty and rating.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No impact results for business_id: {business_id}",
        )

    # Add before-vs-after rating change
    if {"avg_stars_before", "avg_stars_after"}.issubset(rating.columns):
        rating["rating_change"] = rating["avg_stars_after"] - rating["avg_stars_before"]

    return {
        "business_id": business_id,
        "business": business_meta.get(business_id),
        "volume_impact": df_to_records(volume),
        "rating_impact": df_to_records(rating),
    }
