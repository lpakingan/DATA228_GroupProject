# Yelp Review Authenticity Detector
**DATA 228 — Big Data Technologies | Group 5**
Pratiksha Kaushik, Sang Ah Lee, Liana Pakingan, Louisa Stumpf

---

## Overview
This project detects coordinated fake review activity ("review farms") on Yelp restaurant listings using unsupervised machine learning. Since no ground truth labels exist for authentic vs inauthentic reviews, we combine Isolation Forest anomaly detection, graph-based community detection, and time-series spike analysis to identify suspicious businesses and quantify their rating impact.

---

## Dataset
[Yelp Open Dataset](https://business.yelp.com/data/resources/open-dataset/) — 8.65GB uncompressed, 5 JSON files:
- `yelp_academic_dataset_review.json` — ~7M reviews
- `yelp_academic_dataset_user.json` — ~2M users
- `yelp_academic_dataset_business.json`
- `yelp_academic_dataset_tip.json`
- `yelp_academic_dataset_checkin.json`

---

## Pipeline

**Phase 1 — Data & Feature Engineering**
Convert raw JSON to Parquet via PySpark, build base tables, and engineer per-review and per-user behavioral features.

**Phase 2 — Detection & Modeling**
Run Isolation Forest to score anomalous users (tracked via MLflow), detect review volume spikes, build a user similarity graph, and apply Louvain community detection to find coordinated review farm clusters.

**Phase 3 — Impact Analysis**
Measure rating and volume changes before, during, and after suspicious spike events.

**Phase 4 — Deployment & UI**
Serve results via FastAPI and visualize in a Streamlit dashboard with a 0-100 business authenticity scoring system.

---

## Setup

```bash
git clone https://github.com/lpakingan/DATA228_GroupProject
cd DATA228_GroupProject
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Requirements:** Python 3.9+, Java 11 (for PySpark)

---

## Running the Pipeline

Run scripts in this order from the repo root:

```bash
# Phase 1 — Data ingestion and feature engineering
python scripts/download_data.py
python scripts/json_to_parquet.py
python scripts/build_restaurants_base.py
python scripts/build_reviews_base.py
python scripts/build_users_base.py
python scripts/build_restaurant_reviews.py
python scripts/build_review_features.py
python scripts/build_user_features.py

# Phase 2 — Detection and modeling
python scripts/user_anomaly.py
python scripts/detect_review_spikes.py
python scripts/build_user_similarity_graph.py
python scripts/community_detection.py
python scripts/business_suspicious_activity.py \
  --user_anomaly_scores outputs/user_anomaly_scores.parquet \
  --review_spikes outputs/review_spikes.parquet

# Phase 3 — Impact analysis
python scripts/rating_impact.py
python scripts/volume_impact.py
```

---

## Running the Demo

**Streamlit dashboard:**
```bash
streamlit run app_streamlit.py
```

**FastAPI:**
```bash
uvicorn app:app --reload
```
Then open `http://localhost:8000/docs` for the API documentation.

