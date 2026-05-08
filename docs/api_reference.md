# API Reference — Yelp Review Authenticity

FastAPI app: `app.py`. Run with `uvicorn app:app --reload`, then open http://localhost:8000/docs.

## Endpoints

| Method | Path | Returns |
|---|---|---|
| GET | `/` | API index — list of available endpoints |
| GET | `/suspicious_users?limit=50` | Top suspicious users from the Isolation Forest model |
| GET | `/suspicious_businesses?limit=50` | Top businesses with suspicious review spikes |
| GET | `/business/{business_id}/impact` | Volume + rating impact around a single business's spike |

All list endpoints accept `limit` (1–500, default 50). All business‑id rows are auto‑enriched with `name`, `city`, `state`, and `categories` joined from `data/processed/restaurants_base.parquet`.

---

## `GET /` — Index

Lightweight description of the API and its endpoints. Useful as a smoke test.

**Example response**

```json
{
  "message": "Yelp Review Authenticity API",
  "endpoints": [
    "/suspicious_users",
    "/suspicious_businesses",
    "/business/{business_id}/impact"
  ]
}
```

---

## `GET /suspicious_users`

Top users flagged by the Isolation Forest anomaly detector.

**Query params**
- `limit` (int, 1–500, default 50) — max rows to return.

**Sort order:** `anomaly_rank` ascending (rank 1 = most anomalous). Falls back to `anomaly_score` descending if rank is missing.

**Filter:** only users with `is_anomaly == 1` are returned.

**Field reference**

| Field | Type | Meaning |
|---|---|---|
| `user_id` | string | Yelp user identifier |
| `avg_stars_given` | float | Mean star rating across all of the user's reviews (1–5) |
| `pct_5_star_reviews` | float (0–1) | Fraction of the user's reviews that are 5 stars |
| `pct_1_star_reviews` | float (0–1) | Fraction of the user's reviews that are 1 star |
| `review_count` | int | Total reviews this user has posted |
| `reviews_per_day` | float | Average review velocity = `review_count / account_age_days` |
| `account_age_days` | int | Days since the user joined Yelp |
| `num_friends` | int | Size of the user's social graph on Yelp |
| `anomaly_score` | float | Isolation Forest anomaly score; higher = more anomalous (typical range ≈ −0.2 to 0.23 in this run) |
| `is_anomaly` | 0/1 | 1 if the user is in the top contamination% (here 5%) of the IF distribution |
| `anomaly_rank` | int | 1 = most anomalous user; ties broken arbitrarily |

**Example response (truncated to 1 row)**

```json
{
  "count": 50,
  "results": [
    {
      "user_id": "nnImk681KaRqUVHlSfZjGQ",
      "avg_stars_given": 3.90,
      "pct_5_star_reviews": 0.389,
      "pct_1_star_reviews": 0.046,
      "review_count": 499,
      "reviews_per_day": 0.069,
      "account_age_days": 7225,
      "num_friends": 4214,
      "anomaly_score": 0.234,
      "is_anomaly": 1,
      "anomaly_rank": 1
    }
  ]
}
```

> ⚠️ A high anomaly rank does **not** automatically mean fraud. The model isolates statistical outliers; some are legitimate "power users" (high reviews + high friends, like rank 1 above), while others are isolated low‑friend / low‑rating accounts that look more like review bombers. 

---

## `GET /suspicious_businesses`

Businesses with at least one detected review spike, optionally overlapping with flagged users.

**Query params**
- `limit` (int, 1–500, default 50)

**Sort order:** by `z_score` descending, then `suspicious_user_count` descending.

**Field reference**

| Field | Type | Meaning |
|---|---|---|
| `business_id` | string | Yelp business identifier |
| `spike_date` | date (YYYY‑MM‑DD) | Day on which the spike was detected |
| `review_count` | int | Reviews on the spike day |
| `baseline_mean` | float | Mean reviews/day over the 30‑day baseline window preceding the spike |
| `baseline_std` | float | Std. dev. of reviews/day in the baseline window |
| `history_days` | int | Width of the baseline window used (typically 30) |
| `abs_increase` | float | `review_count − baseline_mean` (how many more reviews than typical) |
| `z_score` | float | `(review_count − baseline_mean) / baseline_std` — std devs above normal |
| `total_reviews_in_window` | int | Total reviews in the spike's analysis window |
| `suspicious_reviews_in_window` | int | Reviews authored by flagged anomalous users in the window |
| `suspicious_user_count` | int | Distinct flagged users who reviewed in the window |
| `max_anomaly_score` | float \| null | Highest anomaly score among reviewers in the window; null if 0 suspicious users |
| `avg_anomaly_score` | float \| null | Mean anomaly score among flagged reviewers in the window |
| `window_days` | int | Width of the analysis window (0 = spike day only) |
| `business` | object | `{name, city, state, categories}` joined from restaurants metadata |

**Example response (1 row)**

```json
{
  "count": 50,
  "results": [
    {
      "business_id": "Om9eoEcwPK1lp1-HEjBzeQ",
      "spike_date": "2018-09-27",
      "review_count": 7,
      "baseline_mean": 1.033,
      "baseline_std": 0.183,
      "history_days": 30,
      "abs_increase": 5.97,
      "z_score": 32.68,
      "total_reviews_in_window": 7,
      "suspicious_reviews_in_window": 2,
      "suspicious_user_count": 2,
      "max_anomaly_score": 0.069,
      "avg_anomaly_score": 0.067,
      "window_days": 0,
      "business": {
        "name": "Urban Cantina",
        "city": "Tampa",
        "state": "FL",
        "categories": "Mexican, Restaurants"
      }
    }
  ]
}
```

> The strongest fraud candidates are rows where **both** signals fire: high `z_score` **and** `suspicious_user_count > 0`. A high z‑score with zero suspicious users typically reflects an organic event (viral moment, opening, press coverage).

---

## `GET /business/{business_id}/impact`

Volume + rating impact analysis for one business's spike. Combines `volume_impact.parquet` and `rating_impact.parquet`.

**Path param**
- `business_id` (string) — Yelp business id

**Returns 404** if the business has no impact rows in either parquet.

**Top‑level shape**

```json
{
  "business_id": "...",
  "business": { "name": "...", "city": "...", "state": "...", "categories": "..." },
  "volume_impact": [ { /* one row per spike, see below */ } ],
  "rating_impact": [ { /* one row per spike, see below */ } ]
}
```

### `volume_impact[]` field reference

| Field | Type | Meaning |
|---|---|---|
| `business_id`, `spike_date`, `window_days` | — | Same as in `/suspicious_businesses` |
| `spike_day_review_count` | int | Reviews exactly on the spike day |
| `baseline_mean`, `abs_increase`, `z_score` | — | Same as in `/suspicious_businesses` |
| `suspicious_reviews_in_window` | int | Reviews from flagged users in the impact window |
| `suspicious_user_count` | int | Distinct flagged users in the impact window |
| `total_reviews_in_impact_window` | int | All reviews falling in the before+during+after window |
| `suspicious_review_share` | float (0–1) | `suspicious_reviews_in_window / total_reviews_in_impact_window` |
| `has_suspicious_users` | 0/1 | 1 if at least one flagged user reviewed in the window |
| `reviews_before_spike` | int | Reviews in the N‑day window **before** the spike |
| `reviews_on_spike_date` | int | Reviews on the spike day itself |
| `reviews_after_spike` | int | Reviews in the N‑day window **after** the spike |
| `after_minus_before` | int | `reviews_after_spike − reviews_before_spike` (raw volume delta) |
| `after_before_pct_change` | float | Percent change of after‑window vs before‑window volume |
| `volume_change_label` | string | `"increased"`, `"decreased"`, or `"no_change"` |
| `meaningful_increase_flag` | 0/1 | 1 if the post‑spike increase exceeds a project‑defined threshold |
| `business` | object | Joined metadata |

### `rating_impact[]` field reference

| Field | Type | Meaning |
|---|---|---|
| `business_id`, `spike_date`, `window_days` | — | Same as above |
| `review_count`, `baseline_mean`, `baseline_std`, `history_days`, `abs_increase`, `z_score` | — | Same as in `/suspicious_businesses` |
| `total_reviews_in_window`, `suspicious_reviews_in_window`, `suspicious_user_count`, `max_anomaly_score`, `avg_anomaly_score` | — | Same as in `/suspicious_businesses` |
| `avg_stars_before` | float | Mean stars across reviews in the **before** window |
| `n_reviews_before` | int | Reviews in the before window |
| `avg_stars_during` | float | Mean stars across reviews **on the spike day** |
| `n_reviews_during` | int | Reviews on the spike day |
| `avg_stars_after` | float | Mean stars across reviews in the **after** window |
| `n_reviews_after` | int | Reviews in the after window |
| `before_days`, `during_days`, `after_days` | int | Widths of each window in days |
| `rating_change` | float | `avg_stars_after − avg_stars_before` (added by the API for convenience) |
| `business` | object | Joined metadata |

**Example response — Urban Cantina**

```json
{
  "business_id": "Om9eoEcwPK1lp1-HEjBzeQ",
  "business": {
    "name": "Urban Cantina",
    "city": "Tampa",
    "state": "FL",
    "categories": "Mexican, Restaurants"
  },
  "volume_impact": [
    {
      "business_id": "Om9eoEcwPK1lp1-HEjBzeQ",
      "spike_date": "2018-09-27",
      "window_days": 30,
      "spike_day_review_count": 7,
      "baseline_mean": 1.0333333333333334,
      "abs_increase": 5.966666666666667,
      "z_score": 32.680779264474914,
      "suspicious_reviews_in_window": 2,
      "suspicious_user_count": 2,
      "total_reviews_in_impact_window": 16,
      "suspicious_review_share": 0.125,
      "has_suspicious_users": 1,
      "reviews_before_spike": 3,
      "reviews_on_spike_date": 7,
      "reviews_after_spike": 6,
      "after_minus_before": 3,
      "after_before_pct_change": 100,
      "volume_change_label": "increased",
      "meaningful_increase_flag": 1,
      "business": {
        "name": "Urban Cantina",
        "city": "Tampa",
        "state": "FL",
        "categories": "Mexican, Restaurants"
      }
    }
  ],
  "rating_impact": [
    {
      "business_id": "Om9eoEcwPK1lp1-HEjBzeQ",
      "spike_date": "2018-09-27",
      "review_count": 7,
      "baseline_mean": 1.0333333333333334,
      "baseline_std": 0.18257418583505536,
      "history_days": 30,
      "abs_increase": 5.966666666666667,
      "z_score": 32.680779264474914,
      "total_reviews_in_window": 7,
      "suspicious_reviews_in_window": 2,
      "suspicious_user_count": 2,
      "max_anomaly_score": 0.0685724860983794,
      "avg_anomaly_score": 0.06713286513515049,
      "window_days": 0,
      "avg_stars_before": 3,
      "n_reviews_before": 3,
      "avg_stars_during": 5,
      "n_reviews_during": 7,
      "avg_stars_after": 4.666666666666667,
      "n_reviews_after": 6,
      "before_days": 30,
      "during_days": 0,
      "after_days": 30,
      "rating_change": 1.666666666666667,
      "business": {
        "name": "Urban Cantina",
        "city": "Tampa",
        "state": "FL",
        "categories": "Mexican, Restaurants"
      }
    }
  ]
}
```

---

## Notes

- `max_anomaly_score` and `avg_anomaly_score` are `null` when `suspicious_user_count == 0`, which is expected.
- `window_days: 0` in `rating_impact` means the *during* window is the spike day only (single‑day). It is **not** the same as `window_days` in `volume_impact`, which refers to the surrounding before/after window width.
- The `business` object is `null` for any `business_id` that wasn't in `restaurants_base.parquet` (e.g. non‑restaurant businesses).
