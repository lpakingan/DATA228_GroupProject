# Case Studies: User Anomaly Detection

**Model:** Isolation Forest (config: v1_c005, contamination=0.05)  
**Input:** `outputs/user_anomaly_scores.parquet`  
**Total users scored:** 72,300 (~3,615 flagged as anomalies)

---

## Dataset Overview

| Metric | Value |
|---|---|
| Total users scored | 72,300 |
| Mean review count | 23.8 |
| Median reviews/day | 0.003 |
| Median num_friends | 120 |
| Max reviews/day | 0.269 |
| Max num_friends | 14,995 |

---

## Key Finding: What Drives Anomaly Score

Correlation with anomaly score:

| Feature | Correlation |
|---|---|
| reviews_per_day | 0.504 |
| review_count | 0.474 |
| num_friends | 0.387 |

Review velocity and volume are the strongest signals of anomaly.  
High friend counts also contribute, but may not be the case all the time. In some cases this likely reflects highly active legitimate users (“power users”) rather than suspicious activity.

---

## Case Study 1: Power Users / Social Hubs

Top-ranked anomalous users:

| user_id | review_count | reviews_per_day | num_friends | anomaly_score |
|---|---|---|---|---|
| nnImk681KaRqUVHlSfZjGQ | 499 | 0.069 | 4214 | 0.2339 |
| 6s-g2vFu12OemhiK3FJuOQ | 689 | 0.098 | 4449 | 0.2330 |
| y8aWXOimQ9ZgUgZ6q--nCQ | 229 | 0.034 | 1716 | 0.2312 |
| bJ5FtCtZX3ZZacz2_2PJjA | 765 | 0.112 | 1538 | 0.2306 |
| Ase_kJIYuT6yOsqqVPuWUA | 264 | 0.041 | 1793 | 0.2299 |

These users exhibit extremely high review counts and large social networks.  
Their review frequency is significantly higher than the median user, indicating strong engagement.

**Interpretation:**  
Although these users are flagged as anomalies, their behavior is consistent with highly active “power users” rather than malicious accounts. 
Their anomaly status is driven primarily by scale (activity and connectivity), not necessarily suspicious intent.

---

## Case Study 2: Isolated High-Risk Users (0 Friends)

| user_id | avg_stars | pct_1_star | review_count | anomaly_rank |
|---|---|---|---|---|
| QG-5Xa3R9_TmDDL4g9BiRA | 1.89 | 69.0% | 84 | 588 |
| Z6gS-BqSWT35vY1XtGLLeQ | 1.78 | 57.4% | 68 | 1014 |
| sGCCjnXG_3SoUmqwyswEfA | 2.12 | 54.1% | 135 | 1112 |
| hPtHL1JLtYIIrCDhrjUBaQ | 2.93 | 26.5% | 298 | 1236 |

These users have:
- no social connections (`num_friends = 0`)
- consistently low ratings
- a high proportion of 1-star reviews
- moderate to high review activity

**Interpretation:**  
This combination of strong negative bias and complete social isolation suggests potentially behavior that has been artificially created. These accounts may have been created primarily to post negative reviews, consistent with review bombing patterns.

Not all users in this group exhibit extreme behavior (e.g., some have mixed ratings).
This indicates that anomaly detection captures a spectrum of deviations from typical user behavior.

---

## Case Study 3: High-Activity Reviewers

Users with `reviews_per_day > 0.1`:

| user_id | review_count | reviews_per_day | num_friends | anomaly_rank |
|---|---|---|---|---|
| _BcWyKQL16ndpBdggh2kNA | 1704 | 0.261 | 3708 | 8 |
| -G7Zkl1wIWBBmD0KRy_sCw | 1297 | 0.266 | 1787 | 51 |
| ET8n-r7glWYqZhuR6GcdNw | 1144 | 0.173 | 5958 | 16 |
| ouODopBKF3AqfCkuQEnrDg | 956 | 0.163 | 1141 | 26 |

These users demonstrate extremely high review velocity compared to the median user (0.003 reviews/day), often exceeding it by 30–80 times.

**Interpretation:**  
Unusually high activity levels suggest possible coordinated behavior or automated reviewing patterns. While some may still be actual users, the scale and consistency of activity raise concerns about potential manipulation.


---

## Summary
---
* Analysis was made using median instead of mean as median is not affected by outliers

Anomaly strength is the measure of to how far a user’s behavior deviates from typical patterns:
- Higher anomaly scores -> more extreme deviations
- Lower anomaly scores -> more moderate differences

### Types
A: Power Users (Statistical Outliers)
Key Signals: High review activity + high social connectivity (4k+ friends)

- Likely legitimate, highly engaged users
- While they are mathematically anomalous due to their scale of activity, their high friend count provides social proof that distinguishes them from automated bots

B: Isolated High-Risk Users (Potential Review Bombing)
Key Signals: Zero social connections (num_friends = 0) + High 1-star review proportion

- Exhibit patterns consistent with review bombing
- Suggests accounts created primarily for reputation damage

C: High-Frequency Reviewers (Potential Automated Activity)
Key Signals: Extremely high review frequency (up to 87 times the median user)

- Likely to be coordinated or automated behavior
- ~0.26 reviews/day over the account lifetime suggests the presence of non-organic activity patterns

### Note on Social Connection and Supspicion Relationships:
Strong relationship between social connectivity (friends) and suspicion. 
The model heavily flags users with zero friends as high-risk anomalies (the red cluster at x=0), while also isolating 'Extreme Power Users' with outlier friend counts exceeding 10,000

### Note on Model Sensitivity Analysis:
The model is very sensitive to contamination. When the contamination value increases, the model labels more users as anomalies, but not all of them are extremely suspicious.
At lower contamination levels, only the most extreme outliers are identified, while higher values include more moderate deviations. A contamination level of 0.05 provides a balanced trade-off between detection coverage and anomaly strength.

### Note on Heatmap:
The negative correlation between avg_stars_given and anomaly score (-0.42) indicates that the model currently associates lower ratings with higher suspicion, suggesting that Review Bombers (accounts created for targeted negative attacks) are easier for the Isolation Forest to detect than Promotional Boosters.

Because positive reviews are the "norm" on Yelp, an account that posts fake 5-star reviews "blends in" with organic users more easily compared to someone posting aggressive 1-star reviews.
To catch these fake reviewers, we rely more heavily on the Review Velocity feature (0.50 correlation), which flags them based on how fast they post rather than what they rate.
---

*Generated from `user_anomaly_analysis.ipynb`.*