# Case Studies: User Anomaly Detection

**Model:** IF (Isolation Forest), CONTAMINATION = 0.05, N_ESTIMATORS = 300, MAX_FEATURES = 1.0, RANDOM_STATE = 42  
**Input:** `outputs/user_anomaly_scores.parquet`  
**Total users scored:** 72,300 anomalies (out of 1,445,984 unique users analyzed)

---

### **Dataset Overview**
* **Data Cleanup:**
6 users have `NaN` number_of_friends and account_age_days, so would have to look at manually and was removed for our analysis.

* **Zero-friend anomalies:** 9,732

---

### **Key Finding: What Drives Anomaly Score**

#### **1. High Activity Velocity**
The **`reviews_per_day`** is a leading indicator of anomaly.

**Correlation with Anomaly Score:**
* `anomaly_score`: 1.000000
* `reviews_per_day`: 0.501998
* `review_count`: 0.471799
* `num_friends`: 0.388648

> **What this means:** The model aggressively flags users who post a lot of reviews in a short amount of time (**hyper-active**).

#### **2. Sentiment and "Review Bombing"**
There is a negative correlation between the anomaly score and **`avg_stars_given` (-0.42)** and **`pct_5_star_reviews` (-0.40)**.

> **What this means:** As the star rating goes down, the anomaly score goes up and the model is sensitive to **"review bombing" of 1-star reviews**. These accounts are more anomalous than those posting positive reviews.

#### **3. Social Behavior**
The correlation with **`num_friends` is 0.39**.

> **What this means:** This suggests that extreme social behavior is being flagged -- either having a massive number of friends (fake followers) or a very specific, rare number of friends. The users with many friends are flagged as top anomalous users.

#### **4. Positive Bias**
Users who only gives out 5-star reviews exist.

---

*Generated from `user_anomaly_analysis.ipynb`.*
