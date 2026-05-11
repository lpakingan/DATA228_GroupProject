import os
from datetime import timedelta
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = PROJECT_ROOT / "assets"
YELP_ICON_PNG = ASSETS_DIR / "yelp_icon.png"
DEFAULT_OUTPUTS = PROJECT_ROOT / "outputs"
DEFAULT_PROCESSED = PROJECT_ROOT / "data" / "processed"


@st.cache_data(show_spinner=False)
def load_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)

@st.cache_data(show_spinner=False)
def load_business_names(restaurants_base_path: str) -> pd.DataFrame:
    df = pd.read_parquet(restaurants_base_path, columns=["business_id", "name"])
    df["business_id"] = df["business_id"].astype(str)
    df["name"] = df["name"].astype(str)
    return df.drop_duplicates(subset=["business_id"])


def _try_import_plotly():
    try:
        import plotly.express as px  # type: ignore

        return px
    except Exception:
        return None


def _delta_columns(impact: pd.DataFrame) -> pd.DataFrame:
    df = impact.copy()
    for c in ["avg_stars_before", "avg_stars_during", "avg_stars_after"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "avg_stars_before" in df.columns and "avg_stars_after" in df.columns:
        df["delta_after_before"] = df["avg_stars_after"] - df["avg_stars_before"]
    return df


def main() -> None:
    _page_icon = str(YELP_ICON_PNG) if YELP_ICON_PNG.exists() else "🍽️"
    st.set_page_config(
        page_title="Yelp Review Authenticity Dashboard",
        layout="wide",
        page_icon=_page_icon,
    )

    if YELP_ICON_PNG.exists():
        head_icon, head_title = st.columns([0.06, 0.94], vertical_alignment="center")
        with head_icon:
            st.image(str(YELP_ICON_PNG), width=52)
        with head_title:
            st.markdown("# Yelp Review Authenticity Detector")
    else:
        st.title("Yelp Review Authenticity Detector")

    px = _try_import_plotly()

    with st.sidebar:
        if YELP_ICON_PNG.exists():
            st.image(str(YELP_ICON_PNG), width=44)
        st.header("About")
        st.markdown("""
        This dashboard detects suspicious review activity on Yelp restaurants using unsupervised machine learning.
        
        **How the authenticity score works:**
        - Review spikes detected — 20pts
        - Anomalous users flagged — 20pts
        - Users in a review farm cluster — 40pts
        - Rating changed during spike — 20pts
        
        A score ≥ 50 indicates suspicious activity.
        """)
        st.divider()
        st.caption("DATA 228 — Group 5")

    outputs_dir = DEFAULT_OUTPUTS
    processed_dir = DEFAULT_PROCESSED

    spikes_path = outputs_dir / "review_spikes.parquet"
    suspicious_path = outputs_dir / "suspicious_businesses.parquet"
    impact_path = outputs_dir / "rating_impact.parquet"
    reviews_path = processed_dir / "restaurant_reviews.parquet"
    restaurants_path = processed_dir / "restaurants_base.parquet"
    clusters_path = outputs_dir / "user_clusters.parquet"

    missing = [p for p in [spikes_path, suspicious_path, impact_path, reviews_path, restaurants_path, clusters_path] if not p.exists()]
    if missing:
        st.error("Missing required data files:")
        st.write([str(p) for p in missing])
        st.stop()

    # Load outputs
    spikes = load_parquet(str(spikes_path))
    suspicious = load_parquet(str(suspicious_path))
    impact = load_parquet(str(impact_path))
    restaurants = load_business_names(str(restaurants_path))
    clusters = load_parquet(str(clusters_path))
    biz_reviews = load_parquet(str(reviews_path))
    anomaly_scores = load_parquet(str(suspicious_path.parent / "user_anomaly_scores.parquet"))

    # Minimal cleanup
    for df in [spikes, suspicious, impact]:
        if "business_id" in df.columns:
            df["business_id"] = df["business_id"].astype(str)

    # Base filtered dataframe — no min_susp_users filter, used by overview and demo tabs
    suspicious_with_names = suspicious[
        pd.to_numeric(suspicious.get("suspicious_user_count"), errors="coerce").fillna(0) > 0
    ].copy()
    suspicious_with_names = suspicious_with_names.merge(restaurants, on="business_id", how="left")
    suspicious_with_names["name"] = suspicious_with_names["name"].fillna("(unknown name)")

    impact_enriched = _delta_columns(impact)

    demo_tab, overview_tab, explorer_tab, distributions_tab, exports_tab = st.tabs(
        ["Demo", "Overview", "Case explorer", "Distributions", "Exports"]
    )

    with demo_tab:
        st.subheader("Business Authenticity Checker")
        
        # search box
        business_search = st.selectbox(
            "Search for a restaurant",
            options=restaurants["name"].sort_values().unique(),
        )
        run_check = st.button("Check Authenticity")

        if run_check:
            business_id = restaurants[restaurants["name"] == business_search]["business_id"].iloc[0]
            biz_suspicious = suspicious_with_names[suspicious_with_names["business_id"] == business_id]
            biz_user_ids = biz_reviews[biz_reviews["business_id"] == business_id]["user_id"].unique()
            biz_anomalous_users = anomaly_scores[(anomaly_scores["user_id"].isin(biz_user_ids)) & (anomaly_scores["is_anomaly"] == 1)]
            biz_clusters = clusters[clusters["user_id"].isin(biz_anomalous_users["user_id"])] if not biz_anomalous_users.empty else pd.DataFrame()
            biz_impact = impact_enriched[impact_enriched["business_id"] == business_id]

            score = 0

            # has review spikes
            if not biz_suspicious.empty:
                score += 20

            # has anomalous users
            if not biz_suspicious.empty and biz_suspicious["suspicious_user_count"].sum() > 0:
                score += 20

            # users in a review farm cluster
            if not biz_clusters.empty:
                score += 40

            # rating changed significantly during spike
            if not biz_impact.empty and "delta_after_before" in biz_impact.columns:
                if biz_impact["delta_after_before"].abs().max() > 0.3:
                    score += 20
           
            st.divider()
            if score >= 50:
                st.error(f"⚠️ Suspicious Activity Detected — Authenticity Score: {score}/100")
            else:
                st.success(f"✅ No Suspicious Activity Found — Authenticity Score: {score}/100")

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Review Spikes", len(biz_suspicious))
            with c2:
                st.metric("Suspicious Users", int(biz_suspicious["suspicious_user_count"].sum()) if not biz_suspicious.empty else 0)
            with c3:
                st.metric("Review Farm Clusters", biz_clusters["cluster_id"].nunique() if not biz_clusters.empty else 0)

            st.divider()
            st.subheader("Rating Impact")
            if not biz_impact.empty:
                valid_impact = biz_impact.dropna(subset=["delta_after_before"])
                if not valid_impact.empty:
                    r = valid_impact.loc[valid_impact["delta_after_before"].abs().idxmax()].to_dict()
                else:
                    r = biz_impact.iloc[0].to_dict()
            else:
                r = {}

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Avg stars before", f"{r.get('avg_stars_before', float('nan')):.3f}" if r.get('avg_stars_before') else "n/a")
            with m2:
                st.metric("Avg stars during", f"{r.get('avg_stars_during', float('nan')):.3f}" if r.get('avg_stars_during') else "n/a")
            with m3:
                st.metric("Avg stars after", f"{r.get('avg_stars_after', float('nan')):.3f}" if r.get('avg_stars_after') else "n/a")
            with m4:
                d = r.get("delta_after_before", float("nan"))
                st.metric("Δ (after - before)", f"{d:.3f}" if pd.notna(d) else "n/a")
            st.divider()
            st.subheader("Review Timeline")
            demo_reviews = biz_reviews[biz_reviews["business_id"] == business_id].copy()
            demo_reviews["review_date"] = pd.to_datetime(demo_reviews["date"]).dt.date
            
            daily_counts = (
                demo_reviews.groupby("review_date", as_index=False)
                .size()
                .rename(columns={"size": "reviews"})
                .sort_values("review_date")
            )
            
            if px is not None:
                max_reviews = max(daily_counts["reviews"].max() * 1.2, 5)
                fig = px.line(daily_counts, x="review_date", y="reviews", markers=True, 
                            title="Reviews per day",
                            range_y=[0, max_reviews])
                if not biz_suspicious.empty:
                    for _, spike_row in biz_suspicious.iterrows():
                        fig.add_vline(x=pd.to_datetime(spike_row["spike_date"]), 
                                    line_color="red", line_dash="dash", line_width=2)
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Red lines indicate detected spike dates.")



    with overview_tab:
        st.subheader("Overview")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Spike days", int(len(suspicious)))
        with c2:
            st.metric("Spike days w/ suspicious users", int(len(suspicious_with_names)))
        with c3:
            st.metric("Unique suspicious businesses", int(suspicious_with_names["business_id"].nunique()))
        with c4:
            st.metric("Impact events", int(len(impact)))

        st.divider()
        st.subheader("Top cases (quick picks)")
        pick_df = suspicious_with_names.merge(
            impact_enriched,
            on=["business_id", "spike_date"],
            how="inner",
            suffixes=("", "_impact"),
        )
        if not pick_df.empty and "delta_after_before" in pick_df.columns:
            pick_df = pick_df.sort_values(
                ["suspicious_user_count", "delta_after_before"],
                ascending=[False, False],
            ).head(25)
            st.dataframe(
                pick_df[
                    [
                        "name",
                        "business_id",
                        "spike_date",
                        "suspicious_user_count",
                        "suspicious_reviews_in_window",
                        "review_count",
                        "delta_after_before",
                        "avg_stars_before",
                        "avg_stars_during",
                        "avg_stars_after",
                    ]
                ],
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("Not enough joined data to rank cases (check suspicious_businesses + rating_impact).")

    with explorer_tab:
        st.markdown("### Case explorer — inspect one restaurant")
        st.markdown(
            "Adjust filters, pick a **restaurant** and **spike date**, then review **daily volume**, **daily average stars**, "
            "and **before / during / after** windows from `rating_impact.parquet`."
        )

        with st.container(border=True):
            st.markdown("##### View settings")
            fc1, fc2 = st.columns(2)
            with fc1:
                plot_days = st.number_input(
                    "Timeline window (days before & after spike)",
                    min_value=7,
                    max_value=180,
                    value=30,
                    step=7,
                    help="Charts show reviews within ± this many days of the focused spike date.",
                )
            with fc2:
                min_susp_users = st.slider(
                    "Minimum suspicious users (list filter)",
                    min_value=1,
                    max_value=50,
                    value=1,
                    step=1,
                    help="Only restaurants with at least this many flagged reviewers appear in the lists below.",
                )

        # Apply filter here, only for explorer tab
        suspicious_filtered = suspicious_with_names[
            pd.to_numeric(suspicious_with_names.get("suspicious_user_count"), errors="coerce").fillna(0)
            >= int(min_susp_users)
        ]
        business_ids = sorted(suspicious_filtered["business_id"].unique().tolist())

        if not business_ids:
            st.warning("No restaurants match this filter — try lowering the minimum suspicious users.")
            st.stop()

        st.caption(f"**{len(business_ids):,}** restaurants match the filter.")

        # Quick pick -> sets session selections
        pick_df = suspicious_filtered.merge(
            impact_enriched,
            on=["business_id", "spike_date"],
            how="inner",
            suffixes=("", "_impact"),
        )
        quick_labels: list[str] = []
        quick_map: dict[str, tuple[str, object]] = {}
        if not pick_df.empty and "delta_after_before" in pick_df.columns:
            pick_df = pick_df.sort_values(
                ["suspicious_user_count", "delta_after_before"],
                ascending=[False, False],
            ).head(25)
            for _, r in pick_df.iterrows():
                nm = str(r.get("name", "")).strip()
                label = f"{nm} | {r['business_id']} | {r['spike_date']} | susp_users={int(r['suspicious_user_count'])} | Δ={r['delta_after_before']:.2f}"
                quick_labels.append(label)
                quick_map[label] = (str(r["business_id"]), r["spike_date"])

        default_business = None
        default_spike = None
        default_business_index = 0

        if quick_labels:
            picked = st.selectbox(
                "Quick pick (high-impact cases)",
                options=quick_labels,
                index=0,
                help="Sorted by suspicious users and Δ rating. Updates restaurant / spike defaults.",
            )
            default_business, default_spike = quick_map[picked]
            default_business_index = business_ids.index(default_business) if default_business in business_ids else 0

        id_to_name = (
            suspicious_filtered[["business_id", "name"]]
            .drop_duplicates()
            .set_index("business_id")["name"]
            .to_dict()
        )
        business_label_map = {bid: f"{id_to_name.get(bid, '(unknown name)')} · {bid}" for bid in business_ids}

        selected_business = st.selectbox(
            "Restaurant",
            options=business_ids,
            index=int(default_business_index),
            format_func=lambda bid: business_label_map.get(bid, bid),
            help="Restaurants that passed the suspicious-user filter.",
        )

        # Events for this business
        biz_events = suspicious_filtered[suspicious_filtered["business_id"] == selected_business].copy()
        biz_events = biz_events.sort_values("spike_date")

        ev_h, ev_m = st.columns([3, 1])
        with ev_h:
            st.markdown("##### Spike events for this restaurant")
        with ev_m:
            st.metric("Events", len(biz_events))

        events_display = biz_events[
            [
                "spike_date",
                "review_count",
                "baseline_mean",
                "abs_increase",
                "total_reviews_in_window",
                "suspicious_reviews_in_window",
                "suspicious_user_count",
                "max_anomaly_score",
            ]
        ].rename(
            columns={
                "spike_date": "Spike date",
                "review_count": "Reviews on spike day",
                "baseline_mean": "Baseline avg",
                "abs_increase": "Increase vs baseline",
                "total_reviews_in_window": "Reviews in window",
                "suspicious_reviews_in_window": "Suspicious reviews",
                "suspicious_user_count": "Suspicious users",
                "max_anomaly_score": "Max anomaly score",
            }
        )
        ecfg = {
            "Reviews on spike day": st.column_config.NumberColumn(format="%d"),
            "Baseline avg": st.column_config.NumberColumn(format="%.2f"),
            "Increase vs baseline": st.column_config.NumberColumn(format="%.2f"),
            "Reviews in window": st.column_config.NumberColumn(format="%d"),
            "Suspicious reviews": st.column_config.NumberColumn(format="%d"),
            "Suspicious users": st.column_config.NumberColumn(format="%d"),
            "Max anomaly score": st.column_config.NumberColumn(format="%.3f"),
        }
        st.dataframe(events_display, column_config=ecfg, width="stretch", hide_index=True)
        st.caption("One row per detected spike for the selected restaurant.")

        spike_dates = biz_events["spike_date"].dropna().unique().tolist()
        if not spike_dates:
            st.warning("No spike dates found for this restaurant.")
            st.stop()

        # Choose default spike date from quick pick if it matches this business
        if (
            default_spike is not None
            and default_business is not None
            and str(default_business) == str(selected_business)
            and default_spike in spike_dates
        ):
            default_spike_index = spike_dates.index(default_spike)
        else:
            default_spike_index = len(spike_dates) - 1

        selected_spike_date = st.selectbox(
            "Focus spike date (timelines & impact)",
            spike_dates,
            index=int(default_spike_index),
            help="All charts and metrics below use this spike.",
        )

        # Load review timeline for this business (only for chosen time window)
        reviews = biz_reviews.copy()
        reviews = reviews[["business_id", "date", "stars"]].copy()
        reviews["business_id"] = reviews["business_id"].astype(str)
        reviews = reviews[reviews["business_id"] == selected_business]
        reviews["review_date"] = pd.to_datetime(reviews["date"]).dt.date

        sd = pd.to_datetime(selected_spike_date).date()
        start = sd - timedelta(days=int(plot_days))
        end = sd + timedelta(days=int(plot_days))
        reviews = reviews[(reviews["review_date"] >= start) & (reviews["review_date"] <= end)]

        daily_counts = (
            reviews.groupby("review_date", as_index=False)
            .size()
            .rename(columns={"size": "reviews"})
            .sort_values("review_date")
        )
        daily_avg = (
            reviews.groupby("review_date", as_index=False)["stars"]
            .mean()
            .rename(columns={"stars": "avg_stars"})
            .sort_values("review_date")
        )

        st.markdown("##### Timelines around the focused spike")
        st.caption(f"Window **{start}** → **{end}** · Spike **{sd}** (red dashed line).")

        left, right = st.columns(2, gap="large")
        with left:
            if px is not None:
                fig_vol = px.line(
                    daily_counts,
                    x="review_date",
                    y="reviews",
                    markers=True,
                    color_discrete_sequence=["#2563eb"],
                )
                fig_vol.update_layout(
                    template="plotly_white",
                    title=dict(text="<b>Daily review volume</b>", font=dict(size=14), x=0.02, xanchor="left"),
                    xaxis_title="Date",
                    yaxis_title="Reviews",
                    margin=dict(l=48, r=28, t=56, b=48),
                    height=380,
                    showlegend=False,
                )
                fig_vol.add_vline(x=sd, line_color="#dc2626", line_dash="dash", line_width=2)
                st.plotly_chart(fig_vol, width="stretch")
            else:
                st.line_chart(daily_counts.set_index("review_date")["reviews"])

        with right:
            if px is not None:
                fig_stars = px.line(
                    daily_avg,
                    x="review_date",
                    y="avg_stars",
                    markers=True,
                    color_discrete_sequence=["#0d9488"],
                )
                fig_stars.update_layout(
                    template="plotly_white",
                    title=dict(text="<b>Daily average stars</b>", font=dict(size=14), x=0.02, xanchor="left"),
                    xaxis_title="Date",
                    yaxis_title="Average stars",
                    yaxis=dict(range=[1.0, 5.0]),
                    margin=dict(l=48, r=28, t=56, b=48),
                    height=380,
                    showlegend=False,
                )
                fig_stars.add_vline(x=sd, line_color="#dc2626", line_dash="dash", line_width=2)
                st.plotly_chart(fig_stars, width="stretch")
            else:
                st.line_chart(daily_avg.set_index("review_date")["avg_stars"])

        st.divider()
        st.markdown("##### Before · during · after — rating impact")
        st.caption("From `rating_impact.parquet` for the **focused** spike date.")

        impact_row = impact_enriched[
            (impact_enriched["business_id"] == selected_business) & (impact_enriched["spike_date"] == selected_spike_date)
        ].copy()

        if impact_row.empty:
            st.warning("No matching row in rating_impact.parquet for this restaurant and spike date.")
        else:
            r = impact_row.iloc[0].to_dict()
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric(
                    "Avg stars — before",
                    f"{r.get('avg_stars_before', float('nan')):.3f}",
                    f"n = {int(r.get('n_reviews_before', 0) or 0)} reviews",
                )
            with m2:
                st.metric(
                    "Avg stars — during",
                    f"{r.get('avg_stars_during', float('nan')):.3f}",
                    f"n = {int(r.get('n_reviews_during', 0) or 0)} reviews",
                )
            with m3:
                st.metric(
                    "Avg stars — after",
                    f"{r.get('avg_stars_after', float('nan')):.3f}",
                    f"n = {int(r.get('n_reviews_after', 0) or 0)} reviews",
                )
            with m4:
                d = r.get("delta_after_before", float("nan"))
                st.metric(
                    "Δ rating (after − before)",
                    f"{d:.3f}" if pd.notna(d) else "n/a",
                    help="Post-spike minus pre-spike window averages.",
                )

            bars = pd.DataFrame(
                {
                    "period": ["Before", "During", "After"],
                    "avg_stars": [r.get("avg_stars_before"), r.get("avg_stars_during"), r.get("avg_stars_after")],
                }
            )
            if px is not None:
                fig_b = px.bar(
                    bars,
                    x="period",
                    y="avg_stars",
                    color="period",
                    color_discrete_map={
                        "Before": "#64748b",
                        "During": "#dc2626",
                        "After": "#2563eb",
                    },
                )
                fig_b.update_layout(
                    template="plotly_white",
                    title=dict(text="<b>Average stars by window</b>", font=dict(size=14), x=0.02, xanchor="left"),
                    xaxis_title="Window",
                    yaxis_title="Average stars",
                    yaxis=dict(range=[0, 5.5]),
                    showlegend=False,
                    margin=dict(l=48, r=28, t=56, b=48),
                    height=400,
                )
                st.plotly_chart(fig_b, width="stretch")
            else:
                st.bar_chart(bars.set_index("period")["avg_stars"])

    with distributions_tab:
        st.markdown("### Distributions — rating shift & suspicious activity")
        st.markdown(
            "These charts summarize **all spike events** in `rating_impact.parquet`: how much **average stars** "
            "moved between the **before** and **after** windows around each spike, and whether larger shifts "
            "coincide with **more flagged reviewers** in the same window (exploratory; not causal)."
        )

        with st.expander("Definitions", expanded=True):
            st.markdown(
                """
| Term | Meaning |
|------|---------|
| **Δ rating** | `avg_stars_after − avg_stars_before` for that business and spike date (windows defined in `rating_impact.py`). **Positive** = stars went up after the spike period vs before; **negative** = went down. |
| **Suspicious users** | Distinct anomaly-flagged accounts with reviews in the spike join window (`suspicious_businesses.parquet`). |

**Note:** Δ rating describes **observed** averages over calendar windows — not a simulation of “ratings if fake reviews were removed.”
                """
            )

        if "delta_after_before" not in impact_enriched.columns:
            st.info("rating_impact.parquet needs avg_stars_before/after to compute deltas.")
        else:
            dfd = impact_enriched.dropna(subset=["delta_after_before"]).copy()
            scat = suspicious_with_names.merge(
                dfd[["business_id", "spike_date", "delta_after_before"]],
                on=["business_id", "spike_date"],
                how="inner",
            )

            k1, k2, k3, k4, k5 = st.columns(5)
            med = dfd["delta_after_before"].median()
            mean_d = dfd["delta_after_before"].mean()
            lo, hi = dfd["delta_after_before"].min(), dfd["delta_after_before"].max()
            with k1:
                st.metric("Events (rating impact)", f"{len(dfd):,}")
            with k2:
                st.metric("Median Δ rating", f"{med:+.3f}", help="Middle value of after−before stars")
            with k3:
                st.metric("Mean Δ rating", f"{mean_d:+.3f}")
            with k4:
                st.metric("Min / max Δ", f"{lo:.2f} / {hi:.2f}")
            with k5:
                st.metric("Joined scatter points", f"{len(scat):,}" if not scat.empty else "0")

            st.divider()

            c1, c2 = st.columns(2, gap="large")
            with c1:
                st.markdown("##### 1 — How large are rating shifts?")
                st.caption(
                    "Histogram of **Δ rating** per event. The **box plot** above the bars shows median and spread. "
                    "**Gray dashed line** = no net change (Δ = 0)."
                )
                if px is not None:
                    try:
                        fig_h = px.histogram(
                            dfd,
                            x="delta_after_before",
                            nbins=44,
                            marginal="box",
                            color_discrete_sequence=["#2563eb"],
                        )
                    except Exception:
                        fig_h = px.histogram(
                            dfd,
                            x="delta_after_before",
                            nbins=44,
                            color_discrete_sequence=["#2563eb"],
                        )
                    fig_h.update_layout(
                        template="plotly_white",
                        bargap=0.06,
                        title=dict(
                            text="<b>Distribution of Δ rating</b> <sup>(stars after − stars before)</sup>",
                            font=dict(size=15),
                            x=0.02,
                            xanchor="left",
                        ),
                        xaxis_title="Δ rating (average stars)",
                        yaxis_title="Number of spike events",
                        showlegend=False,
                        margin=dict(l=56, r=28, t=72, b=56),
                        height=440,
                    )
                    fig_h.update_traces(opacity=0.92)
                    fig_h.add_vline(
                        x=0,
                        line_width=2,
                        line_dash="dash",
                        line_color="#94a3b8",
                    )
                    st.plotly_chart(fig_h, width="stretch")
                else:
                    st.bar_chart(dfd["delta_after_before"].value_counts(bins=24).sort_index())

            with c2:
                st.markdown("##### 2 — Suspicious users vs. rating shift")
                st.caption(
                    "Each **dot** is one spike event with both suspicion counts and impact. **Hover** for business id, "
                    "date, and review counts. Use this to **spot outliers** to discuss live — correlation ≠ causation."
                )
                if px is not None and not scat.empty:
                    hover_cols = [
                        "name",
                        "business_id",
                        "spike_date",
                        "suspicious_reviews_in_window",
                    ]
                    hover_cols = [c for c in hover_cols if c in scat.columns]
                    fig_s = px.scatter(
                        scat,
                        x="suspicious_user_count",
                        y="delta_after_before",
                        hover_data=hover_cols,
                        color_discrete_sequence=["#0d9488"],
                    )
                    fig_s.update_traces(
                        marker=dict(size=11, opacity=0.72, line=dict(width=0.6, color="white")),
                    )
                    fig_s.update_layout(
                        template="plotly_white",
                        title=dict(
                            text="<b>Suspicious reviewer count vs. Δ rating</b>",
                            font=dict(size=15),
                            x=0.02,
                            xanchor="left",
                        ),
                        xaxis_title="Suspicious users (distinct, in window)",
                        yaxis_title="Δ rating (average stars after − before)",
                        margin=dict(l=56, r=28, t=72, b=56),
                        height=440,
                    )
                    fig_s.add_hline(
                        y=0,
                        line_width=2,
                        line_dash="dash",
                        line_color="#94a3b8",
                    )
                    st.plotly_chart(fig_s, width="stretch")
                elif px is None:
                    st.info("Install **plotly** for interactive charts (`pip install plotly`).")
                else:
                    st.info(
                        "No overlapping rows between suspicious cases and rating impact for this dataset — "
                        "scatter needs matching `(business_id, spike_date)` in both outputs."
                    )

            st.success(
                "**Presenter tip:** Start with the histogram (typical shifts), then the scatter (exceptions). "
                "Call out that large **Δ** with few suspicious users can still be organic buzz; many suspicious users "
                "with small **Δ** may reflect detection without huge rating moves."
            )

    with exports_tab:
        st.subheader("Exports")
        joined = suspicious_with_names.merge(
            impact_enriched,
            on=["business_id", "spike_date"],
            how="left",
            suffixes=("", "_impact"),
        )
        st.download_button(
            "Download joined cases CSV",
            data=joined.to_csv(index=False).encode("utf-8"),
            file_name="joined_suspicious_cases.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    # Avoid noisy streamlit file watcher + spark issues on macOS
    os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
    main()

