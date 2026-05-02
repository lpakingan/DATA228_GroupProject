import os
from datetime import timedelta
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUTS = PROJECT_ROOT / "outputs"
DEFAULT_PROCESSED = PROJECT_ROOT / "data" / "processed"
DEFAULT_FIGURES = PROJECT_ROOT / "presentation" / "figures"


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
    st.set_page_config(page_title="Yelp Review Authenticity Dashboard", layout="wide")
    st.title("Yelp Review Authenticity Detector")

    px = _try_import_plotly()

    with st.sidebar:
        st.header("Data paths")
        outputs_dir = Path(st.text_input("outputs/", str(DEFAULT_OUTPUTS)))
        processed_dir = Path(st.text_input("data/processed/", str(DEFAULT_PROCESSED)))
        figures_dir = Path(st.text_input("presentation/figures/", str(DEFAULT_FIGURES)))

        spikes_path = outputs_dir / "review_spikes.parquet"
        suspicious_path = outputs_dir / "suspicious_businesses.parquet"
        impact_path = outputs_dir / "rating_impact.parquet"
        reviews_path = processed_dir / "restaurant_reviews.parquet"
        restaurants_path = processed_dir / "restaurants_base.parquet"
        clusters_path = outputs_dir / "user_clusters.parquet"

        st.caption("Required files:")
        st.code(
            "\n".join(
                [
                    str(spikes_path),
                    str(suspicious_path),
                    str(impact_path),
                    str(reviews_path),
                ]
            ),
            language="text",
        )

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

            if not biz_impact.empty:
                st.divider()
                st.subheader("Rating Impact")
                r = biz_impact.iloc[0].to_dict()
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Avg stars before", f"{r.get('avg_stars_before', float('nan')):.3f}")
                with m2:
                    st.metric("Avg stars during", f"{r.get('avg_stars_during', float('nan')):.3f}")
                with m3:
                    st.metric("Avg stars after", f"{r.get('avg_stars_after', float('nan')):.3f}")
                with m4:
                    d = r.get("delta_after_before", float("nan"))
                    st.metric("Δ (after - before)", f"{d:.3f}" if pd.notna(d) else "n/a")



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
        st.subheader("Explore a business")

        plot_days = st.number_input("Timeline plot range (days ±)", min_value=7, max_value=180, value=30, step=7)
        min_susp_users = st.slider("Minimum suspicious users (filter)", min_value=1, max_value=50, value=1, step=1)

        # Apply filter here, only for explorer tab
        suspicious_filtered = suspicious_with_names[
            pd.to_numeric(suspicious_with_names.get("suspicious_user_count"), errors="coerce").fillna(0) >= int(min_susp_users)
        ]
        business_ids = sorted(suspicious_filtered["business_id"].unique().tolist())

        if not business_ids:
            st.warning("No suspicious businesses found for the current filter.")
            st.stop()

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

        if quick_labels:
            default_label = quick_labels[0]
            picked = st.selectbox("Quick pick a case", options=quick_labels, index=0)
            default_business, default_spike = quick_map[picked]
            default_business_index = business_ids.index(default_business) if default_business in business_ids else 0
        else:
            default_business_index = 0
            default_spike = None

        id_to_name = (
            suspicious_filtered[["business_id", "name"]]
            .drop_duplicates()
            .set_index("business_id")["name"]
            .to_dict()
        )
        business_label_map = {bid: f"{id_to_name.get(bid, '(unknown name)')} ({bid})" for bid in business_ids}

        selected_business = st.selectbox(
            "Business",
            options=business_ids,
            index=int(default_business_index),
            format_func=lambda bid: business_label_map.get(bid, bid),
        )

        # Events for this business
        biz_events = suspicious_filtered[suspicious_filtered["business_id"] == selected_business].copy()
        biz_events = biz_events.sort_values("spike_date")

        st.write("Spike events for selected business:")
        st.dataframe(
            biz_events[
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
            ],
            width="stretch",
            hide_index=True,
        )

        spike_dates = biz_events["spike_date"].dropna().unique().tolist()
        if not spike_dates:
            st.warning("No spike_date values found for this business.")
            st.stop()

        # Choose default spike date from quick pick if it matches this business
        if default_spike is not None and str(default_business) == str(selected_business) and default_spike in spike_dates:
            default_spike_index = spike_dates.index(default_spike)
        else:
            default_spike_index = len(spike_dates) - 1

        selected_spike_date = st.selectbox("spike_date", spike_dates, index=int(default_spike_index))

        # Load review timeline for this business (only for chosen time window)
        # Use Spark parquet partitioning directly via pandas: we rely on pyarrow dataset.
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

        left, right = st.columns(2)
        with left:
            st.markdown("**Reviews per day (timeline)**")
            if px is not None:
                fig = px.line(daily_counts, x="review_date", y="reviews", markers=True)
                fig.add_vline(x=sd, line_color="red", line_dash="dash", line_width=2)
                st.plotly_chart(fig, width="stretch")
            else:
                st.line_chart(daily_counts.set_index("review_date")["reviews"])
            st.caption(f"Spike date: {sd} (vertical marker if Plotly is installed).")

        with right:
            st.markdown("**Average stars per day (timeline)**")
            if px is not None:
                fig = px.line(daily_avg, x="review_date", y="avg_stars", markers=True)
                fig.add_hline(y=0, line_color="rgba(0,0,0,0)")
                fig.add_vline(x=sd, line_color="red", line_dash="dash", line_width=2)
                st.plotly_chart(fig, width="stretch")
            else:
                st.line_chart(daily_avg.set_index("review_date")["avg_stars"])

        st.divider()
        st.subheader("Before / During / After impact (this spike event)")

        impact_row = impact_enriched[
            (impact_enriched["business_id"] == selected_business) & (impact_enriched["spike_date"] == selected_spike_date)
        ].copy()

        if impact_row.empty:
            st.warning("No matching row found in rating_impact.parquet for this business/spike_date.")
        else:
            r = impact_row.iloc[0].to_dict()
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("avg_stars_before", f"{r.get('avg_stars_before', float('nan')):.3f}", f"n={int(r.get('n_reviews_before', 0) or 0)}")
            with m2:
                st.metric("avg_stars_during", f"{r.get('avg_stars_during', float('nan')):.3f}", f"n={int(r.get('n_reviews_during', 0) or 0)}")
            with m3:
                st.metric("avg_stars_after", f"{r.get('avg_stars_after', float('nan')):.3f}", f"n={int(r.get('n_reviews_after', 0) or 0)}")
            with m4:
                d = r.get("delta_after_before", float("nan"))
                st.metric("Δ (after - before)", f"{d:.3f}" if pd.notna(d) else "n/a")

            bars = pd.DataFrame(
                {
                    "period": ["before", "during", "after"],
                    "avg_stars": [r.get("avg_stars_before"), r.get("avg_stars_during"), r.get("avg_stars_after")],
                }
            )
            if px is not None:
                fig = px.bar(bars, x="period", y="avg_stars")
                fig.update_yaxes(range=[0, 5.5])
                st.plotly_chart(fig, width="stretch")
            else:
                st.bar_chart(bars.set_index("period")["avg_stars"])

    with distributions_tab:
        st.subheader("Distributions")
        if "delta_after_before" not in impact_enriched.columns:
            st.info("rating_impact.parquet needs avg_stars_before/after to compute deltas.")
        else:
            dfd = impact_enriched.dropna(subset=["delta_after_before"]).copy()
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Histogram: Δ rating (after - before)**")
                if px is not None:
                    fig = px.histogram(dfd, x="delta_after_before", nbins=40)
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.bar_chart(dfd["delta_after_before"].value_counts(bins=20).sort_index())
            with c2:
                st.markdown("**Scatter: suspicious users vs Δ rating**")
                scat = suspicious_with_names.merge(
                    dfd[["business_id", "spike_date", "delta_after_before"]],
                    on=["business_id", "spike_date"],
                    how="inner",
                )
                if px is not None and not scat.empty:
                    fig = px.scatter(
                        scat,
                        x="suspicious_user_count",
                        y="delta_after_before",
                        hover_data=["business_id", "spike_date", "suspicious_reviews_in_window"],
                    )
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.info("Scatter needs joined suspicious + impact rows (or install plotly for interactivity).")

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

        st.divider()
        st.markdown("**Saved presentation figures (if present)**")
        if figures_dir.exists():
            pngs = sorted(figures_dir.glob("*.png"))
            if not pngs:
                st.info(f"No PNGs found in {figures_dir}")
            else:
                for p in pngs[:12]:
                    st.image(str(p), caption=p.name)
        else:
            st.info(f"No figures directory found at {figures_dir}")


if __name__ == "__main__":
    # Avoid noisy streamlit file watcher + spark issues on macOS
    os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
    main()

