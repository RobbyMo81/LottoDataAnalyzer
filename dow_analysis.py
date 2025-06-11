# core/dow_analysis.py
import streamlit as st
import pandas as pd
import plotly.express as px
from .storage import get_store

# ----------------------------------------------------------------------
# Helper — compute counts, expected counts, and deltas
# ----------------------------------------------------------------------
def _compute_stats(df: pd.DataFrame) -> pd.DataFrame:
    df["draw_date"] = pd.to_datetime(df["draw_date"])
    df["dow"] = df["draw_date"].dt.day_name()

    # Re-index to keep weekday order; fill missing with 0
    order = ["Monday", "Tuesday", "Wednesday",
             "Thursday", "Friday", "Saturday", "Sunday"]
    counts = (
        df["dow"]
        .value_counts()
        .reindex(order)
        .fillna(0)
        .astype(int)
    )

    total = counts.sum()
    expected = total / len(order)

    out = pd.DataFrame(
        {
            "Day": counts.index,
            "Frequency": counts.values,
            "Δ_vs_expected": counts.values - expected,
        }
    )
    return out


# ----------------------------------------------------------------------
# Streamlit entry-point
# ----------------------------------------------------------------------
def render_page() -> None:
    st.header("📅 Day-of-Week Analysis")

    df = get_store().latest()          # latest versioned dataset
    
    if df.empty:
        st.warning("No data available. Please upload data in the 'Upload / Data' section.")
        return
        
    stats = _compute_stats(df)

    # ---- Table with gradient highlighting
    st.dataframe(
        stats.style.background_gradient(cmap="Blues", subset=["Frequency"])
    )

    # ---- Quick bar chart
    fig = px.bar(
        stats,
        x="Day",
        y="Frequency",
        color="Δ_vs_expected",
        color_continuous_scale="RdBu",
        title="Draw Count by Day of Week",
        labels={"Frequency": "Times Drawn"},
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---- Context / significance note
    st.caption(
        """
        **Note on significance:** Powerball drawings officially occur only on
        **Monday, Wednesday, and Saturday**. Any variation you see between those
        days is almost certainly random noise rather than a true pattern.
        """
    )
