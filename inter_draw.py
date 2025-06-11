# core/inter_draw.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from .storage import get_store

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _days_since_last(df: pd.DataFrame) -> pd.DataFrame:
    """Return days since each number last appeared (relative to most-recent draw)."""
    df = df.copy()
    df["draw_date"] = pd.to_datetime(df["draw_date"])
    latest_date = df["draw_date"].max()

    last_seen = {}          # {number: last date it appeared}
    for col in ["n1", "n2", "n3", "n4", "n5", "powerball"]:
        tmp = (
            df[["draw_date", col]]
            .rename(columns={col: "num"})
            .groupby("num")["draw_date"]
            .max()
        )
        last_seen.update(tmp.to_dict())

    data = {
        "Number":  list(last_seen.keys()),
        "Last Drawn": [last_seen[n] for n in last_seen],
    }
    out = pd.DataFrame(data)
    out["Days Since Last Drawn"] = (latest_date - out["Last Drawn"]).dt.days
    out.sort_values("Days Since Last Drawn", ascending=False, inplace=True)
    return out.reset_index(drop=True)


def _all_gaps(df: pd.DataFrame) -> pd.Series:
    """Flatten inter-draw gaps (days) for every number across history."""
    df = df.copy()
    df["draw_date"] = pd.to_datetime(df["draw_date"])
    gaps = []
    for n in range(1, 70):                # 1-69 inclusive
        hits = df[
            (df[["n1","n2","n3","n4","n5","powerball"]] == n).any(axis=1)
        ]["draw_date"].sort_values()
        if len(hits) < 2:
            continue
        diffs = hits.diff().dropna().dt.days
        gaps.extend(diffs.tolist())
    return pd.Series(gaps, name="gap_days")


# ----------------------------------------------------------------------
# Streamlit entry-point
# ----------------------------------------------------------------------
def render_page() -> None:
    st.header("⏳ Inter-Draw Gaps")

    df_raw = get_store().latest()
    
    if df_raw.empty:
        st.warning("No data available. Please upload data in the 'Upload / Data' section.")
        return
        
    stats = _days_since_last(df_raw)

    # ---- Most overdue numbers table
    st.subheader("📊 Numbers waiting the longest")
    st.dataframe(stats.head(10), use_container_width=True)

    # ---- Histogram of all gaps across history
    st.subheader("📊 Distribution of gap lengths (all numbers, all history)")
    gaps = _all_gaps(df_raw)
    
    fig = px.histogram(
        gaps, 
        nbins=30,
        labels={"value": "Gap (days)", "count": "Frequency"},
        title="Distribution of time gaps between appearances"
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---- Stats about the gap distribution
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average gap", f"{gaps.mean():.1f} days")
    with col2:
        st.metric("Median gap", f"{gaps.median():.1f} days")
    with col3:
        st.metric("Max gap ever", f"{gaps.max():.0f} days")
