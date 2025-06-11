# core/sums.py
"""
Sum Analysis
------------
• Shows the distribution of the sum of the five white balls
  (optionally include Powerball).
• Lets users enter a custom sum and see where it sits
  in the historical percentile ranks.
• Visualises how the average sum has drifted over time.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from .storage import get_store

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _prep(df: pd.DataFrame, include_pb: bool) -> pd.Series:
    cols = ["n1", "n2", "n3", "n4", "n5"] + (["powerball"] if include_pb else [])
    return df[cols].sum(axis=1).rename("sum")


def _percentile(series: pd.Series, value: int) -> float:
    # manual to avoid scipy dep
    return 100.0 * (series < value).mean()


# ----------------------------------------------------------------------
# Streamlit entry-point
# ----------------------------------------------------------------------
def render_page() -> None:
    st.header("➕ Sum Analysis")

    df_raw = get_store().latest()
    
    if df_raw.empty:
        st.warning("No data available. Please upload data in the 'Upload / Data' section.")
        return

    include_pb = st.checkbox("Include Powerball", value=False)
    sums = _prep(df_raw, include_pb)

    # ---- Histogram
    fig = px.histogram(
        sums,
        nbins=30,
        title="Distribution of draw sums",
        labels={"value": "Sum of balls", "count": "Frequency"},
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---- User-entered sum → percentile feedback
    st.subheader("Where does *your* sum sit?")
    max_possible = 69*5 + (26 if include_pb else 0)
    user_sum = st.number_input(
        "Enter a sum",
        min_value=int(sums.min()),
        max_value=int(sums.max()),
        step=1,
    )
    pct = _percentile(sums, user_sum)
    st.metric(
        label=f"Percentile rank (lower = rarer)",
        value=f"{pct:.1f}-th",
        delta=None,
    )
    st.progress(pct / 100.0)

    # ---- Average sum over time
    st.subheader("Average sum over time")
    df_time = df_raw.copy()
    df_time["draw_date"] = pd.to_datetime(df_time["draw_date"])
    df_time["year"] = df_time["draw_date"].dt.year
    df_time["sum"] = df_time[["n1", "n2", "n3", "n4", "n5"] + (["powerball"] if include_pb else [])].sum(axis=1)
    
    yearly_avg = df_time.groupby("year")["sum"].mean().reset_index()
    
    fig2 = px.line(
        yearly_avg,
        x="year",
        y="sum",
        title="Average sum by year",
        labels={"year": "Year", "sum": "Average Sum"}
    )
    st.plotly_chart(fig2, use_container_width=True)
