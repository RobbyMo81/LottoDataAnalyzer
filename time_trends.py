# core/time_trends.py
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from .storage import get_store

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _aggregate(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Return a long-format table with draw counts per number
    resampled to the chosen frequency (`ME`, `W`, `Y`, etc.).
    """
    df = df.copy()
    df["draw_date"] = pd.to_datetime(df["draw_date"])
    df.set_index("draw_date", inplace=True)

    # Explode to one row per single ball
    long = (
        df[["n1", "n2", "n3", "n4", "n5", "powerball"]]
        .melt(ignore_index=False, value_name="num")["num"]
        .to_frame()
    )

    # Count per period
    counts = long.groupby(
        [pd.Grouper(freq=freq), "num"], observed=True
    ).size()

    return (
        counts.rename("frequency")
        .reset_index()
        .pivot(index="draw_date", columns="num", values="frequency")
        .fillna(0)
        .sort_index()
    )


def _calc_trending(df: pd.DataFrame, window: int = 6) -> pd.Series:
    """
    Simple 'trending' score: difference between the rolling sum
    in the most-recent window and the previous window of equal size.
    """
    recent = df.tail(window).sum()
    prev   = df.iloc[-2*window:-window].sum() if len(df) >= 2*window else 0
    return (recent - prev).sort_values(ascending=False)


# ----------------------------------------------------------------------
# Streamlit entry-point
# ----------------------------------------------------------------------
def render_page() -> None:
    st.header("📈 Time-Based Trends")

    df_raw = get_store().latest()
    
    if df_raw.empty:
        st.warning("No data available. Please upload data in the 'Upload / Data' section.")
        return

    # 🔒 Ensure draw_date is datetime even if the CSV slipped through as string
    df_raw["draw_date"] = pd.to_datetime(df_raw["draw_date"], errors="coerce")
    df_raw = df_raw.dropna(subset=["draw_date"])   # toss rows that still failed

    # ---- Date-range selector
    min_d, max_d = pd.to_datetime(df_raw["draw_date"]).agg(["min", "max"])
    start, end = st.date_input(
        "Select date range",
        value=(min_d.date(), max_d.date()),
        min_value=min_d.date(),
        max_value=max_d.date(),
    )
    if start > end:
        st.error("Start date must be before end date.")
        return

    mask = (df_raw["draw_date"] >= pd.Timestamp(start)) & (
        df_raw["draw_date"] <= pd.Timestamp(end)
    )
    df = df_raw.loc[mask]
    if df.empty:
        st.warning("No draws in that range.")
        return

    # ---- Granularity selector
    freq_map = {"Monthly (ME)": "ME", "Weekly (W)": "W", "Yearly (Y)": "Y"}
    gran = st.selectbox("Aggregation period", list(freq_map.keys()))
    agg_df = _aggregate(df, freq_map[gran])

    # ---- Line chart of total numbers per period
    fig = px.line(
        agg_df.sum(axis=1).to_frame("Total draws"),
        title=f"Total balls drawn – {gran.split()[0].lower()}",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---- Trending numbers
    with st.expander("Show trending numbers"):
        win = st.slider("Rolling window (# periods)", 3, 12, 6)
        trend = _calc_trending(agg_df, win)
        st.write(
            "Top 10 up-trend numbers "
            f"(difference between last {win} periods and prior {win}):"
        )
        st.table(trend.head(10).rename("Δ frequency"))

    st.caption(
        """
        **Trend definition:** The table compares the sum of occurrences
        in the most-recent *N* periods versus the preceding *N* periods.
        Positive values mean the number has appeared more often lately.
        """
    )
