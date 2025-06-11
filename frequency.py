"""
Frequency analysis for Powerball numbers.
"""

import streamlit as st
import pandas as pd 
import numpy as np
import plotly.express as px
from .storage import get_store

MAX_NUMBER = 69  # Powerball numbers go from 1 to 69
NUMBER_COLUMNS = ["n1", "n2", "n3", "n4", "n5", "powerball"]

def calc_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the frequency of lottery numbers from the given DataFrame.
    Assumes the DataFrame contains the columns: 'n1', 'n2', 'n3', 'n4', 'n5', and 'powerball'.
    Returns a DataFrame with the following columns:
        - number: The lottery number (1 to 69).
        - frequency: How many times the number was drawn.
        - Δ_from_expected: The difference between the observed count and the expected count.
    """
    # Initialize counts for numbers 1 through MAX_NUMBER
    number_range = range(1, MAX_NUMBER + 1)
    counts = pd.Series(0, index=number_range, dtype=np.int64)

    # Check for missing columns
    missing_cols = [col for col in NUMBER_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in DataFrame: {missing_cols}")

    # Sum the frequency from each column; reindex in case value_counts() misses some numbers
    for col in NUMBER_COLUMNS:
        col_counts = df[col].value_counts().reindex(number_range, fill_value=0)
        counts += col_counts

    # Calculate the expected frequency given uniform distribution across numbers
    expected = counts.sum() / len(counts)

    # Build the DataFrame with counts, expected delta, and sort by frequency (descending) then number (ascending)
    freq_df = pd.DataFrame({
        "number": counts.index,
        "frequency": counts.values,
        "Δ_from_expected": counts.values - expected
    })
    freq_df = freq_df.sort_values(by=["frequency", "number"], ascending=[False, True])
    return freq_df

def render_page() -> None:
    """
    Render the Streamlit page showing the frequency table and a corresponding bar chart.
    """
    st.header("🎲 Number Frequency")

    # Retrieve the latest data
    store = get_store()
    df = store.latest()
    
    if df.empty:
        st.warning("No data available. Please upload data in the 'Upload / Data' section.")
        return

    # Calculate frequency statistics
    frequency_df = calc_frequency(df)

    # Display a table with a background gradient for visual appeal
    st.dataframe(frequency_df.style.background_gradient(cmap="Greens"))

    # Create a bar chart to display the frequency and delta from expected frequency
    fig = px.bar(
        frequency_df,
        x="number",
        y="frequency",
        color="Δ_from_expected",
        color_continuous_scale="RdBu",
        labels={"frequency": "Times drawn", "number": "Number"}
    )
    st.plotly_chart(fig, use_container_width=True)
