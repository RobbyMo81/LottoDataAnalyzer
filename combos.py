# core/combos.py
"""
Combinatorial analysis – pairs, triplets, or larger.
Provides optional filtering by:
  • combo size (2-5 white balls, Powerball optional)
  • specific number inclusion
  • minimum frequency threshold
"""
import streamlit as st
import pandas as pd
import itertools
from collections import Counter
import plotly.express as px
from .storage import get_store

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
WHITE_COLS = ["n1", "n2", "n3", "n4", "n5"]

def _count_combos(df: pd.DataFrame, k: int, include_pb: bool) -> Counter:
    """
    Return Counter mapping combo tuple -> occurrences.
    Each tuple is sorted ascending so (3,12) and (12,3) are equivalent.
    """
    combos = Counter()
    cols = WHITE_COLS + (["powerball"] if include_pb else [])
    for _, row in df[cols].iterrows():
        balls = row.tolist()
        # Ensure ints & uniqueness
        balls = list(map(int, balls))
        for c in itertools.combinations(balls, k):
            combos[tuple(sorted(c))] += 1
    return combos


def _to_df(counter: Counter) -> pd.DataFrame:
    data = {"Combo": [], "Frequency": []}
    for combo, freq in counter.items():
        data["Combo"].append(", ".join(map(str, combo)))
        data["Frequency"].append(freq)
    return (
        pd.DataFrame(data)
        .sort_values(["Frequency", "Combo"], ascending=[False, True])
        .reset_index(drop=True)
    )

# ----------------------------------------------------------------------
# Streamlit entry-point
# ----------------------------------------------------------------------
def render_page() -> None:
    st.header("🧩 Combinatorial Analysis")

    df = get_store().latest()
    
    if df.empty:
        st.warning("No data available. Please upload data in the 'Upload / Data' section.")
        return
        
    total_draws = len(df)

    # --- Controls
    k = st.slider("Combo size (pair = 2, triplet = 3, etc.)", 2, 5, 2)
    include_pb = st.checkbox("Include Powerball in combos", value=False)
    min_freq = st.slider(
        "Minimum frequency", 1, 
        max(1, int(total_draws / 10)), 
        1, help="Filter out rare combinations"
    )
    
    # --- Compute combinations
    with st.spinner("Counting combinations..."):
        counter = _count_combos(df, k, include_pb)
        
        # Filter by minimum frequency
        if min_freq > 1:
            counter = Counter({c: f for c, f in counter.items() if f >= min_freq})
        
        # Convert to DataFrame
        combos_df = _to_df(counter)
    
    # --- Results
    st.subheader(f"Top combinations (size {k})")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.dataframe(combos_df, use_container_width=True)
    with col2:
        st.metric("Total combinations", len(combos_df))
        st.metric("Max frequency", combos_df["Frequency"].max() if not combos_df.empty else 0)
        st.metric("Avg frequency", f"{combos_df['Frequency'].mean():.2f}" if not combos_df.empty else 0)
        
    # --- Visualization
    if not combos_df.empty:
        st.subheader("Frequency distribution")
        top_n = min(25, len(combos_df))
        
        fig = px.bar(
            combos_df.head(top_n),
            x="Combo",
            y="Frequency",
            title=f"Top {top_n} most frequent combinations",
            labels={"Combo": "Combination", "Frequency": "Times drawn"}
        )
        st.plotly_chart(fig, use_container_width=True)
