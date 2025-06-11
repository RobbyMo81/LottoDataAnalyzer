"""
Powerball Insights
-----------------
A comprehensive data analysis and visualization tool for Powerball lottery data.
"""
import streamlit as st
import importlib
from pathlib import Path

# Core pages to include in navigation
PAGES = {
    "Upload / Data": "ingest",
    "CSV Formatter": "csv_formatter",
    "Data Maintenance": "data_maintenance",
    "Prediction Storage Migration": "prediction_storage_migration_ui",
    "Number Frequency": "frequency",
    "Day of Week Analysis": "dow_analysis",
    "Time Trends": "time_trends", 
    "Inter-Draw Gaps": "inter_draw",
    "Combinatorial Analysis": "combos",
    "Sum Analysis": "sums",
    "ML Experimental": "ml_experimental",
    "AutoML Tuning": "automl_simple",
    "System Architecture": "system_relationship_visualizer",
}

# Include AI analysis module
PAGES["Ask the Numbers (AI)"] = "llm_query"

# App configuration
st.set_page_config(
    page_title="Powerball Insights",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.sidebar.title("🎯 Powerball Insights")
st.sidebar.caption("Statistical analysis and visualization tool")

# Navigation
page_name = st.sidebar.radio("Navigate", list(PAGES.keys()))
page_module = PAGES[page_name]

# Data existence check & warning
data_path = Path("data/powerball_history.csv")
if page_name != "Upload / Data" and page_name != "CSV Formatter" and not data_path.exists():
    st.warning("⚠️ No data loaded yet. Please go to **Upload / Data** to add lottery data first.")

# Documentation expander
with st.sidebar.expander("About & Documentation"):
    st.markdown("""
    **Powerball Insights** provides statistical analysis of lottery data. 
    
    Features:
    - Upload & format lottery data
    - Visualize number frequencies
    - Analyze day-of-week patterns
    - Explore time-based trends
    - Calculate combinatorial statistics
    - Run experimental ML analyses
    
    **Note**: No prediction can improve your odds of winning. This app is for educational and entertainment purposes.
    """)

# Display version at the bottom of sidebar
st.sidebar.divider()
from core import __version__
st.sidebar.caption(f"v{__version__}")

# Import and render the selected page
try:
    page = importlib.import_module(f"core.{page_module}")
    page.render_page()
except Exception as e:
    st.error(f"Error loading page: {e}")
