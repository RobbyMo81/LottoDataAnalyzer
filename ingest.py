# core/ingest.py
import streamlit as st
import pandas as pd
from pandas.errors import EmptyDataError, ParserError
from pathlib import Path
from core.storage import get_store    # will hold the "current" dataframe

# Priority order for datasets (newest/most complete first)
DATASET_PATHS = [
    Path("data/powerball_complete_dataset.csv"),      # Authentic 2025 dataset  
    Path("data/powerball_history_corrected.csv"),     # Date-corrected dataset
    Path("data/powerball_history.csv")                # Original dataset
]
DATA_PATH = DATASET_PATHS[0]  # Default to the complete dataset
MAX_MB = 8

# ----------------------------------------------------------------------
def _load_default_csv() -> pd.DataFrame | None:
    """Return the best available dataset, prioritizing complete authentic data."""
    for dataset_path in DATASET_PATHS:
        if dataset_path.exists():
            try:
                df = pd.read_csv(dataset_path)
                if df.empty or len(df.columns) == 0:
                    continue
                
                # Verify required columns exist
                required_cols = ["draw_date", "n1", "n2", "n3", "n4", "n5", "powerball"]
                if all(col in df.columns for col in required_cols):
                    return df
                    
            except (EmptyDataError, ParserError):
                continue
    
    # No valid dataset found
    return None

# ----------------------------------------------------------------------
def render_page() -> None:
    st.header("📤 Upload / Data")
    st.caption("Drag in a Powerball CSV, use the default file, or manually add new draw results.")

    # Create tabs for different data input methods
    tab1, tab2 = st.tabs(["File Upload", "Manual Entry"])
    
    with tab1:
        st.subheader("📁 File Upload")
        # -------- 1. File uploader (always visible) -----------------------
        file = st.file_uploader("Upload Powerball CSV",
                                type="csv",
                                help=f"Limit {MAX_MB} MB per file • CSV only",
                                accept_multiple_files=False)

        # -------- 2. Decide which dataframe to show ----------------------
        df: pd.DataFrame | None = None

        if file:                              # user just uploaded something
            try:
                df = pd.read_csv(file)
                
                # Standardize date format to YYYY-MM-DD
                if 'draw_date' in df.columns:
                    try:
                        # Convert to datetime then to standardized string format
                        df['draw_date'] = pd.to_datetime(df['draw_date'], errors='coerce')
                        df['draw_date'] = df['draw_date'].dt.strftime('%Y-%m-%d')
                        
                        # Remove any rows where date conversion failed
                        original_count = len(df)
                        df = df.dropna(subset=['draw_date'])
                        converted_count = len(df)
                        
                        if converted_count < original_count:
                            st.warning(f"Removed {original_count - converted_count} rows with invalid dates")
                        
                        st.success(f"Loaded {converted_count:,} rows from upload (dates standardized to YYYY-MM-DD format)")
                    except Exception as date_error:
                        st.warning(f"Date format standardization failed: {date_error}")
                        st.success(f"Loaded {len(df):,} rows from upload")
                else:
                    st.success(f"Loaded {len(df):,} rows from upload")
                    
            except Exception as e:
                st.error(f"❌ Cannot read that CSV: {e}")
                return

        else:                                 # fall back to default on disk
            df = _load_default_csv()
            if df is not None:
                st.success(f"Loaded {len(df):,} rows from {DATA_PATH}")
            else:
                st.info("No valid default dataset yet — upload one to begin.")
                return
        
        # -------- 3. Preview + persist (File Upload Tab) -----------------------------------
        st.dataframe(df.head(), use_container_width=True)

        # save upload as new default
        if file and st.button("💾  Save as default dataset"):
            DATA_PATH.parent.mkdir(exist_ok=True, parents=True)
            df.to_csv(DATA_PATH, index=False)
            st.toast("Saved!", icon="✅")

        # offer to delete the bad file
        if (not file) and st.button("🗑️ Delete invalid default file"):
            DATA_PATH.unlink(missing_ok=True)
            st.toast("Deleted. Upload a new dataset to continue.", icon="🗑️")
            st.rerun()           # refresh the page

        # store for other pages
        get_store().set_latest(df)

    with tab2:
        st.subheader("✏️ Manual Entry")
        st.info("Add individual Powerball draw results with authentic data from official sources.")
        
        # Load current dataset to add to
        current_df = _load_default_csv()
        if current_df is None:
            st.warning("No existing dataset found. Please upload a base dataset first.")
            return
            
        # Manual entry form
        with st.form("manual_entry_form"):
            st.markdown("**Enter New Draw Result**")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Date input
                draw_date = st.date_input("Draw Date", help="Enter the official draw date")
                
                # White balls input
                st.markdown("**White Balls (5 numbers from 1-69):**")
                white_cols = st.columns(5)
                white_balls = []
                for i, col in enumerate(white_cols):
                    with col:
                        num = st.number_input(f"Ball {i+1}", min_value=1, max_value=69, 
                                            value=1, key=f"white_{i}")
                        white_balls.append(num)
                
                # Powerball input
                powerball = st.number_input("Powerball (1-26)", min_value=1, max_value=26, value=1)
            
            with col2:
                st.markdown("**Validation:**")
                # Check for duplicates in white balls
                white_set = set(white_balls)
                if len(white_set) != 5:
                    st.error("❌ White balls must be unique")
                    valid_entry = False
                else:
                    st.success("✅ White balls are unique")
                    valid_entry = True
                
                # Check if date already exists
                if current_df is not None and 'draw_date' in current_df.columns:
                    date_str = str(draw_date)
                    if date_str in current_df['draw_date'].astype(str).values:
                        st.warning("⚠️ Date already exists")
                        valid_entry = False
                    else:
                        st.success("✅ New date")
            
            submitted = st.form_submit_button("Add Draw Result", disabled=not valid_entry)
            
            if submitted and valid_entry:
                # Create new row
                new_row = {
                    'draw_date': str(draw_date),
                    'n1': white_balls[0],
                    'n2': white_balls[1], 
                    'n3': white_balls[2],
                    'n4': white_balls[3],
                    'n5': white_balls[4],
                    'powerball': powerball
                }
                
                # Add to dataframe
                new_df = pd.concat([current_df, pd.DataFrame([new_row])], ignore_index=True)
                
                # Sort by date (newest first)
                new_df['draw_date'] = pd.to_datetime(new_df['draw_date'])
                new_df = new_df.sort_values('draw_date', ascending=False)
                new_df['draw_date'] = new_df['draw_date'].dt.strftime('%Y-%m-%d')
                
                # Save to file
                DATA_PATH.parent.mkdir(exist_ok=True, parents=True)
                new_df.to_csv(DATA_PATH, index=False)
                
                # Update storage
                get_store().set_latest(new_df)
                
                st.success(f"✅ Added draw result for {draw_date}")
                st.success(f"**Numbers:** {', '.join(map(str, sorted(white_balls)))} | **Powerball:** {powerball}")
                st.toast("Draw result added successfully!", icon="🎯")
                st.rerun()
        
        # Show recent entries
        if current_df is not None and not current_df.empty:
            st.markdown("---")
            st.markdown("**Recent Draw Results:**")
            # Show last 10 entries
            recent_df = current_df.head(10)
            st.dataframe(recent_df, use_container_width=True)
