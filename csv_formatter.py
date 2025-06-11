# core/csv_formatter.py
import streamlit as st
import pandas as pd
import io
import re
import ast
from datetime import datetime

REQUIRED = ["draw_date", "n1", "n2", "n3", "n4", "n5", "powerball"]

def _parse_number_array(value):
    """
    Parse various formats of number arrays:
    - "[1, 11, 18, 30, 41]"
    - "1,11,18,30,41"
    - "1 11 18 30 41"
    """
    if pd.isna(value) or value == "":
        return []
    
    # Convert to string if not already
    value_str = str(value).strip()
    
    # Try to parse as Python list format
    if value_str.startswith('[') and value_str.endswith(']'):
        try:
            return ast.literal_eval(value_str)
        except:
            # Remove brackets and split
            value_str = value_str[1:-1]
    
    # Split by various delimiters
    if ',' in value_str:
        numbers = value_str.split(',')
    elif ' ' in value_str:
        numbers = value_str.split()
    else:
        # Single number or space-separated without spaces
        numbers = re.findall(r'\d+', value_str)
    
    # Convert to integers and filter valid numbers
    try:
        return [int(n.strip()) for n in numbers if n.strip().isdigit()]
    except:
        return []

def _detect_data_format(df):
    """
    Detect the format of the uploaded data and suggest parsing strategy.
    """
    formats = {
        'standard': False,
        'grouped_numbers': False,
        'mixed_format': False
    }
    
    # Check for standard format (separate n1, n2, n3, n4, n5, powerball columns)
    standard_cols = ['n1', 'n2', 'n3', 'n4', 'n5', 'powerball']
    if all(col.lower() in [c.lower() for c in df.columns] for col in standard_cols):
        formats['standard'] = True
    
    # Check for grouped numbers format (drawing numbers in array format)
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['drawing', 'numbers', 'balls', 'winning']):
            # Sample some values to see if they contain arrays
            sample_values = df[col].dropna().head(3)
            for val in sample_values:
                if '[' in str(val) or ',' in str(val):
                    formats['grouped_numbers'] = True
                    break
    
    # Check for mixed format
    if not formats['standard'] and not formats['grouped_numbers']:
        formats['mixed_format'] = True
    
    return formats

def _auto_map(columns: list[str]) -> dict[str, str]:
    """
    Try to guess which raw column goes to which required name.
    Returns a dict {required_name: raw_name or ""}.
    """
    mapping = {req: "" for req in REQUIRED}
    lowered = {c.lower(): c for c in columns}

    guesses = {
        "draw_date": ["date", "draw", "drawdate", "draw_date", "weekday"],
        "n1": ["n1", "ball1", "first", "num1"],
        "n2": ["n2", "ball2", "second", "num2"],
        "n3": ["n3", "ball3", "third", "num3"],
        "n4": ["n4", "ball4", "fourth", "num4"],
        "n5": ["n5", "ball5", "fifth", "num5"],
        "powerball": ["pb", "power", "powerball", "bonus"],
    }

    for req, candidates in guesses.items():
        for cand in candidates:
            if cand in lowered:
                mapping[req] = lowered[cand]
                break
    return mapping

def _parse_grouped_format(df):
    """
    Parse data where drawing numbers are grouped together in arrays.
    """
    parsed_rows = []
    
    for idx, row in df.iterrows():
        parsed_row = {}
        
        # Find the column with drawing numbers
        drawing_numbers = []
        powerball = None
        date_value = None
        
        for col, value in row.items():
            col_lower = col.lower()
            
            # Look for date column
            if any(keyword in col_lower for keyword in ['date', 'weekday']):
                date_value = value
            
            # Look for drawing numbers
            elif any(keyword in col_lower for keyword in ['drawing', 'numbers', 'balls', 'winning']):
                numbers = _parse_number_array(value)
                if len(numbers) >= 5:
                    drawing_numbers = numbers[:5]  # Take first 5 as white balls
            
            # Look for powerball
            elif 'powerball' in col_lower or 'pb' in col_lower:
                if pd.notna(value):
                    try:
                        powerball = int(value)
                    except:
                        pass
        
        # If we found drawing numbers but no separate powerball, assume last number is powerball
        if drawing_numbers and powerball is None:
            # Look for powerball in the same row
            for col, value in row.items():
                if 'powerball' in col.lower() and pd.notna(value):
                    try:
                        powerball = int(value)
                        break
                    except:
                        pass
        
        # If still no powerball found, it might be in the numbers array
        if powerball is None and len(drawing_numbers) >= 6:
            powerball = drawing_numbers[5]
            drawing_numbers = drawing_numbers[:5]
        
        # Build the parsed row
        if len(drawing_numbers) == 5 and powerball is not None:
            parsed_row = {
                'draw_date': date_value,
                'n1': drawing_numbers[0],
                'n2': drawing_numbers[1], 
                'n3': drawing_numbers[2],
                'n4': drawing_numbers[3],
                'n5': drawing_numbers[4],
                'powerball': powerball
            }
            parsed_rows.append(parsed_row)
    
    return pd.DataFrame(parsed_rows)


def render_page():
    st.header("🛠️ CSV Formatter")

    raw_file = st.file_uploader(
        "Upload *any* Powerball CSV (we'll re-format it)", type=["csv"]
    )
    if not raw_file:
        st.info("Waiting for file…")
        return

    raw_df = pd.read_csv(raw_file)
    st.write("Preview of uploaded data:")
    st.dataframe(raw_df.head())

    # Detect data format
    formats = _detect_data_format(raw_df)
    
    if formats['grouped_numbers']:
        st.info("🔍 Detected grouped number format (e.g., '[1, 11, 18, 30, 41]')")
        
        if st.button("Auto-parse grouped format"):
            with st.spinner("Parsing grouped number format..."):
                try:
                    formatted_df = _parse_grouped_format(raw_df)
                    
                    if not formatted_df.empty:
                        st.success(f"Successfully parsed {len(formatted_df)} rows!")
                        st.write("Formatted data preview:")
                        st.dataframe(formatted_df.head())
                        
                        # Validate the parsed data
                        missing_cols = [col for col in REQUIRED if col not in formatted_df.columns]
                        if missing_cols:
                            st.warning(f"Missing columns: {missing_cols}")
                        else:
                            # Offer download
                            csv_buffer = io.StringIO()
                            formatted_df.to_csv(csv_buffer, index=False)
                            st.download_button(
                                "Download formatted CSV",
                                csv_buffer.getvalue(),
                                file_name="powerball_formatted.csv",
                                mime="text/csv"
                            )
                    else:
                        st.error("Could not parse the data. Please check the format.")
                        
                except Exception as e:
                    st.error(f"Error parsing data: {str(e)}")
        
        st.markdown("---")
        st.subheader("Manual column mapping (if auto-parse didn't work)")
    
    elif formats['standard']:
        st.success("✅ Standard format detected")
    else:
        st.info("Mixed format detected - use manual mapping below")

    st.subheader("Column mapping")
    cols = list(raw_df.columns)
    default_map = _auto_map(cols)

    mapping = {}
    for req in REQUIRED:
        mapping[req] = st.selectbox(
            f"{req} ⟵", [""] + cols, index=(cols.index(default_map[req]) + 1 if default_map[req] else 0)
        )

    if st.button("Convert ➜"):
        # Validation
        missing = [k for k, v in mapping.items() if v == ""]
        if missing:
            st.error(f"Please map all columns. Missing: {missing}")
            return

        # Build formatted frame
        fmt = raw_df[[mapping[c] for c in REQUIRED]].copy()
        fmt.columns = REQUIRED

        # Date parsing & cleaning
        fmt["draw_date"] = pd.to_datetime(fmt["draw_date"], errors="coerce").dt.date
        if fmt["draw_date"].isna().any():
            st.warning("Some draw_date rows could not be parsed and became NaT.")

        # Sort & type-cast
        fmt = fmt.dropna(subset=["draw_date"]).sort_values("draw_date")
        for c in REQUIRED[1:]:
            fmt[c] = pd.to_numeric(fmt[c], errors="coerce").astype("Int64")

        st.success("✅ Reformatted!")

        st.dataframe(fmt.head())

        # Download link
        csv = fmt.to_csv(index=False)
        st.download_button(
            label="Download formatted CSV",
            data=csv,
            file_name=f"powerball_formatted_{datetime.now().date()}.csv",
            mime="text/csv",
        )
