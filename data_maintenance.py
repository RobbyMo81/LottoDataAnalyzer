"""
Data Maintenance Agent
---------------------
Provides tools for cleaning, maintaining, and optimizing Powerball datasets.
Features:
- Duplicate detection and removal
- Date format standardization
- Missing value handling
- Data validation (range checks)
- Dataset optimization
- Automatic backups
"""

import pandas as pd
import numpy as np
import os
import datetime
import logging
from typing import Optional, Dict, List, Tuple
import joblib
import shutil
from pathlib import Path

# Constants
DATA_DIR = Path("data")
BACKUP_DIR = DATA_DIR / "backups"
DEFAULT_CSV = DATA_DIR / "powerball_history.csv"
MAX_WHITE_BALL = 69
MIN_WHITE_BALL = 1
MAX_POWERBALL = 26
MIN_POWERBALL = 1

# Ensure backup directory exists
BACKUP_DIR.mkdir(exist_ok=True, parents=True)

class DataMaintenanceAgent:
    """
    Data Maintenance Agent for Powerball datasets.
    Provides tools for data cleaning, validation, and optimization.
    """
    
    def __init__(self, df: Optional[pd.DataFrame] = None, file_path: Optional[str] = None):
        """
        Initialize the data maintenance agent.
        
        Args:
            df: DataFrame to maintain, or None to load from file_path
            file_path: Path to CSV file to load, or None to use default
        """
        self.df = df
        self.file_path = file_path or str(DEFAULT_CSV)
        self.issues = {
            'duplicates': 0,
            'missing_values': 0,
            'out_of_range': 0,
            'date_format': 0,
            'total': 0
        }
        
        if self.df is None and os.path.exists(self.file_path):
            self.load_data()
    
    def load_data(self) -> pd.DataFrame:
        """
        Load data from the file path.
        
        Returns:
            DataFrame with loaded data
        """
        try:
            self.df = pd.read_csv(self.file_path)
            return self.df
        except Exception as e:
            logging.error(f"Error loading data from {self.file_path}: {e}")
            self.df = pd.DataFrame()
            return self.df
    
    def backup_data(self) -> str:
        """
        Create a backup of the current dataset.
        
        Returns:
            Path to the backup file
        """
        if self.df is None or self.df.empty:
            return "No data to backup"
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = BACKUP_DIR / f"powerball_backup_{timestamp}.csv"
        
        try:
            self.df.to_csv(backup_file, index=False)
            return str(backup_file)
        except Exception as e:
            logging.error(f"Error creating backup: {e}")
            return f"Backup failed: {e}"
    
    def detect_duplicates(self) -> pd.DataFrame:
        """
        Detect duplicate records in the dataset.
        
        Returns:
            DataFrame containing duplicate records
        """
        if self.df is None or self.df.empty:
            return pd.DataFrame()
        
        # Check for exact duplicates
        duplicates = self.df[self.df.duplicated(keep='first')]
        self.issues['duplicates'] = len(duplicates)
        self.issues['total'] += len(duplicates)
        
        # Ensure we return a DataFrame, not a Series
        if isinstance(duplicates, pd.Series):
            duplicates = duplicates.to_frame()
            
        return duplicates
    
    def remove_duplicates(self) -> pd.DataFrame:
        """
        Remove duplicate records from the dataset.
        
        Returns:
            DataFrame with duplicates removed
        """
        if self.df is None or self.df.empty:
            return pd.DataFrame()
        
        # Count duplicates first
        dup_count = sum(self.df.duplicated())
        self.issues['duplicates'] = dup_count
        self.issues['total'] += dup_count
        
        # Remove duplicates
        self.df = self.df.drop_duplicates(keep='first')
        
        return self.df
    
    def standardize_date_format(self) -> pd.DataFrame:
        """
        Standardize date formats in the dataset.
        
        Returns:
            DataFrame with standardized date formats
        """
        if self.df is None or self.df.empty:
            return pd.DataFrame()
        
        # Identify date column
        date_col = None
        for col in self.df.columns:
            col_lower = col.lower()
            if 'date' in col_lower or 'day' in col_lower or 'time' in col_lower:
                date_col = col
                break
        
        if date_col:
            try:
                # Convert to datetime and standardize format
                # Keep track of how many records were modified
                orig_vals = self.df[date_col].copy()
                self.df[date_col] = pd.to_datetime(self.df[date_col])
                modified = (orig_vals != self.df[date_col].astype(str)).sum()
                self.issues['date_format'] = modified
                self.issues['total'] += modified
            except Exception as e:
                logging.error(f"Error standardizing date format: {e}")
        
        return self.df
    
    def handle_missing_values(self) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Returns:
            DataFrame with missing values handled
        """
        if self.df is None or self.df.empty:
            return pd.DataFrame()
        
        # Count missing values
        missing_count = self.df.isna().sum().sum()
        self.issues['missing_values'] = missing_count
        self.issues['total'] += missing_count
        
        # Handle missing values based on column type
        for col in self.df.columns:
            col_lower = col.lower()
            
            # Numeric columns (like ball numbers)
            if any(term in col_lower for term in ['n1', 'n2', 'n3', 'n4', 'n5', 'powerball', 'ball']):
                # For numeric lottery data, we can't truly impute missing values
                # The best approach is to drop those rows as they represent incomplete data
                # But we'll flag them so the user can decide
                missing_rows = self.df[self.df[col].isna()]
                self.df = self.df.dropna(subset=[col])
            
            # Date columns
            elif any(term in col_lower for term in ['date', 'day', 'time']):
                # Can't reliably impute missing dates
                self.df = self.df.dropna(subset=[col])
            
            # Other columns
            else:
                # For non-critical columns, fill with mode (most common value)
                if self.df[col].isna().any():
                    if self.df[col].dtype == 'object':
                        fill_value = self.df[col].mode()[0] if not self.df[col].mode().empty else "Unknown"
                        self.df[col] = self.df[col].fillna(fill_value)
                    else:
                        fill_value = self.df[col].mode()[0] if not self.df[col].mode().empty else 0
                        self.df[col] = self.df[col].fillna(fill_value)
        
        return self.df
    
    def validate_range(self) -> Dict[str, List[int]]:
        """
        Validate that Powerball numbers are within allowed ranges.
        
        Returns:
            Dictionary with lists of row indices that are out of range
        """
        if self.df is None or self.df.empty:
            return {'white_balls': [], 'powerball': []}
        
        # Check white ball columns (n1-n5)
        white_cols = [col for col in self.df.columns if col in ['n1', 'n2', 'n3', 'n4', 'n5']]
        
        out_of_range = {
            'white_balls': [],
            'powerball': []
        }
        
        # Check white balls (1-69)
        for col in white_cols:
            invalid_idx = self.df[
                (self.df[col] < MIN_WHITE_BALL) | 
                (self.df[col] > MAX_WHITE_BALL)
            ].index.tolist()
            out_of_range['white_balls'].extend(invalid_idx)
        
        # Check Powerball (1-26)
        if 'powerball' in self.df.columns:
            invalid_idx = self.df[
                (self.df['powerball'] < MIN_POWERBALL) | 
                (self.df['powerball'] > MAX_POWERBALL)
            ].index.tolist()
            out_of_range['powerball'].extend(invalid_idx)
        
        # Remove duplicates
        out_of_range['white_balls'] = list(set(out_of_range['white_balls']))
        out_of_range['powerball'] = list(set(out_of_range['powerball']))
        
        # Update issue count
        self.issues['out_of_range'] = len(out_of_range['white_balls']) + len(out_of_range['powerball'])
        self.issues['total'] += self.issues['out_of_range']
        
        return out_of_range
    
    def fix_out_of_range(self) -> pd.DataFrame:
        """
        Fix out-of-range values in the dataset.
        Removes rows with invalid values.
        
        Returns:
            DataFrame with out-of-range values fixed
        """
        if self.df is None or self.df.empty:
            return pd.DataFrame()
        
        # Get out-of-range indices
        out_of_range = self.validate_range()
        
        # Combine all invalid indices
        all_invalid = list(set(out_of_range['white_balls'] + out_of_range['powerball']))
        
        # Remove invalid rows
        if all_invalid:
            self.df = self.df.drop(all_invalid)
        
        return self.df
    
    def optimize_dataset(self) -> pd.DataFrame:
        """
        Optimize the dataset for memory efficiency.
        
        Returns:
            Optimized DataFrame
        """
        if self.df is None or self.df.empty:
            return pd.DataFrame()
        
        # Optimize numeric columns
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                # Check if column contains only integers
                if (self.df[col] % 1 == 0).all():
                    # Check range to determine best integer type
                    col_min = self.df[col].min()
                    col_max = self.df[col].max()
                    
                    if col_min >= 0 and col_max <= 255:
                        self.df[col] = self.df[col].astype('uint8')
                    elif col_min >= 0 and col_max <= 65535:
                        self.df[col] = self.df[col].astype('uint16')
                    elif col_min >= -128 and col_max <= 127:
                        self.df[col] = self.df[col].astype('int8')
                    elif col_min >= -32768 and col_max <= 32767:
                        self.df[col] = self.df[col].astype('int16')
        
        return self.df
    
    def sort_by_date(self) -> pd.DataFrame:
        """
        Sort the dataset by draw date.
        
        Returns:
            DataFrame sorted by date
        """
        if self.df is None or self.df.empty:
            return pd.DataFrame()
        
        # Find date column
        date_col = None
        for col in self.df.columns:
            col_lower = col.lower()
            if 'date' in col_lower or 'day' in col_lower or 'time' in col_lower:
                date_col = col
                break
        
        if date_col:
            # Ensure date column is in datetime format
            self.df[date_col] = pd.to_datetime(self.df[date_col])
            
            # Sort by date (descending so newest is first)
            self.df = self.df.sort_values(date_col, ascending=False)
        
        return self.df
    
    def clean_and_validate(self) -> pd.DataFrame:
        """
        Run all cleaning and validation steps.
        
        Returns:
            Cleaned and validated DataFrame
        """
        # Reset issue counts
        self.issues = {
            'duplicates': 0,
            'missing_values': 0,
            'out_of_range': 0,
            'date_format': 0,
            'total': 0
        }
        
        # Run all steps
        self.backup_data()
        self.remove_duplicates()
        self.standardize_date_format()
        self.handle_missing_values()
        self.fix_out_of_range()
        self.optimize_dataset()
        self.sort_by_date()
        
        return self.df
    
    def save_data(self, file_path: Optional[str] = None) -> str:
        """
        Save the cleaned dataset.
        
        Args:
            file_path: Path to save the cleaned dataset. If None, uses the original path.
            
        Returns:
            Path to the saved file
        """
        if self.df is None or self.df.empty:
            return "No data to save"
        
        save_path = file_path or self.file_path
        
        try:
            self.df.to_csv(save_path, index=False)
            return save_path
        except Exception as e:
            logging.error(f"Error saving cleaned data: {e}")
            return f"Save failed: {e}"
    
    def delete_old_backups(self, keep_count: int = 5) -> Dict[str, int]:
        """
        Delete old backup files, keeping only the most recent ones.
        
        Args:
            keep_count: Number of most recent backups to keep
            
        Returns:
            Dictionary with deletion statistics
        """
        deleted_count = 0
        deleted_size = 0
        
        try:
            if BACKUP_DIR.exists():
                # Get all backup files sorted by creation time (newest first)
                backups = list(BACKUP_DIR.glob("*.csv"))
                backups.sort(key=lambda x: x.stat().st_ctime, reverse=True)
                
                # Delete old backups beyond keep_count
                if len(backups) > keep_count:
                    old_backups = backups[keep_count:]
                    
                    for backup in old_backups:
                        try:
                            file_size = backup.stat().st_size
                            backup.unlink()  # Delete the file
                            deleted_count += 1
                            deleted_size += file_size
                        except Exception as e:
                            logging.warning(f"Could not delete backup {backup.name}: {e}")
                            
        except Exception as e:
            logging.error(f"Error during backup cleanup: {e}")
        
        return {
            "deleted_count": deleted_count,
            "deleted_size_mb": deleted_size / (1024 * 1024),
            "remaining_count": max(0, len(list(BACKUP_DIR.glob("*.csv"))) if BACKUP_DIR.exists() else 0)
        }

    def get_dataset_stats(self) -> Dict:
        """
        Get statistics about the dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        if self.df is None or self.df.empty:
            return {
                "rows": 0,
                "columns": 0,
                "memory_usage": "0 MB",
                "date_range": None,
                "issues": self.issues
            }
        
        # Find date column
        date_col = None
        for col in self.df.columns:
            col_lower = col.lower()
            if 'date' in col_lower or 'day' in col_lower or 'time' in col_lower:
                date_col = col
                break
        
        date_range = None
        if date_col:
            try:
                min_date = pd.to_datetime(self.df[date_col]).min()
                max_date = pd.to_datetime(self.df[date_col]).max()
                date_range = f"{min_date.date()} to {max_date.date()}"
            except:
                date_range = "Unable to determine date range"
        
        return {
            "rows": len(self.df),
            "columns": len(self.df.columns),
            "memory_usage": f"{self.df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB",
            "date_range": date_range,
            "issues": self.issues
        }


def render_page():
    """
    Render the Streamlit page for data maintenance.
    """
    import streamlit as st
    
    st.title("🧹 Data Maintenance Agent")
    st.write("""
    This tool helps you maintain and clean your Powerball dataset to ensure accurate analysis.
    It detects and fixes common data issues like duplicates, missing values, and invalid entries.
    """)
    
    # Initialize agent
    if 'data_agent' not in st.session_state:
        st.session_state.data_agent = DataMaintenanceAgent()
    
    agent = st.session_state.data_agent
    
    # Load Data Section
    st.subheader("Data Source")
    data_source = st.radio(
        "Select data source",
        ["Use current dataset", "Load from CSV", "Upload new file"]
    )
    
    if data_source == "Load from CSV":
        csv_path = st.text_input("Enter CSV file path", value=str(DEFAULT_CSV))
        if st.button("Load Data"):
            agent.file_path = csv_path
            agent.load_data()
            st.success(f"Loaded data with {len(agent.df)} rows")
    
    elif data_source == "Upload new file":
        uploaded_file = st.file_uploader("Upload Powerball CSV", type="csv")
        if uploaded_file is not None:
            # Save uploaded file to temporary location
            temp_path = DATA_DIR / "temp_upload.csv"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            agent.file_path = str(temp_path)
            agent.load_data()
            st.success(f"Uploaded and loaded data with {len(agent.df)} rows")
    
    # Only show the rest if data is loaded
    if agent.df is not None and not agent.df.empty:
        # Dataset Statistics
        st.subheader("Dataset Overview")
        stats = agent.get_dataset_stats()
        
        # Create two columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Rows", stats["rows"])
            st.metric("Columns", stats["columns"])
        
        with col2:
            st.metric("Memory Usage", stats["memory_usage"])
            if stats["date_range"]:
                st.metric("Date Range", stats["date_range"])
        
        # Preview data
        with st.expander("Preview Data"):
            st.dataframe(agent.df.head(10))
        
        # Maintenance Actions
        st.subheader("Maintenance Actions")
        
        maintenance_tabs = st.tabs([
            "Clean All", "Duplicates", "Missing Values", 
            "Range Check", "Optimization", "Backups"
        ])
        
        # Clean All tab
        with maintenance_tabs[0]:
            st.write("Run all cleaning and validation steps at once.")
            
            if st.button("Run Complete Maintenance"):
                with st.spinner("Cleaning and validating data..."):
                    agent.clean_and_validate()
                
                # Display results
                issues = agent.issues
                st.success(f"Maintenance complete: {issues['total']} issues addressed")
                
                # Show issue breakdown
                if issues['total'] > 0:
                    st.write("Issues addressed:")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Duplicates", issues['duplicates'])
                        st.metric("Missing Values", issues['missing_values'])
                    with col2:
                        st.metric("Out of Range", issues['out_of_range'])
                        st.metric("Date Format", issues['date_format'])
                
                # Option to save
                if st.button("💾 Save Clean Dataset"):
                    save_path = agent.save_data()
                    st.success(f"Saved cleaned data to {save_path}")
        
        # Duplicates tab
        with maintenance_tabs[1]:
            st.write("Identify and remove duplicate records.")
            
            if st.button("Check for Duplicates"):
                duplicates = agent.detect_duplicates()
                if len(duplicates) > 0:
                    st.warning(f"Found {len(duplicates)} duplicate records")
                    st.dataframe(duplicates)
                    
                    if st.button("Remove Duplicates"):
                        agent.remove_duplicates()
                        st.success(f"Removed {len(duplicates)} duplicate records")
                else:
                    st.success("No duplicates found!")
        
        # Missing Values tab
        with maintenance_tabs[2]:
            st.write("Identify and handle missing values.")
            
            # Count missing values by column
            missing_by_col = agent.df.isna().sum()
            missing_by_col = missing_by_col[missing_by_col > 0]
            
            if len(missing_by_col) > 0:
                st.warning(f"Found {missing_by_col.sum()} missing values across {len(missing_by_col)} columns")
                
                # Show breakdown by column
                missing_df = pd.DataFrame({
                    'Column': missing_by_col.index,
                    'Missing Values': missing_by_col.values,
                    'Percentage': (missing_by_col.values / len(agent.df) * 100).round(2)
                })
                st.dataframe(missing_df)
                
                if st.button("Handle Missing Values"):
                    agent.handle_missing_values()
                    st.success("Missing values handled")
            else:
                st.success("No missing values found!")
        
        # Range Check tab
        with maintenance_tabs[3]:
            st.write("Check if all Powerball numbers are within valid ranges.")
            
            if st.button("Validate Number Ranges"):
                out_of_range = agent.validate_range()
                
                total_invalid = len(out_of_range['white_balls']) + len(out_of_range['powerball'])
                
                if total_invalid > 0:
                    st.warning(f"Found {total_invalid} numbers out of allowed ranges")
                    
                    if len(out_of_range['white_balls']) > 0:
                        st.write(f"White balls out of range (1-69): {len(out_of_range['white_balls'])} rows")
                        if len(out_of_range['white_balls']) < 20:  # Only show if not too many
                            st.dataframe(agent.df.loc[out_of_range['white_balls']])
                    
                    if len(out_of_range['powerball']) > 0:
                        st.write(f"Powerball out of range (1-26): {len(out_of_range['powerball'])} rows")
                        if len(out_of_range['powerball']) < 20:  # Only show if not too many
                            st.dataframe(agent.df.loc[out_of_range['powerball']])
                    
                    if st.button("Fix Out-of-Range Values"):
                        before_count = len(agent.df)
                        agent.fix_out_of_range()
                        after_count = len(agent.df)
                        st.success(f"Removed {before_count - after_count} rows with invalid values")
                else:
                    st.success("All numbers are within valid ranges!")
        
        # Optimization tab
        with maintenance_tabs[4]:
            st.write("Optimize the dataset for memory efficiency.")
            
            if st.button("Optimize Dataset"):
                before_size = agent.df.memory_usage(deep=True).sum() / (1024 * 1024)
                
                agent.optimize_dataset()
                
                after_size = agent.df.memory_usage(deep=True).sum() / (1024 * 1024)
                reduction = before_size - after_size
                
                st.success(f"Dataset optimized: {before_size:.2f} MB → {after_size:.2f} MB ({reduction:.2f} MB saved)")
        
        # Backups tab
        with maintenance_tabs[5]:
            st.write("Create and manage backups of your dataset.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Create Backup"):
                    backup_path = agent.backup_data()
                    st.success(f"Backup created: {backup_path}")
            
            with col2:
                if st.button("View Backups"):
                    if BACKUP_DIR.exists():
                        backups = list(BACKUP_DIR.glob("*.csv"))
                        if backups:
                            backup_info = []
                            for b in backups:
                                size_mb = b.stat().st_size / (1024 * 1024)
                                created = datetime.datetime.fromtimestamp(b.stat().st_ctime)
                                backup_info.append({
                                    "Filename": b.name,
                                    "Size (MB)": f"{size_mb:.2f}",
                                    "Created": created
                                })
                            
                            # Convert to DataFrame and sort by creation time (newest first)
                            backup_df = pd.DataFrame(backup_info)
                            backup_df = backup_df.sort_values("Created", ascending=False)
                            
                            st.dataframe(backup_df)
                            
                            # Add backup cleanup section
                            st.markdown("---")
                            st.subheader("Backup Cleanup")
                            
                            if len(backups) > 3:
                                st.info(f"You have {len(backups)} backup files. Consider cleaning up old ones to save space.")
                                
                                keep_count = st.number_input(
                                    "Number of recent backups to keep",
                                    min_value=1,
                                    max_value=20,
                                    value=5,
                                    help="Older backups will be permanently deleted"
                                )
                                
                                col_cleanup1, col_cleanup2 = st.columns(2)
                                
                                with col_cleanup1:
                                    if st.button("🗑️ Delete Old Backups", type="secondary"):
                                        if len(backups) > keep_count:
                                            with st.spinner("Deleting old backups..."):
                                                result = agent.delete_old_backups(keep_count)
                                            
                                            if result["deleted_count"] > 0:
                                                st.success(
                                                    f"Successfully deleted {result['deleted_count']} old backup(s), "
                                                    f"freed {result['deleted_size_mb']:.2f} MB. "
                                                    f"{result['remaining_count']} backups remaining."
                                                )
                                                # Force page refresh to update backup list
                                                st.rerun()
                                            else:
                                                st.info("No old backups to delete.")
                                        else:
                                            st.info(f"Only {len(backups)} backups exist, nothing to delete.")
                                
                                with col_cleanup2:
                                    if st.button("⚠️ Delete ALL Backups", type="secondary"):
                                        st.warning("This will delete ALL backup files permanently!")
                                        st.session_state.confirm_delete_all = True
                                    
                                    # Show confirmation button if user clicked delete all
                                    if getattr(st.session_state, 'confirm_delete_all', False):
                                        if st.button("✅ Confirm: Delete ALL backups", type="primary"):
                                            with st.spinner("Deleting all backups..."):
                                                result = agent.delete_old_backups(0)  # Keep 0 = delete all
                                            st.success(f"Deleted all {result['deleted_count']} backup files, freed {result['deleted_size_mb']:.2f} MB")
                                            st.session_state.confirm_delete_all = False
                                            st.rerun()
                            else:
                                st.info("Backup count is manageable. Cleanup not needed.")
                                
                        else:
                            st.info("No backups found.")
                    else:
                        st.info("Backup directory doesn't exist yet.")
        
        # Save Final Dataset
        st.subheader("Save Final Dataset")
        save_format = st.radio("Save format", ["CSV", "Parquet"], horizontal=True)
        
        save_path = st.text_input(
            "Save path", 
            value=str(DATA_DIR / f"powerball_clean.{save_format.lower()}")
        )
        
        if st.button("💾 Save Final Dataset"):
            try:
                if save_format == "CSV":
                    agent.df.to_csv(save_path, index=False)
                else:  # Parquet
                    agent.df.to_parquet(save_path, index=False)
                st.success(f"Dataset saved to {save_path}")
            except Exception as e:
                st.error(f"Error saving dataset: {e}")
    
    else:
        st.info("Please load a dataset to begin maintenance.")


if __name__ == "__main__":
    # For testing
    agent = DataMaintenanceAgent()
    print(agent.get_dataset_stats())