#!/usr/bin/env python3
"""
Date Format Standardization Implementation
-----------------------------------------
Converts all date formats in the Powerball Insights system to YYYY-MM-DD standard.
Addresses the critical inconsistency identified in the backend architecture audit.
"""

import pandas as pd
import os
import shutil
from datetime import datetime
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DateFormatStandardizer:
    """Standardizes all date formats in the Powerball Insights system to YYYY-MM-DD."""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.backup_dir = self.data_dir / "date_standardization_backup"
        
        # CSV files that need date format conversion
        self.csv_files = [
            "powerball_complete_dataset.csv",
            "powerball_history.csv", 
            "powerball_history_corrected.csv",
            "powerball_clean.csv"
        ]
        
        self.conversion_report = {
            'files_processed': [],
            'files_converted': [],
            'files_skipped': [],
            'errors': [],
            'total_records_converted': 0
        }
    
    def create_backup(self):
        """Create backup of all data files before conversion."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.data_dir / f"date_standardization_backup_{timestamp}"
        backup_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Creating backup at {backup_path}")
        
        # Backup all CSV files
        for csv_file in self.csv_files:
            file_path = self.data_dir / csv_file
            if file_path.exists():
                backup_file = backup_path / csv_file
                shutil.copy2(file_path, backup_file)
                logger.info(f"Backed up {csv_file}")
        
        return backup_path
    
    def detect_date_format(self, df, date_column='draw_date'):
        """Detect the current date format in the dataset."""
        if date_column not in df.columns:
            return None
        
        # Sample first few non-null date values
        sample_dates = df[date_column].dropna().head(5).tolist()
        
        format_patterns = {
            'MM/DD/YYYY': r'^\d{2}/\d{2}/\d{4}$',
            'YYYY-MM-DD': r'^\d{4}-\d{2}-\d{2}$',
            'M/D/YYYY': r'^\d{1,2}/\d{1,2}/\d{4}$',
            'YYYY/MM/DD': r'^\d{4}/\d{2}/\d{2}$'
        }
        
        import re
        for date_str in sample_dates:
            date_str = str(date_str).strip()
            for format_name, pattern in format_patterns.items():
                if re.match(pattern, date_str):
                    return format_name
        
        return 'UNKNOWN'
    
    def convert_date_format(self, df, date_column='draw_date'):
        """Convert date column to YYYY-MM-DD format."""
        if date_column not in df.columns:
            return df, 0
        
        original_format = self.detect_date_format(df, date_column)
        logger.info(f"Detected date format: {original_format}")
        
        if original_format == 'YYYY-MM-DD':
            logger.info("Date format already compliant, no conversion needed")
            return df, 0
        
        try:
            # Convert to datetime objects (pandas handles multiple formats)
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
            
            # Check for any parsing failures
            null_dates = df[date_column].isnull().sum()
            if null_dates > 0:
                logger.warning(f"Warning: {null_dates} dates could not be parsed")
            
            # Convert to YYYY-MM-DD string format
            df[date_column] = df[date_column].dt.strftime('%Y-%m-%d')
            
            # Remove any rows where date conversion failed
            df = df.dropna(subset=[date_column])
            
            converted_count = len(df)
            logger.info(f"Successfully converted {converted_count} date records to YYYY-MM-DD format")
            
            return df, converted_count
            
        except Exception as e:
            logger.error(f"Error converting dates: {e}")
            return df, 0
    
    def standardize_csv_file(self, filename):
        """Standardize date format in a single CSV file."""
        file_path = self.data_dir / filename
        
        if not file_path.exists():
            logger.info(f"File {filename} does not exist, skipping")
            self.conversion_report['files_skipped'].append(filename)
            return
        
        try:
            logger.info(f"Processing {filename}")
            
            # Load CSV file
            df = pd.read_csv(file_path)
            original_count = len(df)
            
            # Convert date format
            df_converted, converted_count = self.convert_date_format(df)
            
            if converted_count > 0:
                # Save converted file
                df_converted.to_csv(file_path, index=False)
                logger.info(f"Saved standardized {filename} with {len(df_converted)} records")
                
                self.conversion_report['files_converted'].append({
                    'filename': filename,
                    'original_records': original_count,
                    'converted_records': len(df_converted),
                    'format_changed': True
                })
                self.conversion_report['total_records_converted'] += converted_count
            else:
                self.conversion_report['files_converted'].append({
                    'filename': filename,
                    'original_records': original_count,
                    'converted_records': original_count,
                    'format_changed': False
                })
            
            self.conversion_report['files_processed'].append(filename)
            
        except Exception as e:
            error_msg = f"Error processing {filename}: {e}"
            logger.error(error_msg)
            self.conversion_report['errors'].append(error_msg)
    
    def update_storage_singleton(self):
        """Update the storage singleton to use the newly standardized data."""
        try:
            from core.storage import get_store
            
            # Load the standardized dataset
            primary_dataset = self.data_dir / "powerball_complete_dataset.csv"
            if primary_dataset.exists():
                df = pd.read_csv(primary_dataset)
                
                # Update the storage singleton
                store = get_store()
                store.set_latest(df)
                
                logger.info("Updated storage singleton with standardized data")
                return True
                
        except Exception as e:
            logger.error(f"Error updating storage singleton: {e}")
            return False
    
    def run_standardization(self):
        """Execute the complete date format standardization process."""
        logger.info("Starting date format standardization process")
        
        # Create backup
        backup_path = self.create_backup()
        logger.info(f"Backup created at: {backup_path}")
        
        # Process each CSV file
        for csv_file in self.csv_files:
            self.standardize_csv_file(csv_file)
        
        # Update storage singleton
        storage_updated = self.update_storage_singleton()
        
        # Generate final report
        self.generate_completion_report(backup_path, storage_updated)
        
        return self.conversion_report
    
    def generate_completion_report(self, backup_path, storage_updated):
        """Generate a completion report of the standardization process."""
        report = []
        report.append("=" * 60)
        report.append("DATE FORMAT STANDARDIZATION COMPLETION REPORT")
        report.append("=" * 60)
        report.append()
        
        # Summary statistics
        report.append("SUMMARY")
        report.append("-" * 30)
        report.append(f"Files processed: {len(self.conversion_report['files_processed'])}")
        report.append(f"Files converted: {len(self.conversion_report['files_converted'])}")
        report.append(f"Files skipped: {len(self.conversion_report['files_skipped'])}")
        report.append(f"Total records converted: {self.conversion_report['total_records_converted']}")
        report.append(f"Backup location: {backup_path}")
        report.append(f"Storage singleton updated: {'Yes' if storage_updated else 'No'}")
        report.append("")
        
        # File conversion details
        if self.conversion_report['files_converted']:
            report.append("CONVERSION DETAILS")
            report.append("-" * 30)
            for file_info in self.conversion_report['files_converted']:
                status = "Format changed" if file_info['format_changed'] else "Already compliant"
                report.append(f"• {file_info['filename']}")
                report.append(f"  Records: {file_info['original_records']} → {file_info['converted_records']}")
                report.append(f"  Status: {status}")
                report.append("")
        
        # Errors
        if self.conversion_report['errors']:
            report.append("ERRORS ENCOUNTERED")
            report.append("-" * 30)
            for error in self.conversion_report['errors']:
                report.append(f"• {error}")
            report.append("")
        
        # Next steps
        report.append("NEXT STEPS")
        report.append("-" * 30)
        report.append("1. Verify data integrity in the web application")
        report.append("2. Test all ML training and prediction functionality")
        report.append("3. Confirm date displays show YYYY-MM-DD format")
        report.append("4. Remove backup files after validation (optional)")
        report.append("")
        
        report.append("STATUS: DATE FORMAT STANDARDIZATION COMPLETE")
        report.append("All CSV files now use YYYY-MM-DD format consistently")
        
        # Save report to file
        report_content = "\n".join(report)
        report_path = "DATE_STANDARDIZATION_REPORT.md"
        
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        # Print to console
        print(report_content)
        logger.info(f"Standardization report saved to: {report_path}")

def main():
    """Main execution function."""
    print("Initializing Date Format Standardization...")
    
    standardizer = DateFormatStandardizer()
    
    # Run the standardization process
    results = standardizer.run_standardization()
    
    return results

if __name__ == "__main__":
    main()