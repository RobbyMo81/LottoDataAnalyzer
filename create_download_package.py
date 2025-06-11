#!/usr/bin/env python3
"""
Create downloadable package for Powerball Insights application
"""
import os
import zipfile
from pathlib import Path
from datetime import datetime

def create_application_package():
    """Create a complete application package for download"""
    
    # Create timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"powerball_insights_app_{timestamp}.zip"
    
    # Files and directories to exclude
    exclude_patterns = {
        '.git', '__pycache__', '.pyc', '.pyo', '.pyd',
        '.db', '.joblib', '.log', '.tmp', '.cache',
        '.streamlit', 'tmp', 'temp', '.DS_Store',
        'node_modules', '.env', '.venv', 'venv',
        '.tar.gz', '.zip', 'backups'
    }
    
    # Get current directory
    root_dir = Path('.')
    
    print(f"Creating application package: {zip_filename}")
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Walk through all files in the directory
        for file_path in root_dir.rglob('*'):
            # Skip if file/directory matches exclude patterns
            if any(pattern in str(file_path) for pattern in exclude_patterns):
                continue
                
            # Skip if it's a directory
            if file_path.is_dir():
                continue
                
            # Skip hidden files
            if any(part.startswith('.') for part in file_path.parts):
                continue
                
            # Add file to zip
            arcname = str(file_path.relative_to(root_dir))
            try:
                zipf.write(file_path, arcname)
                print(f"Added: {arcname}")
            except Exception as e:
                print(f"Skipped {arcname}: {e}")
    
    # Get zip file size
    zip_size = os.path.getsize(zip_filename)
    size_mb = zip_size / (1024 * 1024)
    
    print(f"\nPackage created successfully!")
    print(f"Filename: {zip_filename}")
    print(f"Size: {size_mb:.2f} MB")
    print(f"Location: {os.path.abspath(zip_filename)}")
    
    return zip_filename

if __name__ == "__main__":
    create_application_package()