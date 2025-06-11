#!/usr/bin/env python3
"""
Check and recreate the zip file properly
"""
import zipfile
import os
from pathlib import Path

def check_zip_contents(zip_path):
    """Check what's in the zip file"""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            file_list = zipf.namelist()
            print(f"Zip file contains {len(file_list)} files:")
            for file in file_list[:20]:  # Show first 20 files
                print(f"  {file}")
            if len(file_list) > 20:
                print(f"  ... and {len(file_list) - 20} more files")
            return file_list
    except Exception as e:
        print(f"Error reading zip file: {e}")
        return []

def create_proper_zip():
    """Create a proper zip file with all application files"""
    zip_filename = "powerball_insights_complete.zip"
    
    # Essential files to include
    include_files = [
        'app.py',
        'README.md',
        '.gitignore',
        'pyproject.toml',
        'uv.lock'
    ]
    
    # Directories to include
    include_dirs = ['core']
    
    # Files to exclude
    exclude_patterns = {
        '.git', '__pycache__', '.pyc', '.db', '.joblib', 
        '.log', '.streamlit', 'tmp', '.DS_Store'
    }
    
    print(f"Creating new zip file: {zip_filename}")
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add essential files
        for file in include_files:
            if os.path.exists(file):
                zipf.write(file)
                print(f"Added: {file}")
        
        # Add core directory
        core_dir = Path('core')
        if core_dir.exists():
            for file_path in core_dir.rglob('*'):
                if file_path.is_file() and not any(pattern in str(file_path) for pattern in exclude_patterns):
                    arcname = str(file_path)
                    zipf.write(file_path, arcname)
                    print(f"Added: {arcname}")
        
        # Add documentation files
        for md_file in Path('.').glob('*.md'):
            if md_file.name not in ['README.md']:  # README already added
                zipf.write(md_file)
                print(f"Added: {md_file}")
        
        # Add other Python files
        for py_file in Path('.').glob('*.py'):
            if py_file.name not in include_files:
                zipf.write(py_file)
                print(f"Added: {py_file}")
    
    # Check size
    if os.path.exists(zip_filename):
        size_mb = os.path.getsize(zip_filename) / (1024 * 1024)
        print(f"\nNew zip file created: {zip_filename}")
        print(f"Size: {size_mb:.2f} MB")
        return zip_filename
    else:
        print("Failed to create zip file")
        return None

if __name__ == "__main__":
    # Check existing zip
    existing_zip = "powerball_insights_app_20250611_174850.zip"
    print("Checking existing zip file:")
    contents = check_zip_contents(existing_zip)
    
    print("\n" + "="*50)
    
    # Create new zip
    new_zip = create_proper_zip()
    
    if new_zip:
        print(f"\nChecking new zip file:")
        check_zip_contents(new_zip)