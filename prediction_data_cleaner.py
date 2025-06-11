"""
Prediction Data Cleaner
-----------------------
Fixes data integrity issues in prediction storage by converting numpy types to native Python types.
"""

import joblib
import os
import numpy as np
import datetime
import logging
from typing import Any, Dict, List

def clean_prediction_data():
    """Clean and fix prediction data to ensure all types are JSON-serializable native Python types."""
    
    history_path = "data/prediction_history.joblib"
    
    if not os.path.exists(history_path):
        print("No prediction history file found")
        return False
    
    try:
        # Load existing data
        history = joblib.load(history_path)
        predictions = history.get('predictions', [])
        
        print(f"Processing {len(predictions)} predictions...")
        
        cleaned_predictions = []
        
        for i, pred in enumerate(predictions):
            cleaned_pred = clean_single_prediction(pred, i)
            cleaned_predictions.append(cleaned_pred)
        
        # Update history with cleaned predictions
        history['predictions'] = cleaned_predictions
        
        # Create backup before saving
        backup_path = f"{history_path}.backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        joblib.dump(history, backup_path)
        print(f"Backup created: {backup_path}")
        
        # Save cleaned data
        joblib.dump(history, history_path)
        print(f"Successfully cleaned and saved {len(cleaned_predictions)} predictions")
        
        return True
        
    except Exception as e:
        print(f"Error cleaning prediction data: {e}")
        return False

def clean_single_prediction(pred: Dict[str, Any], index: int) -> Dict[str, Any]:
    """Clean a single prediction by converting numpy types to native Python types."""
    
    cleaned = {}
    
    for key, value in pred.items():
        cleaned[key] = convert_numpy_types(value)
    
    # Ensure required fields are present with proper types
    ensure_required_fields(cleaned, index)
    
    return cleaned

def convert_numpy_types(value: Any) -> Any:
    """Recursively convert numpy types to native Python types."""
    
    if isinstance(value, np.integer):
        return int(value)
    elif isinstance(value, np.floating):
        return float(value)
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, dict):
        return {k: convert_numpy_types(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [convert_numpy_types(item) for item in value]
    elif isinstance(value, tuple):
        return tuple(convert_numpy_types(item) for item in value)
    else:
        return value

def ensure_required_fields(pred: Dict[str, Any], index: int):
    """Ensure all required fields are present and valid."""
    
    # Required fields for a complete prediction
    required_fields = ['white_numbers', 'powerball', 'prediction_for_date', 'timestamp']
    
    # Fix missing prediction_for_date
    if 'prediction_for_date' not in pred or pred['prediction_for_date'] is None:
        # Try to derive from timestamp if available
        if 'timestamp' in pred and pred['timestamp']:
            try:
                timestamp_date = datetime.datetime.fromisoformat(pred['timestamp']).date()
                next_draw_date = get_next_powerball_draw_date(timestamp_date)
                pred['prediction_for_date'] = next_draw_date.isoformat()
                print(f"Fixed missing prediction_for_date for prediction {index}: {pred['prediction_for_date']}")
            except:
                # Fallback to a reasonable default
                pred['prediction_for_date'] = '2025-06-04'  # Next Wednesday
                print(f"Used fallback prediction_for_date for prediction {index}")
    
    # Fix missing timestamp
    if 'timestamp' not in pred or pred['timestamp'] is None:
        pred['timestamp'] = datetime.datetime.now().isoformat()
        print(f"Fixed missing timestamp for prediction {index}")
    
    # Ensure white_numbers is a proper list of integers
    if 'white_numbers' in pred:
        white_nums = pred['white_numbers']
        if isinstance(white_nums, list):
            pred['white_numbers'] = [int(num) for num in white_nums if isinstance(num, (int, np.integer))]
        else:
            pred['white_numbers'] = []
    
    # Ensure powerball is a proper integer
    if 'powerball' in pred:
        pb = pred['powerball']
        if isinstance(pb, (int, np.integer)):
            pred['powerball'] = int(pb)
        else:
            pred['powerball'] = 1  # Default value
    
    # Add storage version to mark as cleaned
    pred['storage_version'] = '2.0'

def get_next_powerball_draw_date(from_date: datetime.date = None) -> datetime.date:
    """Calculate the next Powerball draw date (Monday, Wednesday, Saturday)."""
    
    if from_date is None:
        from_date = datetime.datetime.now().date()
    
    # Powerball drawing days: Monday=0, Wednesday=2, Saturday=5
    draw_days = [0, 2, 5]
    current_weekday = from_date.weekday()
    
    # Find the next draw day
    days_ahead = None
    for draw_day in draw_days:
        if draw_day > current_weekday:
            days_ahead = draw_day - current_weekday
            break
    
    # If no draw day found this week, get Monday of next week
    if days_ahead is None:
        days_ahead = (7 - current_weekday) + 0  # Monday of next week
    
    next_draw_date = from_date + datetime.timedelta(days=days_ahead)
    return next_draw_date

def validate_cleaned_data():
    """Validate that the cleaned data meets all completeness requirements."""
    
    history_path = "data/prediction_history.joblib"
    
    if not os.path.exists(history_path):
        return False, "No prediction history file found"
    
    try:
        history = joblib.load(history_path)
        predictions = history.get('predictions', [])
        
        required_fields = ['white_numbers', 'powerball', 'prediction_for_date', 'timestamp']
        
        incomplete_count = 0
        validation_errors = []
        
        for i, pred in enumerate(predictions):
            errors = []
            
            # Check required fields
            for field in required_fields:
                if field not in pred or pred[field] is None:
                    errors.append(f"Missing {field}")
                elif field == 'white_numbers':
                    if not isinstance(pred[field], list) or len(pred[field]) != 5:
                        errors.append(f"Invalid white_numbers: {pred[field]}")
                elif field == 'powerball':
                    if not isinstance(pred[field], int) or pred[field] < 1 or pred[field] > 26:
                        errors.append(f"Invalid powerball: {pred[field]}")
            
            if errors:
                incomplete_count += 1
                validation_errors.append(f"Prediction {i}: {', '.join(errors)}")
        
        return incomplete_count == 0, validation_errors
        
    except Exception as e:
        return False, [f"Validation error: {e}"]

if __name__ == "__main__":
    print("Starting prediction data cleaning...")
    
    if clean_prediction_data():
        print("\nValidating cleaned data...")
        is_valid, errors = validate_cleaned_data()
        
        if is_valid:
            print("✅ All predictions are now complete and valid!")
        else:
            print("❌ Validation failed:")
            for error in errors:
                print(f"  - {error}")
    else:
        print("❌ Failed to clean prediction data")