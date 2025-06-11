"""
Refactored Prediction Storage System
-----------------------------------
Ensures only one prediction is stored per unique Powerball draw date (Monday, Wednesday, Saturday).
Implements comprehensive validation and migration capabilities.
"""

import pandas as pd
import numpy as np
import datetime
import logging
import os
from typing import List, Dict, Optional, Tuple
import joblib
from collections import defaultdict

class PredictionStorageManager:
    """
    Manages prediction storage with one-prediction-per-draw-date enforcement.
    """
    
    def __init__(self, history_file_path: str = "data/prediction_history.joblib"):
        """
        Initialize the prediction storage manager.
        
        Args:
            history_file_path: Path to the prediction history file
        """
        self.history_file_path = history_file_path
        self.prediction_history = {
            'predictions': [],
            'accuracy': [],
            'feedback': []
        }
        self._load_history()
    
    def _load_history(self):
        """Load existing prediction history."""
        try:
            if os.path.exists(self.history_file_path):
                self.prediction_history = joblib.load(self.history_file_path)
                logging.info(f"Loaded prediction history with {len(self.prediction_history.get('predictions', []))} entries")
        except Exception as e:
            logging.warning(f"Could not load prediction history: {e}")
    
    def _save_history(self):
        """Save prediction history to file."""
        try:
            os.makedirs(os.path.dirname(self.history_file_path), exist_ok=True)
            joblib.dump(self.prediction_history, self.history_file_path)
        except Exception as e:
            logging.error(f"Could not save prediction history: {e}")
    
    def _is_valid_draw_date(self, date_str: str) -> bool:
        """
        Check if a date is a valid Powerball draw date (Monday, Wednesday, Saturday).
        
        Args:
            date_str: ISO format date string
            
        Returns:
            True if the date is a valid draw date
        """
        try:
            date_obj = datetime.datetime.fromisoformat(date_str).date()
            # Monday=0, Wednesday=2, Saturday=5
            return date_obj.weekday() in [0, 2, 5]
        except:
            return False
    
    def _get_next_draw_date(self, from_date: Optional[datetime.date] = None) -> str:
        """
        Calculate the next valid Powerball draw date.
        
        Args:
            from_date: Calculate next draw from this date. If None, uses today.
            
        Returns:
            ISO format date string of the next draw
        """
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
        return next_draw_date.isoformat()
    
    def store_prediction(self, prediction: Dict, target_date: Optional[str] = None) -> bool:
        """
        Store a prediction, enforcing one-prediction-per-draw-date rule.
        
        Args:
            prediction: Dictionary containing prediction data
            target_date: Optional target draw date. If None, calculates next draw date.
            
        Returns:
            True if prediction was stored successfully
        """
        try:
            # Determine target date
            if target_date is None:
                target_date = self._get_next_draw_date()
            
            # Ensure target_date is a string
            if not isinstance(target_date, str):
                target_date = str(target_date)
            
            # Check if target date is in the past and adjust to next valid draw date
            target_date_obj = datetime.datetime.fromisoformat(target_date).date()
            current_date = datetime.datetime.now().date()
            
            if target_date_obj < current_date:
                logging.warning(f"Past date detected: {target_date}, adjusting to next valid draw date")
                target_date = self._get_next_draw_date(current_date)
            
            # Validate that it's a valid draw date
            if not self._is_valid_draw_date(target_date):
                logging.warning(f"Invalid draw date: {target_date} (not Monday, Wednesday, or Saturday)")
                # Adjust to next valid draw date
                date_obj = datetime.datetime.fromisoformat(target_date).date()
                target_date = self._get_next_draw_date(date_obj)
            
            # Add metadata to prediction
            enhanced_prediction = prediction.copy()
            enhanced_prediction.update({
                'prediction_for_date': target_date,
                'timestamp': datetime.datetime.now().isoformat(),
                'storage_version': '2.0'  # Mark as new storage format
            })
            
            # Check if prediction already exists for this date
            existing_predictions = self.prediction_history.get('predictions', [])
            existing_index = None
            
            for i, existing_pred in enumerate(existing_predictions):
                if existing_pred.get('prediction_for_date') == target_date:
                    existing_index = i
                    break
            
            # Store or update prediction
            if existing_index is not None:
                # Replace existing prediction
                self.prediction_history['predictions'][existing_index] = enhanced_prediction
                logging.info(f"Updated existing prediction for {target_date}")
            else:
                # Add new prediction
                self.prediction_history['predictions'].append(enhanced_prediction)
                logging.info(f"Added new prediction for {target_date}")
            
            # Save to file
            self._save_history()
            return True
            
        except Exception as e:
            logging.error(f"Error storing prediction: {e}")
            return False
    
    def get_predictions_by_date(self, date_str: str) -> Optional[Dict]:
        """
        Get prediction for a specific date.
        
        Args:
            date_str: ISO format date string
            
        Returns:
            Prediction dictionary or None if not found
        """
        predictions = self.prediction_history.get('predictions', [])
        for pred in predictions:
            if pred.get('prediction_for_date') == date_str:
                return pred
        return None
    
    def analyze_duplicate_predictions(self) -> Dict:
        """
        Analyze the current prediction history for duplicates.
        
        Returns:
            Dictionary with analysis results
        """
        predictions = self.prediction_history.get('predictions', [])
        date_groups = defaultdict(list)
        
        # Group predictions by date
        for i, pred in enumerate(predictions):
            date = pred.get('prediction_for_date', 'unknown')
            date_groups[date].append((i, pred))
        
        # Identify duplicates
        duplicates = {}
        total_duplicates = 0
        
        for date, pred_list in date_groups.items():
            if len(pred_list) > 1:
                duplicates[date] = {
                    'count': len(pred_list),
                    'predictions': pred_list,
                    'is_valid_draw_date': self._is_valid_draw_date(date)
                }
                total_duplicates += len(pred_list) - 1  # All but one are duplicates
        
        return {
            'total_predictions': len(predictions),
            'unique_dates': len(date_groups),
            'duplicate_dates': len(duplicates),
            'total_duplicate_entries': total_duplicates,
            'duplicates': duplicates,
            'efficiency_gain': f"{total_duplicates}/{len(predictions)} ({(total_duplicates/len(predictions)*100):.1f}%)" if predictions else "0/0 (0%)"
        }
    
    def migrate_legacy_data(self, strategy: str = 'keep_latest') -> Dict:
        """
        Migrate existing prediction data to enforce one-prediction-per-date.
        
        Args:
            strategy: Migration strategy ('keep_latest', 'keep_earliest', 'keep_highest_prob')
            
        Returns:
            Dictionary with migration results
        """
        analysis = self.analyze_duplicate_predictions()
        
        if analysis['duplicate_dates'] == 0:
            return {
                'status': 'no_migration_needed',
                'message': 'No duplicate predictions found',
                'analysis': analysis
            }
        
        # Create backup before migration
        backup_path = f"{self.history_file_path}.backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            joblib.dump(self.prediction_history, backup_path)
            logging.info(f"Created backup at {backup_path}")
        except Exception as e:
            logging.error(f"Failed to create backup: {e}")
            return {'status': 'backup_failed', 'error': str(e)}
        
        # Perform migration
        migrated_predictions = []
        migration_log = []
        
        for date, duplicate_info in analysis['duplicates'].items():
            pred_list = duplicate_info['predictions']
            
            # Select which prediction to keep based on strategy
            if strategy == 'keep_latest':
                # Keep the one with the latest timestamp
                selected_pred = max(pred_list, key=lambda x: x[1].get('timestamp', ''))
            elif strategy == 'keep_earliest':
                # Keep the one with the earliest timestamp
                selected_pred = min(pred_list, key=lambda x: x[1].get('timestamp', ''))
            elif strategy == 'keep_highest_prob':
                # Keep the one with the highest probability
                selected_pred = max(pred_list, key=lambda x: x[1].get('probability', 0))
            else:
                # Default to latest
                selected_pred = max(pred_list, key=lambda x: x[1].get('timestamp', ''))
            
            # Add to migrated predictions
            migrated_predictions.append(selected_pred[1])
            
            # Log the migration
            migration_log.append({
                'date': date,
                'original_count': len(pred_list),
                'selected_index': selected_pred[0],
                'selected_timestamp': selected_pred[1].get('timestamp'),
                'removed_count': len(pred_list) - 1
            })
        
        # Add non-duplicate predictions
        predictions = self.prediction_history.get('predictions', [])
        for i, pred in enumerate(predictions):
            date = pred.get('prediction_for_date', 'unknown')
            if date not in analysis['duplicates']:
                migrated_predictions.append(pred)
        
        # Update prediction history
        self.prediction_history['predictions'] = migrated_predictions
        
        # Save migrated data
        self._save_history()
        
        return {
            'status': 'migration_completed',
            'strategy': strategy,
            'backup_path': backup_path,
            'original_count': analysis['total_predictions'],
            'migrated_count': len(migrated_predictions),
            'removed_count': analysis['total_duplicate_entries'],
            'migration_log': migration_log,
            'final_analysis': self.analyze_duplicate_predictions()
        }
    
    def get_prediction_history_dataframe(self) -> pd.DataFrame:
        """
        Get prediction history as a pandas DataFrame for display.
        
        Returns:
            DataFrame with prediction history
        """
        predictions = self.prediction_history.get('predictions', [])
        if not predictions:
            return pd.DataFrame()
        
        # Convert to DataFrame format for display
        history_data = []
        for i, pred in enumerate(predictions):
            # Get date for display
            date = pred.get('prediction_for_date', 'Unknown')
            
            # Format white numbers
            white_nums = pred.get('white_numbers', [])
            white_str = ", ".join(str(n) for n in white_nums) if white_nums else "N/A"
            
            # Get other data
            pb = pred.get('powerball', 0)
            timestamp = pred.get('timestamp', 'Unknown')
            
            # Add to history data
            history_data.append({
                'Date': date,
                'White Numbers': white_str,
                'Powerball': pb,
                'Timestamp': timestamp,
                'Valid Draw Date': 'Yes' if self._is_valid_draw_date(date) else 'No'
            })
        
        return pd.DataFrame(history_data)
    
    def validate_system_integrity(self) -> Dict:
        """
        Validate the integrity of the prediction storage system.
        
        Returns:
            Dictionary with validation results
        """
        analysis = self.analyze_duplicate_predictions()
        predictions = self.prediction_history.get('predictions', [])
        
        # Check for invalid draw dates
        invalid_dates = []
        for pred in predictions:
            date = pred.get('prediction_for_date')
            if date and not self._is_valid_draw_date(date):
                invalid_dates.append(date)
        
        # Check for missing required fields
        incomplete_predictions = []
        for i, pred in enumerate(predictions):
            required_fields = ['white_numbers', 'powerball', 'prediction_for_date', 'timestamp']
            missing_fields = [field for field in required_fields if field not in pred]
            if missing_fields:
                incomplete_predictions.append({
                    'index': i,
                    'missing_fields': missing_fields
                })
        
        return {
            'total_predictions': len(predictions),
            'unique_dates': analysis['unique_dates'],
            'has_duplicates': analysis['duplicate_dates'] > 0,
            'duplicate_count': analysis['duplicate_dates'],
            'invalid_draw_dates': len(invalid_dates),
            'invalid_dates': invalid_dates,
            'incomplete_predictions': len(incomplete_predictions),
            'incomplete_details': incomplete_predictions,
            'system_healthy': (
                analysis['duplicate_dates'] == 0 and 
                len(invalid_dates) == 0 and 
                len(incomplete_predictions) == 0
            )
        }