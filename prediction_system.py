"""
Enhanced Prediction System for Powerball Lottery
------------------------------------------------
This module integrates multiple analysis tools to create a more sophisticated 
prediction model that generates 5 distinct predictions with probability scores.

The system leverages:
- Frequency Analysis
- Day of Week Analysis
- Time Trends Analysis
- Inter-Draw Gap Analysis 
- Combinatorial Analysis
- Sum Analysis

Each tool contributes to the final prediction scores, and the system learns
from past predictions to continuously improve its accuracy.
"""

import pandas as pd
import numpy as np
import datetime
import logging
import pickle
import os
from typing import List, Dict, Tuple, Optional
from collections import Counter
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
import random

def get_next_powerball_draw_date(from_date: datetime.date = None) -> datetime.date:
    """
    Static function to calculate the next Powerball draw date.
    Powerball draws occur on Monday, Wednesday, and Saturday.
    
    Args:
        from_date: Calculate next draw from this date. If None, uses today.
        
    Returns:
        Date of the next Powerball draw
    """
    if from_date is None:
        from_date = datetime.datetime.now().date()
    
    # Powerball drawing days: Monday=0, Wednesday=2, Saturday=5
    draw_days = [0, 2, 5]
    
    # Find the next draw day
    for days_ahead in range(8):  # Check up to 7 days ahead
        check_date = from_date + datetime.timedelta(days=days_ahead)
        if check_date.weekday() in draw_days:
            return check_date
    
    # Fallback (should never reach here)
    return from_date + datetime.timedelta(days=1)

def validate_powerball_dates(df: pd.DataFrame, date_column: str = 'draw_date') -> Dict:
    """
    Validate that dates in the dataset follow the Powerball schedule.
    
    Args:
        df: DataFrame containing draw data
        date_column: Name of the date column
        
    Returns:
        Dictionary with validation results
    """
    if date_column not in df.columns:
        return {'valid': False, 'error': f'Column {date_column} not found'}
    
    try:
        dates = pd.to_datetime(df[date_column])
        draw_days = [0, 2, 5]  # Monday, Wednesday, Saturday
        
        valid_dates = dates.dt.weekday.isin(draw_days)
        invalid_count = (~valid_dates).sum()
        
        return {
            'valid': invalid_count == 0,
            'total_draws': len(df),
            'invalid_dates': invalid_count,
            'valid_percentage': (valid_dates.sum() / len(df)) * 100,
            'invalid_weekdays': dates[~valid_dates].dt.day_name().tolist()
        }
    except Exception as e:
        return {'valid': False, 'error': str(e)}

# Import analysis tools
from .frequency import calc_frequency
from .inter_draw import _days_since_last, _all_gaps
from .time_trends import _aggregate, _calc_trending
from .combos import _count_combos
from .dow_analysis import _compute_stats
from .sums import _prep as _prep_sums

# Constants
WHITE_MIN, WHITE_MAX = 1, 69
PB_MIN, PB_MAX = 1, 26
WHITE_BALLS_COUNT = 5
MODEL_FILE_PATH = "data/prediction_models.joblib"
HISTORY_FILE_PATH = "data/prediction_history.joblib"

class PredictionSystem:
    """Integrated prediction system that leverages multiple analysis tools."""
    
    def __init__(self, df_history: Optional[pd.DataFrame] = None):
        """
        Initialize the prediction system.
        
        Args:
            df_history: DataFrame containing historical draw data
        """
        # Initialize default values
        self.df_history = pd.DataFrame() if df_history is None else df_history.copy()
        self.feature_weights = {
            'frequency': 0.20,  # Basic frequency analysis
            'recency': 0.15,    # Inter-draw gap analysis
            'trends': 0.15,     # Time trend analysis
            'combo': 0.15,      # Combination analysis
            'sum': 0.15,        # Sum distribution analysis
            'dow': 0.10,        # Day of week analysis
            'ml': 0.10          # Machine learning model
        }
        
        # Machine learning models 
        # Create a dictionary for storing model objects (will be Pipeline instances)
        self.models = {}
        # Pre-fill with None values to avoid key errors
        self.models['white_balls'] = None
        self.models['powerball'] = None
        
        # Cross-validation performance metrics
        self.model_cv_performance = {
            'white_balls': None,
            'powerball': None
        }
        
        # Feature importance storage
        self.feature_importance = {
            'white_balls': None,
            'powerball': None
        }
        
        # History of predictions for learning
        self.prediction_history = {
            'predictions': [],  # List of past predictions
            'accuracy': [],     # Accuracy metrics for each prediction
            'feedback': []      # Feedback on which tools performed best
        }
        
        # Cache for expensive computation results
        self._prediction_cache = {}
        
        # Load prediction history if available
        self._load_history()
        
        # Set the dataframe if one was provided (which also initializes the cache)
        if df_history is not None:
            self.set_dataframe(df_history)
        
    def _calculate_next_draw_date(self) -> str:
        """
        Calculate the next Powerball draw date.
        Uses the static function for consistent date calculation.
        
        Returns:
            ISO format date string of the next draw
        """
        try:
            next_draw = get_next_powerball_draw_date()
            return next_draw.isoformat()
        except Exception as e:
            logging.warning(f"Error calculating next draw date: {e}")
            return datetime.datetime.now().date().isoformat()
    
    def clear_cache(self):
        """Clear the prediction cache. Should be called when dataframe changes."""
        self._prediction_cache = {}
        
    def set_dataframe(self, df_history: pd.DataFrame):
        """
        Update the dataframe and clear the cache.
        
        Args:
            df_history: DataFrame containing historical draw data
        """
        self.df_history = df_history
        self.clear_cache()  # Clear cache when dataframe changes
        
    def _load_history(self):
        """Load prediction history and models if available."""
        try:
            if os.path.exists(HISTORY_FILE_PATH):
                self.prediction_history = joblib.load(HISTORY_FILE_PATH)
                
            if os.path.exists(MODEL_FILE_PATH):
                self.models = joblib.load(MODEL_FILE_PATH)
        except Exception as e:
            logging.warning(f"Could not load prediction history or models: {e}")
    
    def _save_history(self):
        """Save prediction history and models for future use."""
        try:
            os.makedirs(os.path.dirname(HISTORY_FILE_PATH), exist_ok=True)
            joblib.dump(self.prediction_history, HISTORY_FILE_PATH)
            
            if self.models['white_balls'] is not None:
                joblib.dump(self.models, MODEL_FILE_PATH)
        except Exception as e:
            logging.warning(f"Could not save prediction history or models: {e}")
    
    def update_with_new_draw(self, new_draw: Dict):
        """
        Update the system with the results of a new draw to improve future predictions.
        
        Args:
            new_draw: Dictionary containing the new draw results
        """
        if not self.prediction_history['predictions']:
            return
            
        # Add timestamp to track when this update occurred
        update_time = datetime.datetime.now().isoformat()
            
        # Get most recent prediction
        last_prediction = self.prediction_history['predictions'][-1]
        
        # Calculate accuracy metrics
        accuracy = self._calculate_accuracy(last_prediction, new_draw)
        self.prediction_history['accuracy'].append(accuracy)
        
        # Analyze which tools performed best for this prediction
        tool_performance = self._analyze_tool_performance(last_prediction, new_draw)
        self.prediction_history['feedback'].append(tool_performance)
        
        # Adjust feature weights based on tool performance
        self._adjust_weights(tool_performance)
        
        # Clear prediction cache since dataframe has changed
        self.clear_cache()
        
        # Retrain models with new data
        if self.df_history is not None:
            self._train_models()
        
        # Save updated history
        self._save_history()
    
    def _calculate_accuracy(self, prediction: Dict, actual: Dict) -> Dict:
        """
        Calculate accuracy metrics between prediction and actual draw.
        
        Args:
            prediction: Dictionary containing predicted numbers
            actual: Dictionary containing actual draw numbers
        
        Returns:
            Dictionary with accuracy metrics
        """
        # Extract predicted and actual numbers
        pred_whites = set(prediction['white_numbers'])
        actual_whites = set(actual['white_numbers'])
        pred_pb = prediction['powerball']
        actual_pb = actual['powerball']
        
        # Calculate metrics
        white_matches = len(pred_whites.intersection(actual_whites))
        pb_match = pred_pb == actual_pb
        
        return {
            'white_matches': white_matches,
            'pb_match': pb_match,
            'white_accuracy': white_matches / WHITE_BALLS_COUNT,
            'full_match': white_matches == WHITE_BALLS_COUNT and pb_match
        }
    
    def _analyze_tool_performance(self, prediction: Dict, actual: Dict) -> Dict:
        """
        Analyze which prediction tools performed best.
        
        Args:
            prediction: Dictionary containing predicted numbers
            actual: Dictionary containing actual draw numbers
        
        Returns:
            Dictionary with tool performance scores
        """
        tool_scores = {}
        
        # For each tool, calculate how well it predicted the actual numbers
        for tool, tool_predictions in prediction['tool_contributions'].items():
            tool_whites = set(tool_predictions['white_numbers'])
            actual_whites = set(actual['white_numbers'])
            
            white_matches = len(tool_whites.intersection(actual_whites))
            pb_match = tool_predictions['powerball'] == actual['powerball']
            
            # Score is weighted combination of white ball and powerball accuracy
            tool_scores[tool] = (white_matches / WHITE_BALLS_COUNT) * 0.8 + (1 if pb_match else 0) * 0.2
        
        return tool_scores
    
    def _adjust_weights(self, tool_performance: Dict):
        """
        Adjust feature weights based on tool performance.
        
        Args:
            tool_performance: Dictionary with tool performance scores
        """
        # Normalize scores
        total_score = sum(tool_performance.values())
        if total_score == 0:
            return
            
        normalized_scores = {tool: score/total_score for tool, score in tool_performance.items()}
        
        # Apply learning rate to weight adjustments (gradual change)
        learning_rate = 0.1
        for tool in self.feature_weights:
            if tool in normalized_scores:
                # Compute weighted average of old and new weights
                self.feature_weights[tool] = (1 - learning_rate) * self.feature_weights[tool] + \
                                             learning_rate * normalized_scores[tool]
        
        # Renormalize weights to ensure they sum to 1
        total_weight = sum(self.feature_weights.values())
        self.feature_weights = {tool: weight/total_weight for tool, weight in self.feature_weights.items()}
    
    def _train_models(self):
        """Train machine learning models with cross-validation on historical data."""
        if self.df_history is None or len(self.df_history) < 50:
            return
        
        try:
            # Prepare data
            df = self.df_history.sort_values('draw_date').copy()
            df['draw_date'] = pd.to_datetime(df['draw_date'])
            
            # Feature engineering
            X = self._engineer_features(df)
            
            # Train white ball model using a regression approach with all white balls as targets
            y_white = df[['n1', 'n2', 'n3', 'n4', 'n5']].values
            
            # Use more powerful XGBoost regressor with cross-validation
            from sklearn.model_selection import cross_val_score, TimeSeriesSplit
            from sklearn.ensemble import GradientBoostingRegressor
            
            # Create time series cross-validation splits
            tscv = TimeSeriesSplit(n_splits=5)
            
            # White ball models with cross-validation
            # Use MultiOutputRegressor for white balls since we're predicting 5 values at once
            base_regressor = GradientBoostingRegressor(n_estimators=100, random_state=42)
            white_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', MultiOutputRegressor(base_regressor))
            ])
            
            # Perform cross-validation and store scores
            white_scores = []
            for train_idx, val_idx in tscv.split(X):
                # Train on time series split
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y_white[train_idx], y_white[val_idx]
                
                # Fit on training data
                white_pipeline.fit(X_train, y_train)
                
                # Calculate validation error
                val_pred = white_pipeline.predict(X_val)
                # Use mean absolute error for evaluation
                mae = np.mean(np.abs(y_val - val_pred))
                white_scores.append(mae)
            
            # Log cross-validation results
            avg_mae = np.mean(white_scores)
            logging.info(f"White ball model cross-validation MAE: {avg_mae:.2f}")
            
            # Refit on entire dataset for final model
            white_pipeline.fit(X, y_white)
            if 'white_balls' in self.models:
                self.models['white_balls'] = white_pipeline
            
            # Train powerball model with cross-validation
            y_pb = df['powerball'].values
            
            pb_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', GradientBoostingRegressor(n_estimators=100, random_state=42))
            ])
            
            # Cross-validate powerball model
            pb_scores = []
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y_pb[train_idx], y_pb[val_idx]
                
                pb_pipeline.fit(X_train, y_train)
                val_pred = pb_pipeline.predict(X_val)
                mae = np.mean(np.abs(y_val - val_pred))
                pb_scores.append(mae)
            
            # Log results
            avg_pb_mae = np.mean(pb_scores)
            logging.info(f"Powerball model cross-validation MAE: {avg_pb_mae:.2f}")
            
            # Refit on entire dataset
            pb_pipeline.fit(X, y_pb)
            if 'powerball' in self.models:
                self.models['powerball'] = pb_pipeline
            
            # Store cross-validation performance for potential adaptive weighting
            self.model_cv_performance = {
                'white_balls': avg_mae,
                'powerball': avg_pb_mae
            }
        
        except Exception as e:
            logging.error(f"Error training models: {e}")
            
    def _calculate_feature_importance(self) -> Dict:
        """
        Calculate feature importance for ML models.
        
        Returns:
            Dictionary with feature importance data for visualization
        """
        if self.df_history is None or len(self.df_history) < 50:
            return {'white_balls': None, 'powerball': None}
            
        if self.models['white_balls'] is None or self.models['powerball'] is None:
            return {'white_balls': None, 'powerball': None}
            
        try:
            # Get feature names from engineered features
            df = self.df_history.copy()
            df['draw_date'] = pd.to_datetime(df['draw_date'])
            
            # Create feature names based on the feature engineering process
            feature_names = []
            
            # Day of week features
            for i in range(7):
                feature_names.append(f'dow_{i}')
                
            # Draw number
            feature_names.append('draw_number')
            
            # Sum statistics for different windows
            for window in [5, 10, 20]:
                if len(df) >= window:
                    feature_names.extend([f'sum_mean_{window}', f'sum_std_{window}'])
            
            # Lag features
            draw_columns = ['n1', 'n2', 'n3', 'n4', 'n5', 'powerball']
            for col in draw_columns:
                for lag in [1, 2, 3]:
                    if len(df) > lag:
                        feature_names.append(f'{col}_lag{lag}')
            
            # Extract feature importances from models
            importances = {}
            
            for model_name, model in self.models.items():
                if model is not None and hasattr(model, 'named_steps') and 'regressor' in model.named_steps:
                    regressor = model.named_steps['regressor']
                    if hasattr(regressor, 'feature_importances_'):
                        # Create dataframe of importances
                        feature_imp = regressor.feature_importances_
                        
                        # Create dataframe (safely handle mismatched lengths)
                        imp_df = pd.DataFrame({
                            'Feature': feature_names[:len(feature_imp)] if len(feature_names) >= len(feature_imp) 
                                      else feature_names + [f'Feature_{i}' for i in range(len(feature_names), len(feature_imp))],
                            'Importance': feature_imp
                        })
                        
                        # Sort by importance
                        imp_df = imp_df.sort_values('Importance', ascending=False).reset_index(drop=True)
                        
                        # Store in results
                        importances[model_name] = imp_df
                    else:
                        # Model doesn't have feature_importances_ attribute
                        importances[model_name] = None
                else:
                    importances[model_name] = None
            
            return importances
        except Exception as e:
            logging.error(f"Error calculating feature importance: {e}")
            return {'white_balls': None, 'powerball': None}
    
    def _engineer_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Engineer features for machine learning models.
        
        Args:
            df: DataFrame with historical draw data
            
        Returns:
            Array of engineered features
        """
        # Create a copy to avoid SettingWithCopyWarning
        df = df.copy()
        features = []
        
        # Add timestamp features
        df['day_of_week'] = df['draw_date'].dt.dayofweek
        df['month'] = df['draw_date'].dt.month
        df['year'] = df['draw_date'].dt.year
        
        # One-hot encode day of week
        dow_dummies = pd.get_dummies(df['day_of_week'], prefix='dow')
        features.append(dow_dummies)
        
        # Add trend features
        df['draw_number'] = np.arange(len(df))  # Using numpy for performance
        features.append(df[['draw_number']])
        
        # Add rolling statistics for white ball sums
        white_sums = df[['n1', 'n2', 'n3', 'n4', 'n5']].sum(axis=1)
        
        # Create separate dataframe for rolling features to avoid chained indexing
        roll_features = pd.DataFrame(index=df.index)
        for window in [5, 10, 20]:
            if len(df) >= window:
                roll_features[f'sum_mean_{window}'] = white_sums.rolling(window=window, min_periods=1).mean()
                roll_features[f'sum_std_{window}'] = white_sums.rolling(window=window, min_periods=1).std()
        
        features.append(roll_features)
        
        # Add lag features for previous draws - vectorized approach
        draw_columns = ['n1', 'n2', 'n3', 'n4', 'n5', 'powerball']
        lag_features = pd.DataFrame(index=df.index)
        
        # Create all lag features in one go - optimized for performance and memory usage
        if len(df) > 3:  # Ensure we have enough data for lag 3
            # Using vectorized operations for better performance
            for col in draw_columns:
                values = df[col].values  # Get numpy array for faster operations
                for lag in [1, 2, 3]:
                    # Create lag columns with numpy operations (faster than pandas shift)
                    lagged = np.zeros(len(df))
                    if lag < len(df):
                        lagged[lag:] = values[:-lag]
                    lag_features[f'{col}_lag{lag}'] = lagged
            
            # Add all lag features together
            features.append(lag_features)
        
        # Combine all features
        if features:
            return pd.concat(features, axis=1).fillna(0).values
        else:
            # Return basic features if we don't have enough data for advanced features
            return np.zeros((len(df), 1))
    
    def generate_frequency_predictions(self) -> Dict:
        """
        Generate predictions based on frequency analysis.
        
        Returns:
            Dictionary with predictions and probability scores
        """
        if self.df_history is None or self.df_history.empty:
            return {'white_numbers': [], 'powerball': 0, 'probability': 0}
        
        # Use cached results if available with the same dataframe
        cache_key = f"freq_{id(self.df_history)}_{len(self.df_history)}"
        if cache_key in self._prediction_cache:
            return self._prediction_cache[cache_key]
            
        # Calculate frequency table
        freq_df = calc_frequency(self.df_history)
        
        # Sort numbers by frequency for white balls and powerball separately
        white_freq = freq_df.copy()
        pb_freq = self.df_history['powerball'].value_counts().reset_index()
        pb_freq.columns = ['number', 'frequency']
        
        # Add probability based on normalized frequency
        white_freq['probability'] = white_freq['frequency'] / white_freq['frequency'].sum()
        pb_freq['probability'] = pb_freq['frequency'] / pb_freq['frequency'].sum()
        
        # Use weighted random selection instead of always picking top frequencies
        # This eliminates bias while still favoring frequent numbers
        
        # For white balls: weighted random selection
        white_weights = white_freq['frequency'].values
        white_balls = np.random.choice(
            white_freq['number'].values,
            size=WHITE_BALLS_COUNT,
            replace=False,
            p=white_weights / white_weights.sum()
        ).tolist()
        
        # For powerball: weighted random selection
        pb_weights = pb_freq['frequency'].values
        powerball = np.random.choice(
            pb_freq['number'].values,
            p=pb_weights / pb_weights.sum()
        )
        
        # Calculate combined probability
        white_probs = white_freq[white_freq['number'].isin(white_balls)]['probability'].tolist()
        pb_prob = pb_freq[pb_freq['number'] == powerball]['probability'].iloc[0]
        
        # Simplified probability calculation
        combined_prob = np.mean(white_probs) * pb_prob
        
        result = {
            'white_numbers': white_balls,
            'powerball': powerball,
            'probability': combined_prob
        }
        
        # Cache the result
        self._prediction_cache[cache_key] = result
        
        return result
    
    def generate_recency_predictions(self) -> Dict:
        """
        Generate predictions based on recency (days since last appearance).
        
        Returns:
            Dictionary with predictions and probability scores
        """
        if self.df_history is None or self.df_history.empty:
            return {'white_numbers': [], 'powerball': 0, 'probability': 0}
            
        # Use cached results if available with the same dataframe
        cache_key = f"recency_{id(self.df_history)}_{len(self.df_history)}"
        if cache_key in self._prediction_cache:
            return self._prediction_cache[cache_key]
            
        # Calculate days since last appearance for each number
        days_since_df = _days_since_last(self.df_history)
        
        # Convert to a more usable format and separate white balls and powerball
        # Create a copy to avoid SettingWithCopyWarning
        days_since_df = days_since_df.copy()
        days_since_df['is_powerball'] = days_since_df['Number'].apply(lambda x: x in range(1, 27))
        white_df = days_since_df[~days_since_df['is_powerball']].copy()
        pb_df = days_since_df[days_since_df['is_powerball']].copy()
        
        # We want numbers that haven't appeared for a while but not too long
        # Find the median gap
        gaps = _all_gaps(self.df_history)
        median_gap = gaps.median()
        
        # Calculate score as proximity to optimal recency (median gap)
        white_df = white_df.copy()  # Create a copy to avoid SettingWithCopyWarning
        pb_df = pb_df.copy()  # Create a copy to avoid SettingWithCopyWarning
        white_df.loc[:, 'recency_score'] = 1 / (1 + abs(white_df['Days Since Last Drawn'] - median_gap))
        pb_df.loc[:, 'recency_score'] = 1 / (1 + abs(pb_df['Days Since Last Drawn'] - median_gap))
        
        # Select top WHITE_BALLS_COUNT white balls by recency score
        white_balls = white_df.sort_values('recency_score', ascending=False).head(WHITE_BALLS_COUNT)['Number'].tolist()
        powerball = pb_df.sort_values('recency_score', ascending=False).head(1)['Number'].iloc[0]
        
        # Calculate combined probability
        white_scores = white_df[white_df['Number'].isin(white_balls)]['recency_score'].tolist()
        pb_score = float(pb_df[pb_df['Number'] == powerball]['recency_score'].iloc[0])
        
        # Normalized probability
        white_prob = np.mean(white_scores) / white_df['recency_score'].sum()
        pb_prob = pb_score / pb_df['recency_score'].sum()
        combined_prob = white_prob * pb_prob
        
        result = {
            'white_numbers': white_balls,
            'powerball': powerball,
            'probability': combined_prob
        }
        
        # Cache the result
        self._prediction_cache[cache_key] = result
        
        return result
    
    def generate_trend_predictions(self) -> Dict:
        """
        Generate predictions based on time trend analysis.
        
        Returns:
            Dictionary with predictions and probability scores
        """
        if self.df_history is None or len(self.df_history) < 10:
            return {'white_numbers': [], 'powerball': 0, 'probability': 0}
            
        # Use cached results if available with the same dataframe
        cache_key = f"trend_{id(self.df_history)}_{len(self.df_history)}"
        if cache_key in self._prediction_cache:
            return self._prediction_cache[cache_key]
            
        # Aggregate by month to see trends
        try:
            agg_df = _aggregate(self.df_history, 'ME')
            
            # Calculate trending score
            trending = _calc_trending(agg_df, window=3)
            
            # Convert trending Series to DataFrame for easier filtering
            trend_df = trending.reset_index()
            trend_df.columns = ['number', 'trend_score']
            
            # Create separate trending scores for white balls and powerball
            # Create copies to avoid SettingWithCopyWarning
            white_trend_df = trend_df[trend_df['number'] <= WHITE_MAX].copy()
            pb_trend_df = trend_df[trend_df['number'] <= PB_MAX].copy()
            
            # Select top WHITE_BALLS_COUNT white balls by trending score
            white_balls = white_trend_df.sort_values('trend_score', ascending=False).head(WHITE_BALLS_COUNT)['number'].tolist()
            
            # Select top powerball
            if not pb_trend_df.empty:
                powerball = pb_trend_df.sort_values('trend_score', ascending=False).head(1)['number'].iloc[0]
            else:
                powerball = 1  # Default if no data available
            
            # Calculate probability based on normalized trending scores
            white_scores = []
            for num in white_balls:
                score = white_trend_df[white_trend_df['number'] == num]['trend_score']
                if not score.empty:
                    white_scores.append(float(score.iloc[0]))
                else:
                    white_scores.append(0.0)
                    
            if not pb_trend_df.empty:
                pb_score = float(pb_trend_df[pb_trend_df['number'] == powerball]['trend_score'].iloc[0]) \
                           if not pb_trend_df[pb_trend_df['number'] == powerball].empty else 0.0
            else:
                pb_score = 0.0
            
            # Normalize scores
            white_sum = white_trend_df['trend_score'].sum() if not white_trend_df.empty else 1.0
            pb_sum = pb_trend_df['trend_score'].sum() if not pb_trend_df.empty else 1.0
            
            if white_sum > 0 and pb_sum > 0 and white_scores:
                white_prob = np.mean(white_scores) / white_sum
                pb_prob = pb_score / pb_sum
                combined_prob = white_prob * pb_prob
            else:
                combined_prob = 0.0001  # Small non-zero probability
                
            result = {
                'white_numbers': white_balls,
                'powerball': powerball,
                'probability': combined_prob
            }
            
            # Cache the result
            self._prediction_cache[cache_key] = result
            
            return result
        except Exception as e:
            logging.warning(f"Error in trend prediction: {e}")
            return {'white_numbers': [], 'powerball': 0, 'probability': 0}
    
    def generate_combo_predictions(self) -> Dict:
        """
        Generate predictions based on combinatorial analysis.
        
        Returns:
            Dictionary with predictions and probability scores
        """
        if self.df_history is None or self.df_history.empty:
            return {'white_numbers': [], 'powerball': 0, 'probability': 0}
            
        # Use cached results if available with the same dataframe
        cache_key = f"combo_{id(self.df_history)}_{len(self.df_history)}"
        if cache_key in self._prediction_cache:
            return self._prediction_cache[cache_key]
            
        try:
            # Get most frequent pairs
            combo_counter = _count_combos(self.df_history, 2, include_pb=False)
            
            # Assign score to each number based on its appearance in top pairs
            number_scores = Counter()
            for combo, count in combo_counter.most_common(20):  # Use top 20 pairs
                for num in combo:
                    number_scores[num] += count
            
            # Select top WHITE_BALLS_COUNT white balls by score
            white_balls = [num for num, _ in number_scores.most_common(WHITE_BALLS_COUNT)]
            
            # For powerball, we'll use frequency directly
            pb_counts = self.df_history['powerball'].value_counts()
            powerball = pb_counts.idxmax()
            
            # Calculate probability
            total_pairs = sum(combo_counter.values())
            white_scores = [number_scores[num] for num in white_balls]
            total_scores = sum(number_scores.values())
            pb_score = pb_counts[powerball]
            total_pb = pb_counts.sum()
            
            white_prob = np.mean(white_scores) / total_scores if total_scores > 0 else 0
            pb_prob = pb_score / total_pb if total_pb > 0 else 0
            combined_prob = white_prob * pb_prob
            
            result = {
                'white_numbers': white_balls,
                'powerball': powerball,
                'probability': combined_prob
            }
            
            # Cache the result
            self._prediction_cache[cache_key] = result
            
            return result
        except Exception as e:
            logging.warning(f"Error in combo prediction: {e}")
            return {'white_numbers': [], 'powerball': 0, 'probability': 0}
    
    def generate_sum_predictions(self) -> Dict:
        """
        Generate predictions based on sum analysis.
        
        Returns:
            Dictionary with predictions and probability scores
        """
        if self.df_history is None or self.df_history.empty:
            return {'white_numbers': [], 'powerball': 0, 'probability': 0}
            
        # Use cached results if available with the same dataframe
        cache_key = f"sum_{id(self.df_history)}_{len(self.df_history)}"
        if cache_key in self._prediction_cache:
            return self._prediction_cache[cache_key]
            
        try:
            # Calculate historical sums
            sums = _prep_sums(self.df_history, include_pb=False)
            
            # Find the most common sum range
            hist, bin_edges = np.histogram(sums, bins=10)
            most_common_bin = np.argmax(hist)
            
            # Target sum range
            min_sum = bin_edges[most_common_bin]
            max_sum = bin_edges[most_common_bin + 1]
            
            # Generate combinations that have sums in this range
            # We'll use a random approach since exhaustive search would be too expensive
            valid_combos = []
            attempts = 0
            max_attempts = 1000
            
            while len(valid_combos) < 10 and attempts < max_attempts:
                attempts += 1
                # Generate random 5 numbers
                combo = sorted(random.sample(range(WHITE_MIN, WHITE_MAX + 1), WHITE_BALLS_COUNT))
                combo_sum = sum(combo)
                
                if min_sum <= combo_sum <= max_sum:
                    valid_combos.append(combo)
            
            # If we failed to find valid combos, fall back to frequency
            if not valid_combos:
                return self.generate_frequency_predictions()
                
            # Select a combo at random from valid ones
            white_balls = random.choice(valid_combos)
            
            # For powerball, use frequency
            pb_counts = self.df_history['powerball'].value_counts()
            powerball = pb_counts.idxmax()
            
            # Calculate probability
            bin_prob = hist[most_common_bin] / len(sums)
            pb_prob = pb_counts[powerball] / pb_counts.sum()
            
            # Very crude probability estimation
            combo_count = 1
            for i in range(WHITE_MIN, WHITE_MAX + 1):
                for j in range(i + 1, WHITE_MAX + 1):
                    for k in range(j + 1, WHITE_MAX + 1):
                        for l in range(k + 1, WHITE_MAX + 1):
                            for m in range(l + 1, WHITE_MAX + 1):
                                combo = [i, j, k, l, m]
                                if min_sum <= sum(combo) <= max_sum:
                                    combo_count += 1
            
            white_prob = bin_prob / combo_count if combo_count > 0 else 0
            combined_prob = white_prob * pb_prob
            
            result = {
                'white_numbers': white_balls,
                'powerball': powerball,
                'probability': combined_prob
            }
            
            # Cache the result
            self._prediction_cache[cache_key] = result
            
            return result
        except Exception as e:
            logging.warning(f"Error in sum prediction: {e}")
            return {'white_numbers': [], 'powerball': 0, 'probability': 0}
    
    def generate_dow_predictions(self) -> Dict:
        """
        Generate predictions based on day of week analysis.
        
        Returns:
            Dictionary with predictions and probability scores
        """
        if self.df_history is None or self.df_history.empty:
            return {'white_numbers': [], 'powerball': 0, 'probability': 0}
            
        # Use cached results if available with the same dataframe
        cache_key = f"dow_{id(self.df_history)}_{len(self.df_history)}"
        if cache_key in self._prediction_cache:
            return self._prediction_cache[cache_key]
            
        try:
            # Calculate day of week stats
            dow_stats = _compute_stats(self.df_history)
            
            # Verify the column exists before accessing it
            if 'Δ_vs_expected' in dow_stats.columns:
                # Find the day with highest delta from expected
                best_day = dow_stats.loc[dow_stats['Δ_vs_expected'].idxmax(), 'Day']
            elif 'Δ_from_expected' in dow_stats.columns:
                # For backward compatibility with older implementations
                best_day = dow_stats.loc[dow_stats['Δ_from_expected'].idxmax(), 'Day']
            else:
                # Fall back to the day with highest frequency if delta column not found
                logging.warning("Neither 'Δ_vs_expected' nor 'Δ_from_expected' found in dow_stats columns")
                best_day = dow_stats.loc[dow_stats['Frequency'].idxmax(), 'Day']
            
            # Filter draws that happened on this day
            df_history_copy = self.df_history.copy()
            df_history_copy['draw_date'] = pd.to_datetime(df_history_copy['draw_date'])
            df_history_copy['dow'] = df_history_copy['draw_date'].dt.day_name()
            
            best_day_draws = df_history_copy[df_history_copy['dow'] == best_day]
            
            # If we don't have draws on this day, fall back to frequency
            if best_day_draws.empty:
                return self.generate_frequency_predictions()
                
            # Generate predictions based on frequency within these draws
            white_counts = Counter()
            for col in ['n1', 'n2', 'n3', 'n4', 'n5']:
                white_counts.update(best_day_draws[col])
                
            pb_counts = Counter(best_day_draws['powerball'])
            
            # Use weighted random selection for day-of-week predictions too
            if white_counts:
                white_numbers = list(white_counts.keys())
                white_weights = list(white_counts.values())
                total_weight = sum(white_weights)
                white_probs = [w/total_weight for w in white_weights]
                
                white_balls = np.random.choice(
                    white_numbers,
                    size=min(WHITE_BALLS_COUNT, len(white_numbers)),
                    replace=False,
                    p=white_probs
                ).tolist()
            else:
                white_balls = []
            
            if pb_counts:
                pb_numbers = list(pb_counts.keys())
                pb_weights = list(pb_counts.values())
                total_pb_weight = sum(pb_weights)
                pb_probs = [w/total_pb_weight for w in pb_weights]
                
                powerball = np.random.choice(pb_numbers, p=pb_probs)
            else:
                powerball = 1
            
            # Calculate probability
            day_prob = dow_stats.loc[dow_stats['Day'] == best_day, 'Frequency'].iloc[0] / dow_stats['Frequency'].sum()
            
            white_scores = [white_counts[num] for num in white_balls]
            total_white = sum(white_counts.values())
            
            pb_score = pb_counts[powerball] if powerball in pb_counts else 0
            total_pb = sum(pb_counts.values())
            
            white_prob = np.mean(white_scores) / total_white if total_white > 0 else 0
            pb_prob = pb_score / total_pb if total_pb > 0 else 0
            
            combined_prob = day_prob * white_prob * pb_prob
            
            result = {
                'white_numbers': white_balls,
                'powerball': powerball,
                'probability': combined_prob
            }
            
            # Cache the result
            self._prediction_cache[cache_key] = result
            
            return result
        except Exception as e:
            logging.warning(f"Error in day of week prediction: {e}")
            return {'white_numbers': [], 'powerball': 0, 'probability': 0}
    
    def generate_ml_predictions(self) -> Dict:
        """
        Generate predictions based on machine learning models.
        
        Returns:
            Dictionary with predictions and probability scores
        """
        if self.df_history is None or self.df_history.empty:
            return {'white_numbers': [], 'powerball': 0, 'probability': 0}
            
        # Use cached results if available with the same dataframe
        cache_key = f"ml_{id(self.df_history)}_{len(self.df_history)}"
        if cache_key in self._prediction_cache:
            return self._prediction_cache[cache_key]
            
        # Check if models exist
        need_training = False
        if 'white_balls' not in self.models or self.models['white_balls'] is None:
            need_training = True
        if 'powerball' not in self.models or self.models['powerball'] is None:
            need_training = True
            
        # Train models if they don't exist
        if need_training:
            self._train_models()
            # Also generate feature importance when training
            try:
                self.feature_importance = self._calculate_feature_importance()
            except Exception as e:
                logging.warning(f"Could not calculate feature importance: {e}")
                self.feature_importance = {'white_balls': None, 'powerball': None}
            
            # If still no models, fall back to frequency
            if 'white_balls' not in self.models or self.models['white_balls'] is None:
                return self.generate_frequency_predictions()
        
        try:
            # Prepare features for prediction
            df = self.df_history.sort_values('draw_date').copy()
            X = self._engineer_features(df)
            
            if X.size == 0:
                return self.generate_frequency_predictions()
                
            # Get the latest feature set
            X_latest = X[-1:].reshape(1, -1)
            
            white_balls = []
            
            # Check if we can make predictions
            if 'white_balls' in self.models and self.models['white_balls'] is not None:
                # Predict white balls (returns a 2D array with one row of 5 predictions)
                white_preds_raw = self.models['white_balls'].predict(X_latest)[0]
                
                # Round to integers and ensure they're within bounds
                white_preds = [max(WHITE_MIN, min(WHITE_MAX, int(round(p)))) for p in white_preds_raw]
                
                # Ensure all predicted numbers are unique
                seen = set()
                for p in white_preds:
                    while p in seen:
                        p = (p % WHITE_MAX) + 1
                    seen.add(p)
                    white_balls.append(p)
                    
                white_balls.sort()
            else:
                # Fallback if model doesn't exist
                white_balls = [1, 2, 3, 4, 5]  # Default
            
            # Predict powerball
            if 'powerball' in self.models and self.models['powerball'] is not None:
                pb_pred_raw = self.models['powerball'].predict(X_latest)[0]
                powerball = max(PB_MIN, min(PB_MAX, int(round(pb_pred_raw))))
            else:
                # Fallback if model doesn't exist
                powerball = 1  # Default
            
            # For probability, we'll use a placeholder
            # Ideally, we'd use model confidence metrics, but this is a simplified example
            combined_prob = 0.0001  # Just a placeholder
            
            result = {
                'white_numbers': white_balls,
                'powerball': powerball,
                'probability': combined_prob
            }
            
            # Cache the result
            self._prediction_cache[cache_key] = result
            
            return result
        except Exception as e:
            logging.warning(f"Error in ML prediction: {e}")
            return self.generate_frequency_predictions()
    
    def generate_weighted_predictions(self, count: int = 5) -> List[Dict]:
        """
        Generate multiple weighted predictions based on all analysis tools.
        
        Args:
            count: Number of predictions to generate
            
        Returns:
            List of dictionaries with predictions and probability scores
        """
        if self.df_history is None or self.df_history.empty:
            return [{'white_numbers': [1, 2, 3, 4, 5], 'powerball': 1, 'probability': 0, 'sources': {}}] * count
        
        # Generate predictions from each tool
        tool_predictions = {
            'frequency': self.generate_frequency_predictions(),
            'recency': self.generate_recency_predictions(),
            'trends': self.generate_trend_predictions(),
            'combo': self.generate_combo_predictions(),
            'sum': self.generate_sum_predictions(),
            'dow': self.generate_dow_predictions(),
            'ml': self.generate_ml_predictions()
        }
        
        # Create a pool of potential white ball numbers with scores
        white_ball_scores = {}
        for tool, weight in self.feature_weights.items():
            pred = tool_predictions[tool]
            if not pred['white_numbers']:
                continue
                
            for num in pred['white_numbers']:
                if num not in white_ball_scores:
                    white_ball_scores[num] = 0
                white_ball_scores[num] += weight
        
        # Create a pool of potential powerball numbers with scores
        pb_scores = {}
        for tool, weight in self.feature_weights.items():
            pred = tool_predictions[tool]
            if pred['powerball'] == 0:
                continue
                
            pb = pred['powerball']
            if pb not in pb_scores:
                pb_scores[pb] = 0
            pb_scores[pb] += weight
        
        # Generate 'count' predictions
        predictions = []
        seen_combinations = set()
        
        for _ in range(count):
            # Select white balls based on weighted random selection
            selected_whites = []
            remaining_whites = white_ball_scores.copy()
            
            for _ in range(WHITE_BALLS_COUNT):
                if not remaining_whites:
                    # Fill in with random numbers if we run out
                    while len(selected_whites) < WHITE_BALLS_COUNT:
                        num = random.randint(WHITE_MIN, WHITE_MAX)
                        if num not in selected_whites:
                            selected_whites.append(num)
                    break
                
                # Weighted random selection
                total = sum(remaining_whites.values())
                r = random.uniform(0, total)
                running_sum = 0
                
                for num, score in sorted(remaining_whites.items()):
                    running_sum += score
                    if running_sum >= r:
                        selected_whites.append(num)
                        del remaining_whites[num]
                        break
                        
            selected_whites.sort()
            
            # Select powerball based on weighted random selection
            selected_pb = 1  # Default
            if pb_scores:
                total = sum(pb_scores.values())
                r = random.uniform(0, total)
                running_sum = 0
                
                for pb, score in sorted(pb_scores.items()):
                    running_sum += score
                    if running_sum >= r:
                        selected_pb = pb
                        break
            
            # Generate a unique key for this combination
            combo_key = tuple(selected_whites + [selected_pb])
            
            # If we've seen this combo before, try again with some randomness
            attempts = 0
            while combo_key in seen_combinations and attempts < 10:
                attempts += 1
                # Add some randomness
                if random.random() < 0.5:
                    # Replace a white ball
                    idx = random.randint(0, WHITE_BALLS_COUNT - 1)
                    new_num = random.randint(WHITE_MIN, WHITE_MAX)
                    while new_num in selected_whites:
                        new_num = random.randint(WHITE_MIN, WHITE_MAX)
                    selected_whites[idx] = new_num
                    selected_whites.sort()
                else:
                    # Replace powerball
                    selected_pb = random.randint(PB_MIN, PB_MAX)
                
                combo_key = tuple(selected_whites + [selected_pb])
            
            seen_combinations.add(combo_key)
            
            # Calculate probability score based on individual number scores
            white_prob = np.mean([white_ball_scores.get(num, 0.1) for num in selected_whites])
            pb_prob = pb_scores.get(selected_pb, 0.1)
            combined_prob = white_prob * pb_prob / (sum(self.feature_weights.values()) ** 2)
            
            # Record which tool contributed to each selected number
            sources = {num: [] for num in selected_whites}
            sources[selected_pb] = []
            
            for tool, pred in tool_predictions.items():
                for num in pred['white_numbers']:
                    if num in sources:
                        sources[num].append(tool)
                if pred['powerball'] == selected_pb:
                    sources[selected_pb].append(tool)
            
            # Create prediction entry
            prediction = {
                'white_numbers': selected_whites,
                'powerball': selected_pb,
                'probability': combined_prob,
                'tool_contributions': tool_predictions,  # Store all tool predictions for learning
                'sources': sources,  # Track which tools suggested each number
                'timestamp': datetime.datetime.now().isoformat(),  # Add current timestamp
                'prediction_for_date': self._calculate_next_draw_date()  # Add projected draw date
            }
            
            predictions.append(prediction)
        
        # Sort predictions by probability
        predictions.sort(key=lambda x: x['probability'], reverse=True)
        
        # Store predictions for learning using new storage manager
        from .prediction_storage_refactor import PredictionStorageManager
        storage_manager = PredictionStorageManager(HISTORY_FILE_PATH)
        storage_manager.store_prediction(predictions[0])
        
        return predictions

def get_enhanced_predictions(df_history: pd.DataFrame, count: int = 5) -> List[Dict]:
    """
    Generate enhanced predictions using the integrated prediction system.
    
    Args:
        df_history: DataFrame with historical draw data
        count: Number of predictions to generate
        
    Returns:
        List of dictionaries with predictions
    """
    # Initialize prediction system
    ps = PredictionSystem(df_history)
    
    # Generate predictions
    return ps.generate_weighted_predictions(count)

def update_model_with_new_draw(df_history: pd.DataFrame, new_draw: Dict):
    """
    Update the prediction model with a new draw result.
    
    Args:
        df_history: DataFrame with historical draw data
        new_draw: Dictionary with the new draw results
    """
    # Initialize prediction system
    ps = PredictionSystem(df_history)
    
    # Update with new draw
    ps.update_with_new_draw(new_draw)