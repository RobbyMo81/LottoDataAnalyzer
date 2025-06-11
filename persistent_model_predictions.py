"""
Persistent Model Predictions System
==================================
Implements SQLite-based storage for model-specific predictions with comprehensive data management.

This system ensures that predictions from each ML model (Ridge Regression, Random Forest, 
Gradient Boosting) are stored persistently, independently managed, and always accessible
without requiring model retraining.
"""

import sqlite3
import json
import datetime
import logging
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import pandas as pd
from .datetime_manager import datetime_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelPrediction:
    """Data class for model prediction records."""
    model_name: str
    prediction_id: str
    white_numbers: List[int]
    powerball: int
    probability: float
    features_used: List[str]
    hyperparameters: Dict[str, Any]
    performance_metrics: Dict[str, float]
    created_at: datetime.datetime
    prediction_set_id: str  # Groups 5 predictions together
    version: int = 1

class PersistentModelPredictionManager:
    """
    Manages persistent storage of model predictions using SQLite database.
    
    Key Features:
    - Model-specific prediction storage
    - Version control and retention policies
    - Data integrity checks
    - Independent model management
    """
    
    def __init__(self, db_path: str = "data/model_predictions.db"):
        """Initialize the prediction manager with SQLite database."""
        self.db_path = db_path
        self.ensure_database_directory()
        self.initialize_database()
        
    def ensure_database_directory(self):
        """Ensure the database directory exists."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
    def initialize_database(self):
        """Initialize SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create predictions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    prediction_id TEXT NOT NULL,
                    prediction_set_id TEXT NOT NULL,
                    white_numbers TEXT NOT NULL,  -- JSON array
                    powerball INTEGER NOT NULL,
                    probability REAL NOT NULL,
                    features_used TEXT NOT NULL,  -- JSON array
                    hyperparameters TEXT NOT NULL,  -- JSON object
                    performance_metrics TEXT NOT NULL,  -- JSON object
                    created_at TIMESTAMP NOT NULL,
                    version INTEGER DEFAULT 1,
                    is_active BOOLEAN DEFAULT TRUE,
                    UNIQUE(model_name, prediction_id, version)
                )
            ''')
            
            # Create prediction sets table for grouping
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS prediction_sets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    set_id TEXT UNIQUE NOT NULL,
                    model_name TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    is_current BOOLEAN DEFAULT TRUE,
                    total_predictions INTEGER DEFAULT 5,
                    training_duration REAL,
                    notes TEXT
                )
            ''')
            
            # Create indexes for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_model_name ON model_predictions(model_name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_prediction_set ON model_predictions(prediction_set_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_active ON model_predictions(is_active)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_current_sets ON prediction_sets(is_current)')
            
            conn.commit()
            logger.info(f"Database initialized at {self.db_path}")
    
    def store_model_predictions(self, 
                              model_name: str,
                              predictions: List[Dict[str, Any]],
                              hyperparameters: Dict[str, Any],
                              performance_metrics: Dict[str, float],
                              features_used: List[str],
                              training_duration: float = 0.0,
                              notes: str = "") -> str:
        """
        Store a complete set of predictions for a specific model.
        
        Args:
            model_name: Name of the model (e.g., 'Ridge Regression')
            predictions: List of 5 prediction dictionaries
            hyperparameters: Model configuration used
            performance_metrics: Training performance metrics
            features_used: List of feature names used in training
            training_duration: Time taken for training
            notes: Optional notes about this prediction set
            
        Returns:
            prediction_set_id: Unique identifier for this prediction set
        """
        if len(predictions) < 1:
            raise ValueError(f"Expected at least 1 prediction, got {len(predictions)}")
        
        # Log if we're storing non-standard prediction count
        if len(predictions) != 5:
            logger.info(f"Storing {len(predictions)} predictions for {model_name} (non-standard count)")
        
        # Generate unique set ID with microseconds for uniqueness
        timestamp = datetime_manager.format_for_database()
        set_id = f"{model_name.lower().replace(' ', '_')}_{datetime_manager.get_utc_timestamp().replace(':', '').replace('-', '').replace('T', '_').split('.')[0]}"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            try:
                # Mark previous prediction sets as not current
                cursor.execute('''
                    UPDATE prediction_sets 
                    SET is_current = FALSE 
                    WHERE model_name = ? AND is_current = TRUE
                ''', (model_name,))
                
                # Mark previous predictions as inactive
                cursor.execute('''
                    UPDATE model_predictions 
                    SET is_active = FALSE 
                    WHERE model_name = ? AND is_active = TRUE
                ''', (model_name,))
                
                # Insert new prediction set
                cursor.execute('''
                    INSERT INTO prediction_sets 
                    (set_id, model_name, created_at, is_current, total_predictions, training_duration, notes)
                    VALUES (?, ?, ?, TRUE, ?, ?, ?)
                ''', (set_id, model_name, timestamp, len(predictions), training_duration, notes))
                
                # Insert individual predictions
                for i, pred in enumerate(predictions):
                    prediction_id = f"{set_id}_pred_{i+1}"
                    
                    cursor.execute('''
                        INSERT INTO model_predictions
                        (model_name, prediction_id, prediction_set_id, white_numbers, powerball,
                         probability, features_used, hyperparameters, performance_metrics,
                         created_at, version, is_active)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, TRUE)
                    ''', (
                        model_name,
                        prediction_id,
                        set_id,
                        json.dumps(pred['white_numbers']),
                        pred['powerball'],
                        pred.get('probability', 0.0),
                        json.dumps(features_used),
                        json.dumps(hyperparameters),
                        json.dumps(performance_metrics),
                        timestamp
                    ))
                
                conn.commit()
                logger.info(f"Stored {len(predictions)} predictions for {model_name} (set: {set_id})")
                return set_id
                
            except Exception as e:
                conn.rollback()
                logger.error(f"Error storing predictions for {model_name}: {e}")
                raise
    
    def get_current_predictions(self, model_name: str) -> Optional[List[Dict[str, Any]]]:
        """
        Get the current active predictions for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of prediction dictionaries or None if no predictions exist
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT prediction_id, white_numbers, powerball, probability,
                       features_used, hyperparameters, performance_metrics, created_at
                FROM model_predictions
                WHERE model_name = ? AND is_active = TRUE
                ORDER BY prediction_id
            ''', (model_name,))
            
            rows = cursor.fetchall()
            if not rows:
                return None
            
            predictions = []
            for row in rows:
                pred_id, white_nums, powerball, prob, features, params, metrics, created = row
                
                # Convert bytes to proper types if needed
                if isinstance(powerball, bytes):
                    powerball = int.from_bytes(powerball, byteorder='little')
                
                # Ensure white_numbers are integers
                white_numbers = json.loads(white_nums)
                if isinstance(white_numbers, list):
                    white_numbers = [int(num) if isinstance(num, bytes) else int(num) for num in white_numbers]
                
                predictions.append({
                    'prediction_id': pred_id,
                    'white_numbers': white_numbers,
                    'powerball': int(powerball),
                    'probability': float(prob),
                    'features_used': json.loads(features),
                    'hyperparameters': json.loads(params),
                    'performance_metrics': json.loads(metrics),
                    'created_at': created
                })
            
            return predictions
    
    def get_all_current_predictions(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get current predictions for all models.
        
        Returns:
            Dictionary mapping model names to their current predictions
        """
        models = ['Ridge Regression', 'Random Forest', 'Gradient Boosting']
        all_predictions = {}
        
        for model in models:
            predictions = self.get_current_predictions(model)
            if predictions:
                all_predictions[model] = predictions
        
        return all_predictions
    
    def get_prediction_history(self, model_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get prediction history for a model.
        
        Args:
            model_name: Name of the model
            limit: Maximum number of prediction sets to return
            
        Returns:
            List of prediction set information
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT set_id, created_at, is_current, total_predictions, training_duration, notes
                FROM prediction_sets
                WHERE model_name = ?
                ORDER BY created_at DESC
                LIMIT ?
            ''', (model_name, limit))
            
            rows = cursor.fetchall()
            history = []
            
            for row in rows:
                set_id, created_at, is_current, total_preds, duration, notes = row
                history.append({
                    'set_id': set_id,
                    'created_at': created_at,
                    'is_current': bool(is_current),
                    'total_predictions': total_preds,
                    'training_duration': duration,
                    'notes': notes or ""
                })
            
            return history
    
    def store_predictions(self, model_name: str, predictions: List[Dict[str, Any]], 
                         features_used: List[str], hyperparameters: Dict[str, Any],
                         performance_metrics: Dict[str, float], training_duration: float = None,
                         notes: str = None) -> str:
        """
        Store predictions for a model (alias for store_model_predictions).
        
        Args:
            model_name: Name of the ML model
            predictions: List of prediction dictionaries
            features_used: List of feature names used in training
            hyperparameters: Model hyperparameters
            performance_metrics: Model performance metrics
            training_duration: Time taken to train the model
            notes: Optional notes about the training session
            
        Returns:
            Prediction set ID
        """
        return self.store_model_predictions(
            model_name, predictions, 
            features_used, hyperparameters, performance_metrics, 
            training_duration, notes
        )
    
    def get_predictions_by_model(self, model_name: str) -> Optional[List[Dict[str, Any]]]:
        """
        Get predictions by model name (alias for get_current_predictions).
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of prediction dictionaries or None if no predictions exist
        """
        return self.get_current_predictions(model_name)
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total predictions by model
            cursor.execute('''
                SELECT model_name, COUNT(*) as total,
                       SUM(CASE WHEN is_active THEN 1 ELSE 0 END) as active
                FROM model_predictions
                GROUP BY model_name
            ''')
            model_stats = {row[0]: {'total': row[1], 'active': row[2]} for row in cursor.fetchall()}
            
            # Total prediction sets
            cursor.execute('SELECT COUNT(*) FROM prediction_sets')
            total_sets = cursor.fetchone()[0]
            
            # Current prediction sets
            cursor.execute('SELECT COUNT(*) FROM prediction_sets WHERE is_current = TRUE')
            current_sets = cursor.fetchone()[0]
            
            # Database file size
            db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
            
            return {
                'model_statistics': model_stats,
                'total_prediction_sets': total_sets,
                'current_prediction_sets': current_sets,
                'database_size_bytes': db_size,
                'database_size_mb': round(db_size / (1024 * 1024), 2)
            }

class DatabaseMaintenanceManager:
    """
    Manages database maintenance tasks including data retention and integrity checks.
    """
    
    def __init__(self, prediction_manager: PersistentModelPredictionManager, 
                 retention_limit: int = 5):
        """
        Initialize maintenance manager.
        
        Args:
            prediction_manager: The prediction manager instance
            retention_limit: Number of prediction sets to keep per model (default: 5)
        """
        self.prediction_manager = prediction_manager
        self.retention_limit = retention_limit
        self.db_path = prediction_manager.db_path
        
    def run_data_integrity_checks(self) -> Dict[str, Any]:
        """
        Run comprehensive data integrity checks.
        
        Returns:
            Dictionary containing integrity check results
        """
        results = {
            'timestamp': datetime.datetime.now().isoformat(),
            'checks_performed': [],
            'issues_found': [],
            'recommendations': []
        }
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check 1: Verify each model has exactly one current prediction set
            results['checks_performed'].append('current_prediction_set_validation')
            cursor.execute('''
                SELECT model_name, COUNT(*) as current_count
                FROM prediction_sets
                WHERE is_current = TRUE
                GROUP BY model_name
                HAVING current_count != 1
            ''')
            
            multiple_current = cursor.fetchall()
            if multiple_current:
                for model, count in multiple_current:
                    results['issues_found'].append({
                        'type': 'multiple_current_sets',
                        'model': model,
                        'count': count,
                        'severity': 'high'
                    })
            
            # Check 2: Verify active predictions match current sets
            results['checks_performed'].append('active_prediction_consistency')
            cursor.execute('''
                SELECT p.model_name, COUNT(*) as active_predictions
                FROM model_predictions p
                WHERE p.is_active = TRUE
                GROUP BY p.model_name
                HAVING active_predictions != 5
            ''')
            
            inconsistent_active = cursor.fetchall()
            if inconsistent_active:
                for model, count in inconsistent_active:
                    results['issues_found'].append({
                        'type': 'inconsistent_active_predictions',
                        'model': model,
                        'active_count': count,
                        'expected': 5,
                        'severity': 'medium'
                    })
            
            # Check 3: Verify JSON data integrity
            results['checks_performed'].append('json_data_validation')
            cursor.execute('SELECT id, white_numbers, features_used, hyperparameters FROM model_predictions')
            
            for row_id, white_nums, features, params in cursor.fetchall():
                try:
                    json.loads(white_nums)
                    json.loads(features)
                    json.loads(params)
                except json.JSONDecodeError as e:
                    results['issues_found'].append({
                        'type': 'json_corruption',
                        'row_id': row_id,
                        'error': str(e),
                        'severity': 'high'
                    })
            
            # Generate recommendations
            if not results['issues_found']:
                results['recommendations'].append('Database integrity is excellent. No issues found.')
            else:
                high_severity = [issue for issue in results['issues_found'] if issue['severity'] == 'high']
                if high_severity:
                    results['recommendations'].append('High severity issues detected. Run resolve_integrity_issues() immediately.')
                else:
                    results['recommendations'].append('Minor issues detected. Consider running maintenance routine.')
        
        return results
    
    def apply_retention_policy(self) -> Dict[str, Any]:
        """
        Apply configurable retention policy to limit stored prediction sets.
        
        Returns:
            Dictionary containing retention policy results
        """
        results = {
            'timestamp': datetime.datetime.now().isoformat(),
            'retention_limit': self.retention_limit,
            'models_processed': [],
            'sets_removed': 0,
            'predictions_removed': 0
        }
        
        models = ['Ridge Regression', 'Random Forest', 'Gradient Boosting']
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for model in models:
                # Get all prediction sets for this model, ordered by creation time (newest first)
                cursor.execute('''
                    SELECT set_id, created_at
                    FROM prediction_sets
                    WHERE model_name = ?
                    ORDER BY created_at DESC
                ''', (model,))
                
                all_sets = cursor.fetchall()
                
                if len(all_sets) > self.retention_limit:
                    # Keep only the newest N sets
                    sets_to_keep = [row[0] for row in all_sets[:self.retention_limit]]
                    sets_to_remove = [row[0] for row in all_sets[self.retention_limit:]]
                    
                    if sets_to_remove:
                        # Remove old prediction sets and their predictions
                        placeholders = ','.join(['?' for _ in sets_to_remove])
                        
                        # Count predictions to be removed
                        cursor.execute(f'''
                            SELECT COUNT(*) FROM model_predictions
                            WHERE prediction_set_id IN ({placeholders})
                        ''', sets_to_remove)
                        predictions_to_remove = cursor.fetchone()[0]
                        
                        # Remove predictions
                        cursor.execute(f'''
                            DELETE FROM model_predictions
                            WHERE prediction_set_id IN ({placeholders})
                        ''', sets_to_remove)
                        
                        # Remove prediction sets
                        cursor.execute(f'''
                            DELETE FROM prediction_sets
                            WHERE set_id IN ({placeholders})
                        ''', sets_to_remove)
                        
                        results['models_processed'].append({
                            'model': model,
                            'sets_removed': len(sets_to_remove),
                            'predictions_removed': predictions_to_remove,
                            'sets_kept': len(sets_to_keep)
                        })
                        
                        results['sets_removed'] += len(sets_to_remove)
                        results['predictions_removed'] += predictions_to_remove
                        
                        logger.info(f"Removed {len(sets_to_remove)} old prediction sets for {model}")
                else:
                    results['models_processed'].append({
                        'model': model,
                        'sets_removed': 0,
                        'predictions_removed': 0,
                        'sets_kept': len(all_sets),
                        'note': 'Within retention limit'
                    })
            
            conn.commit()
        
        return results
    
    def resolve_integrity_issues(self) -> Dict[str, Any]:
        """
        Automatically resolve common integrity issues.
        
        Returns:
            Dictionary containing resolution results
        """
        results = {
            'timestamp': datetime.datetime.now().isoformat(),
            'issues_resolved': [],
            'manual_intervention_required': []
        }
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Fix multiple current prediction sets
            models = ['Ridge Regression', 'Random Forest', 'Gradient Boosting']
            
            for model in models:
                cursor.execute('''
                    SELECT set_id, created_at
                    FROM prediction_sets
                    WHERE model_name = ? AND is_current = TRUE
                    ORDER BY created_at DESC
                ''', (model,))
                
                current_sets = cursor.fetchall()
                
                if len(current_sets) > 1:
                    # Keep only the newest set as current
                    newest_set = current_sets[0][0]
                    older_sets = [row[0] for row in current_sets[1:]]
                    
                    # Mark older sets as not current
                    placeholders = ','.join(['?' for _ in older_sets])
                    cursor.execute(f'''
                        UPDATE prediction_sets
                        SET is_current = FALSE
                        WHERE set_id IN ({placeholders})
                    ''', older_sets)
                    
                    # Update corresponding predictions
                    cursor.execute(f'''
                        UPDATE model_predictions
                        SET is_active = FALSE
                        WHERE prediction_set_id IN ({placeholders})
                    ''', older_sets)
                    
                    results['issues_resolved'].append({
                        'type': 'multiple_current_sets_fixed',
                        'model': model,
                        'kept_current': newest_set,
                        'marked_inactive': len(older_sets)
                    })
            
            conn.commit()
        
        return results
    
    def get_maintenance_config(self) -> Dict[str, Any]:
        """Get current maintenance configuration."""
        return {
            'retention_limit': self.retention_limit,
            'database_path': self.db_path,
            'maintenance_features': [
                'Data integrity checks',
                'Configurable retention policy',
                'Automatic issue resolution',
                'Database statistics tracking'
            ]
        }
    
    def update_retention_limit(self, new_limit: int) -> bool:
        """
        Update the retention limit.
        
        Args:
            new_limit: New retention limit (minimum 1)
            
        Returns:
            True if updated successfully
        """
        if new_limit < 1:
            raise ValueError("Retention limit must be at least 1")
        
        self.retention_limit = new_limit
        logger.info(f"Updated retention limit to {new_limit}")
        return True

# Global instances for easy access
_prediction_manager = None
_maintenance_manager = None

def get_prediction_manager() -> PersistentModelPredictionManager:
    """Get the global prediction manager instance."""
    global _prediction_manager
    if _prediction_manager is None:
        _prediction_manager = PersistentModelPredictionManager()
    return _prediction_manager

def get_maintenance_manager() -> DatabaseMaintenanceManager:
    """Get the global maintenance manager instance."""
    global _maintenance_manager
    if _maintenance_manager is None:
        _maintenance_manager = DatabaseMaintenanceManager(get_prediction_manager())
    return _maintenance_manager