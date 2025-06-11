#!/usr/bin/env python3
"""
Legacy Prediction Data Migration Script
--------------------------------------
Migrates all prediction data from joblib files to SQLite database.
This script handles the complete consolidation from legacy storage to modern persistence.
"""

import os
import sys
import json
import joblib
import datetime
import logging
import shutil
from typing import Dict, List, Any
import numpy as np

# Add the project root to Python path
sys.path.insert(0, '.')

from core.persistent_model_predictions import get_prediction_manager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LegacyDataMigrator:
    """Handles migration from joblib to SQLite storage."""
    
    def __init__(self):
        self.prediction_manager = get_prediction_manager()
        self.legacy_files = {
            'history': 'data/prediction_history.joblib',
            'models': 'data/prediction_models.joblib'
        }
        self.backup_dir = f'data/legacy_backup_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
        
    def safe_json_convert(self, obj):
        """Convert numpy types and other objects to JSON-serializable format."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self.safe_json_convert(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.safe_json_convert(item) for item in obj]
        else:
            return obj
    
    def create_backup(self):
        """Create backup of all legacy files before migration."""
        try:
            os.makedirs(self.backup_dir, exist_ok=True)
            
            for file_type, file_path in self.legacy_files.items():
                if os.path.exists(file_path):
                    backup_path = os.path.join(self.backup_dir, os.path.basename(file_path))
                    shutil.copy2(file_path, backup_path)
                    logger.info(f"Backed up {file_path} to {backup_path}")
            
            # Also backup any ML memory models
            ml_memory_dir = 'data/ml_memory'
            if os.path.exists(ml_memory_dir):
                backup_ml_dir = os.path.join(self.backup_dir, 'ml_memory')
                shutil.copytree(ml_memory_dir, backup_ml_dir)
                logger.info(f"Backed up ML memory directory to {backup_ml_dir}")
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return False
    
    def load_legacy_data(self):
        """Load all legacy prediction data from joblib files."""
        legacy_data = {}
        
        # Load prediction history
        history_path = self.legacy_files['history']
        if os.path.exists(history_path):
            try:
                history = joblib.load(history_path)
                legacy_data['history'] = self.safe_json_convert(history)
                logger.info(f"Loaded {len(history.get('predictions', []))} predictions from legacy history")
            except Exception as e:
                logger.error(f"Failed to load prediction history: {e}")
                legacy_data['history'] = None
        
        # Load prediction models
        models_path = self.legacy_files['models']
        if os.path.exists(models_path):
            try:
                models = joblib.load(models_path)
                legacy_data['models'] = models  # Keep models as-is for now
                logger.info(f"Loaded {len(models)} trained models from legacy storage")
            except Exception as e:
                logger.error(f"Failed to load prediction models: {e}")
                legacy_data['models'] = None
        
        return legacy_data
    
    def migrate_predictions(self, legacy_history: Dict) -> int:
        """Migrate historical predictions to SQLite database."""
        if not legacy_history or 'predictions' not in legacy_history:
            logger.info("No legacy predictions to migrate")
            return 0
        
        migrated_count = 0
        predictions = legacy_history['predictions']
        
        for i, prediction in enumerate(predictions):
            try:
                # Extract prediction data
                white_numbers = prediction.get('white_numbers', [])
                powerball = prediction.get('powerball', 1)
                probability = prediction.get('probability', 0.0)
                timestamp = prediction.get('timestamp', datetime.datetime.now().isoformat())
                prediction_date = prediction.get('prediction_for_date', 
                                                datetime.datetime.now().strftime('%Y-%m-%d'))
                
                # Convert timestamp to proper format
                if isinstance(timestamp, str):
                    try:
                        parsed_time = datetime.datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        created_at = parsed_time.strftime('%Y-%m-%d %H:%M:%S')
                    except:
                        created_at = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                else:
                    created_at = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # Create prediction set ID
                set_id = f"legacy_migration_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{i:03d}"
                
                # Store prediction set metadata
                set_metadata = {
                    'set_id': set_id,
                    'model_name': 'Legacy_Prediction_System',
                    'created_at': created_at,
                    'is_current': False,  # Legacy data is not current
                    'total_predictions': 1,
                    'training_duration': 0.0,
                    'notes': f'Migrated from legacy joblib storage - original timestamp: {timestamp}'
                }
                
                # Store prediction data
                prediction_data = {
                    'model_name': 'Legacy_Prediction_System',
                    'prediction_id': f"legacy_{i:03d}",
                    'prediction_set_id': set_id,
                    'white_numbers': json.dumps(white_numbers),
                    'powerball': int(powerball),
                    'probability': float(probability),
                    'features_used': json.dumps(list(prediction.get('tool_contributions', {}).keys())),
                    'hyperparameters': json.dumps(prediction.get('tool_contributions', {})),
                    'performance_metrics': json.dumps({
                        'sources': prediction.get('sources', {}),
                        'storage_version': prediction.get('storage_version', '1.0')
                    }),
                    'created_at': created_at,
                    'version': 1,
                    'is_active': False  # Legacy predictions are not active
                }
                
                # Insert into database
                self.prediction_manager._execute_query("""
                    INSERT OR REPLACE INTO prediction_sets 
                    (set_id, model_name, created_at, is_current, total_predictions, training_duration, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (set_metadata['set_id'], set_metadata['model_name'], set_metadata['created_at'],
                     set_metadata['is_current'], set_metadata['total_predictions'], 
                     set_metadata['training_duration'], set_metadata['notes']))
                
                self.prediction_manager._execute_query("""
                    INSERT OR REPLACE INTO model_predictions 
                    (model_name, prediction_id, prediction_set_id, white_numbers, powerball, 
                     probability, features_used, hyperparameters, performance_metrics, 
                     created_at, version, is_active)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (prediction_data['model_name'], prediction_data['prediction_id'],
                     prediction_data['prediction_set_id'], prediction_data['white_numbers'],
                     prediction_data['powerball'], prediction_data['probability'],
                     prediction_data['features_used'], prediction_data['hyperparameters'],
                     prediction_data['performance_metrics'], prediction_data['created_at'],
                     prediction_data['version'], prediction_data['is_active']))
                
                migrated_count += 1
                logger.info(f"Migrated prediction {i+1}/{len(predictions)}")
                
            except Exception as e:
                logger.error(f"Failed to migrate prediction {i}: {e}")
                continue
        
        return migrated_count
    
    def migrate_accuracy_data(self, legacy_history: Dict) -> int:
        """Store accuracy data as metadata in SQLite."""
        if not legacy_history or 'accuracy' not in legacy_history:
            return 0
        
        try:
            accuracy_data = legacy_history['accuracy']
            feedback_data = legacy_history.get('feedback', [])
            
            # Store as a special metadata entry
            metadata_entry = {
                'accuracy_history': accuracy_data,
                'feedback_history': feedback_data,
                'migration_timestamp': datetime.datetime.now().isoformat(),
                'original_count': len(accuracy_data)
            }
            
            # Create a special entry in prediction_sets for metadata
            set_id = f"legacy_metadata_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            self.prediction_manager._execute_query("""
                INSERT OR REPLACE INTO prediction_sets 
                (set_id, model_name, created_at, is_current, total_predictions, training_duration, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (set_id, 'Legacy_Accuracy_Metadata', 
                 datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                 False, 0, 0.0, json.dumps(metadata_entry)))
            
            logger.info(f"Migrated {len(accuracy_data)} accuracy entries and {len(feedback_data)} feedback entries")
            return len(accuracy_data)
            
        except Exception as e:
            logger.error(f"Failed to migrate accuracy data: {e}")
            return 0
    
    def cleanup_legacy_files(self, preserve_models: bool = True):
        """Remove or archive legacy joblib files after successful migration."""
        try:
            # Remove prediction history file
            history_path = self.legacy_files['history']
            if os.path.exists(history_path):
                os.remove(history_path)
                logger.info(f"Removed legacy prediction history: {history_path}")
            
            # Optionally preserve models for now (they might still be needed)
            if not preserve_models:
                models_path = self.legacy_files['models']
                if os.path.exists(models_path):
                    os.remove(models_path)
                    logger.info(f"Removed legacy prediction models: {models_path}")
            else:
                logger.info("Preserved legacy model files for compatibility")
                
        except Exception as e:
            logger.error(f"Failed to cleanup legacy files: {e}")
    
    def run_migration(self) -> Dict[str, Any]:
        """Execute the complete migration process."""
        logger.info("Starting legacy prediction data migration...")
        
        # Create backup
        if not self.create_backup():
            return {'success': False, 'error': 'Failed to create backup'}
        
        # Load legacy data
        legacy_data = self.load_legacy_data()
        
        # Migrate predictions
        predictions_migrated = 0
        if legacy_data.get('history'):
            predictions_migrated = self.migrate_predictions(legacy_data['history'])
            accuracy_migrated = self.migrate_accuracy_data(legacy_data['history'])
        
        # Migration summary
        result = {
            'success': True,
            'backup_location': self.backup_dir,
            'predictions_migrated': predictions_migrated,
            'accuracy_entries_migrated': accuracy_migrated if 'accuracy_migrated' in locals() else 0,
            'models_preserved': True,  # We're preserving models for now
            'migration_timestamp': datetime.datetime.now().isoformat()
        }
        
        # Cleanup legacy files (preserve models for compatibility)
        self.cleanup_legacy_files(preserve_models=True)
        
        logger.info(f"Migration completed: {result}")
        return result

def main():
    """Main migration execution."""
    print("=" * 60)
    print("LEGACY PREDICTION DATA MIGRATION")
    print("=" * 60)
    
    migrator = LegacyDataMigrator()
    result = migrator.run_migration()
    
    if result['success']:
        print(f"\n✓ Migration completed successfully!")
        print(f"✓ Backup created at: {result['backup_location']}")
        print(f"✓ Predictions migrated: {result['predictions_migrated']}")
        print(f"✓ Accuracy entries migrated: {result['accuracy_entries_migrated']}")
        print(f"✓ Legacy models preserved for compatibility")
    else:
        print(f"\n✗ Migration failed: {result.get('error', 'Unknown error')}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())