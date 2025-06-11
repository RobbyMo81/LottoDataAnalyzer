"""
Persistent Model Prediction Test Suite
=====================================
Comprehensive testing for the persistent model prediction system.
"""

import unittest
import tempfile
import os
import sqlite3
import json
from datetime import datetime
from core.persistent_model_predictions import (
    PersistentModelPredictionManager,
    DatabaseMaintenanceManager
)

class TestPersistentModelPredictions(unittest.TestCase):
    """Test suite for persistent model prediction system."""
    
    def setUp(self):
        """Set up test environment with temporary database."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_db_path = os.path.join(self.temp_dir, "test_predictions.db")
        self.manager = PersistentModelPredictionManager(self.test_db_path)
        self.maintenance = DatabaseMaintenanceManager(self.manager, retention_limit=3)
        
        # Sample prediction data
        self.sample_predictions = [
            {
                'white_numbers': [1, 11, 18, 30, 41],
                'powerball': 15,
                'probability': 0.0012345
            },
            {
                'white_numbers': [5, 12, 22, 35, 45],
                'powerball': 8,
                'probability': 0.0011234
            },
            {
                'white_numbers': [8, 15, 25, 38, 48],
                'powerball': 12,
                'probability': 0.0010123
            },
            {
                'white_numbers': [3, 9, 19, 29, 39],
                'powerball': 7,
                'probability': 0.0009012
            },
            {
                'white_numbers': [6, 16, 26, 36, 46],
                'powerball': 20,
                'probability': 0.0008901
            }
        ]
        
        self.sample_hyperparams = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42
        }
        
        self.sample_metrics = {
            'cv_mae_mean': 2.5,
            'cv_mae_std': 0.8,
            'cv_score_best': 1.9,
            'cv_score_worst': 3.2
        }
        
        self.sample_features = [
            'ordinal', 'pair_1_2', 'pair_3_4', 'sum_mean_5', 'dow_1'
        ]
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_database_initialization(self):
        """Test database tables are created correctly."""
        with sqlite3.connect(self.test_db_path) as conn:
            cursor = conn.cursor()
            
            # Check tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            self.assertIn('model_predictions', tables)
            self.assertIn('prediction_sets', tables)
    
    def test_store_model_predictions(self):
        """Test storing predictions for a model."""
        set_id = self.manager.store_model_predictions(
            model_name="Random Forest",
            predictions=self.sample_predictions,
            hyperparameters=self.sample_hyperparams,
            performance_metrics=self.sample_metrics,
            features_used=self.sample_features,
            training_duration=15.5,
            notes="Test prediction set"
        )
        
        self.assertIsNotNone(set_id)
        self.assertTrue(set_id.startswith("random_forest_"))
        
        # Verify predictions were stored
        predictions = self.manager.get_current_predictions("Random Forest")
        self.assertIsNotNone(predictions)
        self.assertEqual(len(predictions), 5)
        
        # Verify prediction data
        first_pred = predictions[0]
        self.assertEqual(first_pred['white_numbers'], [1, 11, 18, 30, 41])
        self.assertEqual(first_pred['powerball'], 15)
        self.assertEqual(first_pred['probability'], 0.0012345)
    
    def test_multiple_model_storage(self):
        """Test storing predictions for multiple models independently."""
        # Store for Random Forest
        rf_set_id = self.manager.store_model_predictions(
            model_name="Random Forest",
            predictions=self.sample_predictions,
            hyperparameters=self.sample_hyperparams,
            performance_metrics=self.sample_metrics,
            features_used=self.sample_features,
            training_duration=12.0
        )
        
        # Store for Ridge Regression
        ridge_predictions = [
            {
                'white_numbers': [2, 13, 24, 35, 47],
                'powerball': 18,
                'probability': 0.0015678
            },
            {
                'white_numbers': [7, 17, 27, 37, 49],
                'powerball': 5,
                'probability': 0.0014567
            },
            {
                'white_numbers': [4, 14, 28, 40, 50],
                'powerball': 11,
                'probability': 0.0013456
            },
            {
                'white_numbers': [9, 19, 31, 42, 52],
                'powerball': 22,
                'probability': 0.0012345
            },
            {
                'white_numbers': [10, 20, 32, 43, 53],
                'powerball': 14,
                'probability': 0.0011234
            }
        ]
        
        ridge_set_id = self.manager.store_model_predictions(
            model_name="Ridge Regression",
            predictions=ridge_predictions,
            hyperparameters={'alpha': 1.0},
            performance_metrics={'cv_mae_mean': 3.2},
            features_used=self.sample_features,
            training_duration=8.0
        )
        
        # Verify both models have predictions
        rf_preds = self.manager.get_current_predictions("Random Forest")
        ridge_preds = self.manager.get_current_predictions("Ridge Regression")
        
        self.assertIsNotNone(rf_preds)
        self.assertIsNotNone(ridge_preds)
        self.assertEqual(len(rf_preds), 5)
        self.assertEqual(len(ridge_preds), 5)
        
        # Verify predictions are different
        self.assertNotEqual(rf_preds[0]['white_numbers'], ridge_preds[0]['white_numbers'])
    
    def test_model_prediction_update(self):
        """Test updating predictions for a specific model doesn't affect others."""
        # Store initial predictions for both models
        self.manager.store_model_predictions(
            model_name="Random Forest",
            predictions=self.sample_predictions,
            hyperparameters=self.sample_hyperparams,
            performance_metrics=self.sample_metrics,
            features_used=self.sample_features
        )
        
        self.manager.store_model_predictions(
            model_name="Gradient Boosting",
            predictions=self.sample_predictions,
            hyperparameters={'n_estimators': 200},
            performance_metrics=self.sample_metrics,
            features_used=self.sample_features
        )
        
        # Get initial predictions
        initial_rf = self.manager.get_current_predictions("Random Forest")
        initial_gb = self.manager.get_current_predictions("Gradient Boosting")
        
        # Update Random Forest predictions
        new_predictions = [
            {
                'white_numbers': [10, 20, 30, 40, 50],
                'powerball': 25,
                'probability': 0.002
            },
            {
                'white_numbers': [11, 21, 31, 41, 51],
                'powerball': 24,
                'probability': 0.0019
            },
            {
                'white_numbers': [12, 22, 32, 42, 52],
                'powerball': 23,
                'probability': 0.0018
            },
            {
                'white_numbers': [13, 23, 33, 43, 53],
                'powerball': 22,
                'probability': 0.0017
            },
            {
                'white_numbers': [14, 24, 34, 44, 54],
                'powerball': 21,
                'probability': 0.0016
            }
        ]
        
        self.manager.store_model_predictions(
            model_name="Random Forest",
            predictions=new_predictions,
            hyperparameters=self.sample_hyperparams,
            performance_metrics=self.sample_metrics,
            features_used=self.sample_features
        )
        
        # Verify Random Forest predictions updated
        updated_rf = self.manager.get_current_predictions("Random Forest")
        self.assertEqual(updated_rf[0]['white_numbers'], [10, 20, 30, 40, 50])
        
        # Verify Gradient Boosting predictions unchanged
        unchanged_gb = self.manager.get_current_predictions("Gradient Boosting")
        self.assertEqual(initial_gb[0]['white_numbers'], unchanged_gb[0]['white_numbers'])
    
    def test_get_all_current_predictions(self):
        """Test retrieving current predictions for all models."""
        # Store predictions for multiple models
        models = ["Random Forest", "Ridge Regression", "Gradient Boosting"]
        
        for model in models:
            self.manager.store_model_predictions(
                model_name=model,
                predictions=self.sample_predictions,
                hyperparameters=self.sample_hyperparams,
                performance_metrics=self.sample_metrics,
                features_used=self.sample_features
            )
        
        all_predictions = self.manager.get_all_current_predictions()
        
        self.assertEqual(len(all_predictions), 3)
        for model in models:
            self.assertIn(model, all_predictions)
            self.assertEqual(len(all_predictions[model]), 5)
    
    def test_prediction_history(self):
        """Test prediction history tracking."""
        model_name = "Random Forest"
        
        # Store multiple prediction sets
        for i in range(3):
            modified_predictions = [
                {
                    'white_numbers': [i+1, i+11, i+18, i+30, i+41],
                    'powerball': i+15,
                    'probability': 0.001 + i*0.0001
                }
                for _ in range(5)
            ]
            
            self.manager.store_model_predictions(
                model_name=model_name,
                predictions=modified_predictions,
                hyperparameters=self.sample_hyperparams,
                performance_metrics=self.sample_metrics,
                features_used=self.sample_features,
                notes=f"Training round {i+1}"
            )
        
        history = self.manager.get_prediction_history(model_name, limit=10)
        self.assertEqual(len(history), 3)
        
        # Verify only the latest is current
        current_count = sum(1 for h in history if h['is_current'])
        self.assertEqual(current_count, 1)
    
    def test_database_maintenance_integrity_checks(self):
        """Test database integrity checking."""
        # Store some predictions
        self.manager.store_model_predictions(
            model_name="Random Forest",
            predictions=self.sample_predictions,
            hyperparameters=self.sample_hyperparams,
            performance_metrics=self.sample_metrics,
            features_used=self.sample_features
        )
        
        # Run integrity checks
        results = self.maintenance.run_data_integrity_checks()
        
        self.assertIn('timestamp', results)
        self.assertIn('checks_performed', results)
        self.assertIn('issues_found', results)
        self.assertIn('recommendations', results)
        
        # Should have no issues with clean data
        self.assertEqual(len(results['issues_found']), 0)
    
    def test_retention_policy(self):
        """Test configurable retention policy."""
        model_name = "Random Forest"
        
        # Store more prediction sets than retention limit
        for i in range(5):  # retention limit is 3
            modified_predictions = [
                {
                    'white_numbers': [i+1, i+11, i+18, i+30, i+41],
                    'powerball': i+15,
                    'probability': 0.001 + i*0.0001
                }
                for _ in range(5)
            ]
            
            self.manager.store_model_predictions(
                model_name=model_name,
                predictions=modified_predictions,
                hyperparameters=self.sample_hyperparams,
                performance_metrics=self.sample_metrics,
                features_used=self.sample_features
            )
        
        # Apply retention policy
        results = self.maintenance.apply_retention_policy()
        
        # Should have removed 2 sets (5 - 3 = 2)
        self.assertEqual(results['sets_removed'], 2)
        
        # Verify only 3 sets remain
        history = self.manager.get_prediction_history(model_name, limit=10)
        self.assertEqual(len(history), 3)
    
    def test_database_statistics(self):
        """Test database statistics generation."""
        # Store predictions for multiple models
        models = ["Random Forest", "Ridge Regression", "Gradient Boosting"]
        
        for model in models:
            self.manager.store_model_predictions(
                model_name=model,
                predictions=self.sample_predictions,
                hyperparameters=self.sample_hyperparams,
                performance_metrics=self.sample_metrics,
                features_used=self.sample_features
            )
        
        stats = self.manager.get_database_stats()
        
        self.assertIn('model_statistics', stats)
        self.assertIn('total_prediction_sets', stats)
        self.assertIn('current_prediction_sets', stats)
        self.assertIn('database_size_bytes', stats)
        
        # Should have 3 models with 5 predictions each
        self.assertEqual(stats['total_prediction_sets'], 3)
        self.assertEqual(stats['current_prediction_sets'], 3)
        
        for model in models:
            self.assertIn(model, stats['model_statistics'])
            self.assertEqual(stats['model_statistics'][model]['active'], 5)
            self.assertEqual(stats['model_statistics'][model]['total'], 5)

def run_comprehensive_tests():
    """Run all tests and return results summary."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestPersistentModelPredictions)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, 'w'))
    result = runner.run(suite)
    
    # Return summary
    return {
        'tests_run': result.testsRun,
        'failures': len(result.failures),
        'errors': len(result.errors),
        'success': result.wasSuccessful(),
        'failure_details': result.failures + result.errors
    }

if __name__ == '__main__':
    unittest.main()