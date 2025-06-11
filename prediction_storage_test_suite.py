"""
Comprehensive Test Suite for Prediction Storage Refactoring
----------------------------------------------------------
Tests all scenarios for one-prediction-per-draw-date enforcement.
"""

import unittest
import tempfile
import os
import datetime
from unittest.mock import patch
import pandas as pd
import joblib

from .prediction_storage_refactor import PredictionStorageManager

class TestPredictionStorageRefactor(unittest.TestCase):
    """Test suite for prediction storage refactoring."""
    
    def setUp(self):
        """Set up test environment with temporary files."""
        self.test_dir = tempfile.mkdtemp()
        self.test_history_path = os.path.join(self.test_dir, "test_prediction_history.joblib")
        self.manager = PredictionStorageManager(self.test_history_path)
        
        # Sample prediction data
        self.sample_prediction = {
            'white_numbers': [1, 11, 18, 30, 41],
            'powerball': 14,
            'probability': 0.00001,
            'tool_contributions': {'frequency': {'white_numbers': [1, 11], 'powerball': 14}},
            'sources': {1: ['frequency'], 11: ['frequency']}
        }
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_valid_draw_date_validation(self):
        """Test validation of Powerball draw dates (Monday, Wednesday, Saturday)."""
        # Test valid dates
        self.assertTrue(self.manager._is_valid_draw_date('2025-06-02'))  # Monday
        self.assertTrue(self.manager._is_valid_draw_date('2025-06-04'))  # Wednesday  
        self.assertTrue(self.manager._is_valid_draw_date('2025-06-07'))  # Saturday
        
        # Test invalid dates
        self.assertFalse(self.manager._is_valid_draw_date('2025-06-03'))  # Tuesday
        self.assertFalse(self.manager._is_valid_draw_date('2025-06-05'))  # Thursday
        self.assertFalse(self.manager._is_valid_draw_date('2025-06-06'))  # Friday
        self.assertFalse(self.manager._is_valid_draw_date('2025-06-01'))  # Sunday
    
    def test_next_draw_date_calculation(self):
        """Test calculation of next valid draw dates."""
        # Test from Monday (should get Wednesday)
        monday = datetime.date(2025, 6, 2)
        next_draw = self.manager._get_next_draw_date(monday)
        self.assertEqual(next_draw, '2025-06-04')  # Wednesday
        
        # Test from Tuesday (should get Wednesday)
        tuesday = datetime.date(2025, 6, 3)
        next_draw = self.manager._get_next_draw_date(tuesday)
        self.assertEqual(next_draw, '2025-06-04')  # Wednesday
        
        # Test from Wednesday (should get Saturday)
        wednesday = datetime.date(2025, 6, 4)
        next_draw = self.manager._get_next_draw_date(wednesday)
        self.assertEqual(next_draw, '2025-06-07')  # Saturday
        
        # Test from Sunday (should get Monday)
        sunday = datetime.date(2025, 6, 1)
        next_draw = self.manager._get_next_draw_date(sunday)
        self.assertEqual(next_draw, '2025-06-02')  # Monday
    
    def test_first_prediction_storage(self):
        """Test storing the first prediction for a future draw date."""
        target_date = '2025-06-04'  # Wednesday
        result = self.manager.store_prediction(self.sample_prediction, target_date)
        
        self.assertTrue(result)
        stored_pred = self.manager.get_predictions_by_date(target_date)
        self.assertIsNotNone(stored_pred)
        self.assertEqual(stored_pred['white_numbers'], [1, 11, 18, 30, 41])
        self.assertEqual(stored_pred['powerball'], 14)
        self.assertEqual(stored_pred['prediction_for_date'], target_date)
    
    def test_prediction_overwrite(self):
        """Test that second prediction for same date overwrites the first."""
        target_date = '2025-06-04'  # Wednesday
        
        # Store first prediction
        first_prediction = self.sample_prediction.copy()
        first_prediction['white_numbers'] = [1, 2, 3, 4, 5]
        first_prediction['powerball'] = 10
        
        result1 = self.manager.store_prediction(first_prediction, target_date)
        self.assertTrue(result1)
        
        # Store second prediction for same date
        second_prediction = self.sample_prediction.copy()
        second_prediction['white_numbers'] = [10, 20, 30, 40, 50]
        second_prediction['powerball'] = 20
        
        result2 = self.manager.store_prediction(second_prediction, target_date)
        self.assertTrue(result2)
        
        # Verify only the second prediction is stored
        stored_pred = self.manager.get_predictions_by_date(target_date)
        self.assertEqual(stored_pred['white_numbers'], [10, 20, 30, 40, 50])
        self.assertEqual(stored_pred['powerball'], 20)
        
        # Verify only one prediction total
        predictions = self.manager.prediction_history['predictions']
        predictions_for_date = [p for p in predictions if p.get('prediction_for_date') == target_date]
        self.assertEqual(len(predictions_for_date), 1)
    
    def test_multiple_distinct_dates(self):
        """Test storing predictions for multiple distinct draw dates."""
        dates = ['2025-06-09', '2025-06-11', '2025-06-14']  # Mon, Wed, Sat
        
        for i, date in enumerate(dates):
            prediction = self.sample_prediction.copy()
            prediction['white_numbers'] = [i+1, i+2, i+3, i+4, i+5]
            prediction['powerball'] = i + 10
            
            result = self.manager.store_prediction(prediction, date)
            self.assertTrue(result)
        
        # Verify all predictions are stored
        for i, date in enumerate(dates):
            stored_pred = self.manager.get_predictions_by_date(date)
            self.assertIsNotNone(stored_pred)
            self.assertEqual(stored_pred['white_numbers'], [i+1, i+2, i+3, i+4, i+5])
            self.assertEqual(stored_pred['powerball'], i + 10)
        
        # Verify total count
        self.assertEqual(len(self.manager.prediction_history['predictions']), 3)
    
    def test_invalid_date_handling(self):
        """Test behavior when predicting for non-draw days."""
        invalid_date = '2025-06-03'  # Tuesday
        result = self.manager.store_prediction(self.sample_prediction, invalid_date)
        
        # Should succeed but store for next valid draw date
        self.assertTrue(result)
        
        # Should be stored for the next valid draw date (Saturday, since Tuesday is past)
        stored_pred = self.manager.get_predictions_by_date('2025-06-07')
        self.assertIsNotNone(stored_pred)
        
        # Should not be stored for the invalid date
        invalid_stored = self.manager.get_predictions_by_date(invalid_date)
        self.assertIsNone(invalid_stored)
    
    def test_prediction_after_draw_date(self):
        """Test system behavior for predictions made after draw date."""
        # Use a past date (should be automatically adjusted to next valid draw)
        past_date = '2025-05-31'  # Past Saturday
        
        with patch('core.prediction_storage_refactor.datetime') as mock_datetime:
            # Mock current date as June 3, 2025 (Tuesday)
            mock_datetime.datetime.now.return_value = datetime.datetime(2025, 6, 3)
            mock_datetime.datetime.fromisoformat.side_effect = datetime.datetime.fromisoformat
            mock_datetime.timedelta = datetime.timedelta
            mock_datetime.date = datetime.date
            
            result = self.manager.store_prediction(self.sample_prediction, past_date)
            self.assertTrue(result)  # Should succeed by adjusting to next valid date
            
            # Should NOT be stored for the past date
            stored_pred = self.manager.get_predictions_by_date(past_date)
            self.assertIsNone(stored_pred)
            
            # Should be stored for the next valid draw date instead
            next_valid_date = '2025-06-04'  # Next Wednesday after June 3
            stored_pred_next = self.manager.get_predictions_by_date(next_valid_date)
            self.assertIsNotNone(stored_pred_next)
    
    def test_duplicate_analysis(self):
        """Test analysis of duplicate predictions."""
        # Create test data with duplicates
        target_date = '2025-06-04'
        
        # Add multiple predictions manually to test analysis
        predictions = []
        for i in range(3):
            pred = self.sample_prediction.copy()
            pred['prediction_for_date'] = target_date
            pred['timestamp'] = f'2025-06-03T1{i}:00:00'
            predictions.append(pred)
        
        self.manager.prediction_history['predictions'] = predictions
        
        analysis = self.manager.analyze_duplicate_predictions()
        
        self.assertEqual(analysis['total_predictions'], 3)
        self.assertEqual(analysis['unique_dates'], 1)
        self.assertEqual(analysis['duplicate_dates'], 1)
        self.assertEqual(analysis['total_duplicate_entries'], 2)
        self.assertIn(target_date, analysis['duplicates'])
    
    def test_migration_keep_latest(self):
        """Test migration strategy to keep latest predictions."""
        target_date = '2025-06-04'
        
        # Create predictions with different timestamps
        predictions = []
        for i in range(3):
            pred = self.sample_prediction.copy()
            pred['prediction_for_date'] = target_date
            pred['timestamp'] = f'2025-06-03T1{i}:00:00'
            pred['powerball'] = 10 + i  # Different powerball for identification
            predictions.append(pred)
        
        self.manager.prediction_history['predictions'] = predictions
        
        # Perform migration
        result = self.manager.migrate_legacy_data('keep_latest')
        
        self.assertEqual(result['status'], 'migration_completed')
        self.assertEqual(result['migrated_count'], 1)
        self.assertEqual(result['removed_count'], 2)
        
        # Verify latest prediction is kept
        stored_pred = self.manager.get_predictions_by_date(target_date)
        self.assertEqual(stored_pred['powerball'], 12)  # Latest one
    
    def test_migration_keep_highest_prob(self):
        """Test migration strategy to keep highest probability predictions."""
        target_date = '2025-06-04'
        
        # Create predictions with different probabilities
        predictions = []
        probs = [0.00001, 0.00003, 0.00002]
        for i, prob in enumerate(probs):
            pred = self.sample_prediction.copy()
            pred['prediction_for_date'] = target_date
            pred['probability'] = prob
            pred['powerball'] = 10 + i
            predictions.append(pred)
        
        self.manager.prediction_history['predictions'] = predictions
        
        # Perform migration
        result = self.manager.migrate_legacy_data('keep_highest_prob')
        
        self.assertEqual(result['status'], 'migration_completed')
        
        # Verify highest probability prediction is kept
        stored_pred = self.manager.get_predictions_by_date(target_date)
        self.assertEqual(stored_pred['powerball'], 11)  # Index 1 has highest prob
    
    def test_prediction_history_dataframe(self):
        """Test generation of prediction history DataFrame."""
        # Add test predictions using future valid draw dates
        dates = ['2025-06-09', '2025-06-11']  # Monday and Wednesday
        for i, date in enumerate(dates):
            pred = self.sample_prediction.copy()
            pred['prediction_for_date'] = date
            pred['powerball'] = 10 + i
            self.manager.store_prediction(pred, date)
        
        df = self.manager.get_prediction_history_dataframe()
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertIn('Date', df.columns)
        self.assertIn('White Numbers', df.columns)
        self.assertIn('Powerball', df.columns)
        self.assertIn('Valid Draw Date', df.columns)
        
        # Verify all entries are valid draw dates
        self.assertTrue(all(df['Valid Draw Date'] == 'Yes'))
    
    def test_system_integrity_validation(self):
        """Test comprehensive system integrity validation."""
        # Add valid predictions
        self.manager.store_prediction(self.sample_prediction, '2025-06-04')
        
        validation = self.manager.validate_system_integrity()
        
        self.assertEqual(validation['total_predictions'], 1)
        self.assertEqual(validation['unique_dates'], 1)
        self.assertFalse(validation['has_duplicates'])
        self.assertEqual(validation['invalid_draw_dates'], 0)
        self.assertEqual(validation['incomplete_predictions'], 0)
        self.assertTrue(validation['system_healthy'])

def run_comprehensive_tests():
    """Run all tests and return results."""
    import io
    import sys
    
    # Capture test output
    test_output = io.StringIO()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestPredictionStorageRefactor)
    
    # Run tests
    runner = unittest.TextTestRunner(stream=test_output, verbosity=2)
    result = runner.run(suite)
    
    # Get output
    output = test_output.getvalue()
    
    return {
        'tests_run': result.testsRun,
        'failures': len(result.failures),
        'errors': len(result.errors),
        'success': result.wasSuccessful(),
        'output': output,
        'failure_details': result.failures,
        'error_details': result.errors
    }

if __name__ == '__main__':
    results = run_comprehensive_tests()
    print(f"Tests run: {results['tests_run']}")
    print(f"Failures: {results['failures']}")
    print(f"Errors: {results['errors']}")
    print(f"Success: {results['success']}")
    if results['failures'] or results['errors']:
        print("\nDetails:")
        print(results['output'])