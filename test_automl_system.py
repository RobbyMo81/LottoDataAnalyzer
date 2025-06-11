"""
Comprehensive AutoML System Test & Verification
----------------------------------------------
Full system test to identify and resolve all compatibility issues.
"""

import sys
import traceback
from pathlib import Path

# Add core to path
sys.path.append('core')

def test_json_serialization():
    """Test JSON serialization utilities with problematic data types."""
    print("Testing JSON serialization utilities...")
    
    from core.json_serialization_utils import safe_json_dumps, sanitize_config_for_json
    import numpy as np
    
    # Test data with problematic types
    test_data = {
        "boolean_val": True,
        "numpy_bool": np.bool_(True),
        "numpy_int": np.int64(42),
        "numpy_float": np.float64(3.14),
        "normal_int": 42,
        "normal_float": 3.14,
        "string": "test",
        "list": [1, 2, 3],
        "nested_dict": {
            "inner_bool": False,
            "inner_numpy": np.bool_(False)
        }
    }
    
    try:
        # Test sanitization
        sanitized = sanitize_config_for_json(test_data)
        print(f"✓ Sanitization successful: {type(sanitized)}")
        
        # Test safe serialization
        json_str = safe_json_dumps(sanitized)
        print(f"✓ JSON serialization successful: {len(json_str)} characters")
        
        return True
    except Exception as e:
        print(f"✗ JSON serialization test failed: {e}")
        traceback.print_exc()
        return False

def test_experiment_tracker():
    """Test experiment tracker with realistic configuration."""
    print("\nTesting experiment tracker...")
    
    try:
        from core.experiment_tracker import ExperimentTracker
        
        # Initialize tracker
        tracker = ExperimentTracker()
        
        # Test configuration with boolean values (the problematic case)
        test_config = {
            "model_name": "RandomForest",
            "n_estimators": 100,
            "max_depth": 10,
            "fit_intercept": True,  # This was causing the error
            "random_state": 42,
            "parameter_spaces": {
                "n_estimators": [50, 100, 200],
                "fit_intercept": [True, False]  # Boolean values
            }
        }
        
        # Start experiment job with unique ID
        import time
        unique_job_id = f"test_job_{int(time.time())}"
        job_id = tracker.start_experiment(
            experiment_name="test_experiment",
            job_id=unique_job_id,
            config=test_config
        )
        
        print(f"✓ Experiment tracker test successful: {job_id}")
        return True
        
    except Exception as e:
        print(f"✗ Experiment tracker test failed: {e}")
        traceback.print_exc()
        return False

def test_tuning_wizard():
    """Test tuning wizard with boolean parameters."""
    print("\nTesting tuning wizard...")
    
    try:
        from core.ml_tuning_simplified import SimplifiedTuningWizard
        from core.experiment_tracker import get_experiment_tracker
        import pandas as pd
        import numpy as np
        
        # Create sample data
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        
        # Initialize wizard
        tracker = get_experiment_tracker()
        wizard = SimplifiedTuningWizard(experiment_tracker=tracker)
        
        # Test configuration with problematic boolean values
        config = {
            "model_name": "Ridge",
            "search_strategy": "random_search",
            "n_trials": 3,
            "parameter_spaces": None,  # Use default grid with booleans
            "random_state": 42
        }
        
        # Run tuning
        result = wizard.run_tuning(config, X, y)
        
        print(f"✓ Tuning wizard test successful: {result.best_trial.score}")
        return True
        
    except Exception as e:
        print(f"✗ Tuning wizard test failed: {e}")
        traceback.print_exc()
        return False

def test_automl_interface():
    """Test the AutoML interface integration."""
    print("\nTesting AutoML interface...")
    
    try:
        # This would test the Streamlit interface, but we'll simulate the data flow
        from core.automl_simple import prepare_ml_data
        from core import storage
        
        # Get some data
        store = storage._Store()
        df = store.latest()
        
        if df is not None and len(df) > 0:
            X, y, feature_cols = prepare_ml_data(df)
            print(f"✓ Data preparation successful: {X.shape}, {len(feature_cols)} features")
            return True
        else:
            print("✗ No data available for testing")
            return False
            
    except Exception as e:
        print(f"✗ AutoML interface test failed: {e}")
        traceback.print_exc()
        return False

def run_comprehensive_test():
    """Run all system tests."""
    print("=" * 60)
    print("COMPREHENSIVE AUTOML SYSTEM TEST")
    print("=" * 60)
    
    tests = [
        test_json_serialization,
        test_experiment_tracker,
        test_tuning_wizard,
        test_automl_interface
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} {test.__name__}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready.")
    else:
        print("⚠️  Some tests failed. System needs fixes.")
    
    return passed == total

if __name__ == "__main__":
    run_comprehensive_test()