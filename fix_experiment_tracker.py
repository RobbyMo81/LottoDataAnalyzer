"""
Fix Experiment Tracker JSON Serialization
-----------------------------------------
Applies safe JSON serialization to all instances in experiment tracker.
"""

import re

def fix_experiment_tracker():
    """Replace all json.dumps calls with safe_json_dumps in experiment tracker."""
    
    with open('core/experiment_tracker.py', 'r') as f:
        content = f.read()
    
    # Replace json.dumps calls with safe serialization
    replacements = [
        (r'json\.dumps\(run\.hyperparameters\)', 
         'safe_json_dumps(sanitize_config_for_json(run.hyperparameters))'),
        (r'json\.dumps\(run\.metrics\)', 
         'safe_json_dumps(run.metrics)'),
        (r'json\.dumps\(run\.artifacts\)', 
         'safe_json_dumps(run.artifacts or {})'),
        (r'json\.dumps\(run\.metadata\)', 
         'safe_json_dumps(run.metadata)'),
        (r'json\.dumps\(run\.tags\)', 
         'safe_json_dumps(run.tags or [])'),
        (r'json\.dumps\(\{[^}]*"best_trial_id"[^}]*\}\)', 
         'safe_json_dumps({"best_trial_id": tuning_result.best_trial.trial_id if tuning_result.best_trial else None})')
    ]
    
    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)
    
    # Write back the fixed content
    with open('core/experiment_tracker.py', 'w') as f:
        f.write(content)
    
    print("✓ Fixed all JSON serialization calls in experiment tracker")

if __name__ == "__main__":
    fix_experiment_tracker()