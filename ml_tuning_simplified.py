"""
Simplified Hyperparameter Tuning System
--------------------------------------
Core tuning functionality with grid search and random search strategies.
"""

import json
import uuid
import time
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import make_scorer, mean_absolute_error

@dataclass
class TrialResult:
    """Result of a single hyperparameter trial."""
    trial_id: str
    trial_number: int
    hyperparameters: Dict[str, Any]
    objective_value: float
    cv_mean: float
    cv_std: float
    cv_scores: List[float]
    training_time: float
    model_artifact_path: Optional[str] = None

@dataclass
class TuningJobResult:
    """Result of a complete tuning job."""
    job_id: str
    experiment_name: str
    best_trial: TrialResult
    all_trials: List[TrialResult]
    total_time: float
    convergence_history: List[float]
    search_space_coverage: float

class SimplifiedTuningWizard:
    """Simplified hyperparameter tuning with grid and random search."""
    
    def __init__(self, experiment_tracker=None):
        self.experiment_tracker = experiment_tracker
        
        # Available models
        self.models = {
            "RandomForest": RandomForestRegressor,
            "GradientBoosting": GradientBoostingRegressor,
            "Ridge": Ridge,
            "Lasso": Lasso,
            "SVR": SVR,
            "MLP": MLPRegressor
        }
        
        # Default parameter grids
        self.default_grids = {
            "RandomForest": {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, 20],
                "min_samples_split": [2, 5, 10]
            },
            "GradientBoosting": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 10]
            },
            "Ridge": {
                "alpha": [0.1, 1.0, 10.0],
                "fit_intercept": [True, False]
            },
            "Lasso": {
                "alpha": [0.1, 1.0, 10.0],
                "fit_intercept": [True, False]
            },
            "SVR": {
                "C": [0.1, 1.0, 10.0],
                "kernel": ["rbf", "linear"],
                "epsilon": [0.01, 0.1, 1.0]
            },
            "MLP": {
                "hidden_layer_sizes": [(50,), (100,), (50, 50)],
                "alpha": [0.0001, 0.001, 0.01],
                "learning_rate_init": [0.001, 0.01, 0.1]
            }
        }
    
    def run_tuning_job(self, config: Dict[str, Any], X: np.ndarray, y: np.ndarray) -> TuningJobResult:
        """Execute a hyperparameter tuning job."""
        start_time = time.time()
        job_id = str(uuid.uuid4())
        
        # Start experiment tracking
        if self.experiment_tracker:
            self.experiment_tracker.start_experiment(
                config["experiment_name"], 
                job_id, 
                config
            )
        
        # Generate parameter combinations
        if config.get("search_strategy") == "grid_search":
            param_combinations = self._generate_grid_combinations(config)
        else:  # random_search
            param_combinations = self._generate_random_combinations(config)
        
        # Execute trials
        trials = []
        convergence_history = []
        best_score = float('-inf')
        
        for i, params in enumerate(param_combinations):
            trial_result = self._evaluate_trial(
                trial_number=i+1,
                hyperparameters=params,
                model_class=self.models[config["model_name"]],
                X=X, y=y,
                cv_folds=config.get("cv_folds", 5),
                scoring=config.get("objective_metric", "neg_mean_absolute_error")
            )
            
            trials.append(trial_result)
            
            # Update convergence history
            if trial_result.objective_value > best_score:
                best_score = trial_result.objective_value
            convergence_history.append(best_score)
            
            # Log trial to experiment tracker
            if self.experiment_tracker:
                self.experiment_tracker.log_trial(job_id, trial_result)
        
        # Find best trial
        best_trial = max(trials, key=lambda t: t.objective_value)
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Create result object
        result = TuningJobResult(
            job_id=job_id,
            experiment_name=config["experiment_name"],
            best_trial=best_trial,
            all_trials=trials,
            total_time=total_time,
            convergence_history=convergence_history,
            search_space_coverage=len(trials)
        )
        
        # Complete experiment tracking
        if self.experiment_tracker:
            self.experiment_tracker.complete_experiment(job_id, result)
        
        return result
    
    def _generate_grid_combinations(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate all combinations for grid search."""
        model_name = config["model_name"]
        param_spaces = config.get("parameter_spaces")
        
        # Use default grid if parameter_spaces is None or empty
        if param_spaces is None or not param_spaces:
            param_grid = self.default_grids.get(model_name, {})
        else:
            param_grid = param_spaces
        
        # Convert to list of parameter combinations
        combinations = []
        
        def generate_combinations(params, current_combo=None):
            if current_combo is None:
                current_combo = {}
            
            if not params:
                combinations.append(current_combo.copy())
                return
            
            param_name = list(params.keys())[0]
            param_values = params[param_name]
            remaining_params = {k: v for k, v in params.items() if k != param_name}
            
            for value in param_values:
                current_combo[param_name] = value
                generate_combinations(remaining_params, current_combo)
                del current_combo[param_name]
        
        generate_combinations(param_grid)
        
        # Limit to max trials
        max_trials = config.get("n_trials", 50)
        return combinations[:max_trials]
    
    def _generate_random_combinations(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate random combinations for random search."""
        model_name = config["model_name"]
        param_spaces = config.get("parameter_spaces")
        
        # Use default grid if parameter_spaces is None or empty
        if param_spaces is None or not param_spaces:
            param_grid = self.default_grids.get(model_name, {})
        else:
            param_grid = param_spaces
            
        n_trials = config.get("n_trials", 20)
        
        combinations = []
        np.random.seed(config.get("random_state", 42))
        
        for _ in range(n_trials):
            combo = {}
            for param_name, param_values in param_grid.items():
                combo[param_name] = np.random.choice(param_values)
            combinations.append(combo)
        
        return combinations
    
    def _evaluate_trial(self, trial_number: int, hyperparameters: Dict[str, Any], 
                       model_class, X: np.ndarray, y: np.ndarray, 
                       cv_folds: int, scoring: str) -> TrialResult:
        """Evaluate a single trial with given hyperparameters."""
        trial_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Create model with hyperparameters
            model = model_class(**hyperparameters)
            
            # Perform cross-validation
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            
            # Convert scoring metric
            if scoring == "neg_mean_absolute_error":
                scorer = make_scorer(mean_absolute_error, greater_is_better=False)
            else:
                scorer = scoring
            
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scorer)
            
            # Calculate metrics
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            objective_value = cv_mean  # Higher is better (negative errors become less negative)
            
            training_time = time.time() - start_time
            
            return TrialResult(
                trial_id=trial_id,
                trial_number=trial_number,
                hyperparameters=hyperparameters,
                objective_value=objective_value,
                cv_mean=cv_mean,
                cv_std=cv_std,
                cv_scores=cv_scores.tolist(),
                training_time=training_time
            )
            
        except Exception as e:
            # Return failed trial
            training_time = time.time() - start_time
            return TrialResult(
                trial_id=trial_id,
                trial_number=trial_number,
                hyperparameters=hyperparameters,
                objective_value=float('-inf'),
                cv_mean=float('-inf'),
                cv_std=0.0,
                cv_scores=[],
                training_time=training_time
            )

    def run_tuning(self, config: Dict[str, Any], X, y) -> TuningJobResult:
        """
        Run hyperparameter tuning with the given configuration.
        
        Args:
            config: Tuning configuration dictionary
            X: Feature matrix
            y: Target vector
            
        Returns:
            TuningJobResult with best trial and all trials
        """
        # Generate parameter combinations based on search strategy
        if config["search_strategy"] == "grid_search":
            combinations = self._generate_grid_combinations(config)
        else:  # random_search
            combinations = self._generate_random_combinations(config)
        
        # Initialize tracking
        all_trials = []
        best_trial = None
        best_score = float('-inf')
        
        # Run trials
        for i, params in enumerate(combinations):
            trial_result = self._evaluate_trial(
                params, X, y, 
                cv_folds=config.get("cv_folds", 5),
                scoring=config.get("objective_metric", "neg_mean_absolute_error")
            )
            all_trials.append(trial_result)
            
            if trial_result.objective_value > best_score:
                best_score = trial_result.objective_value
                best_trial = trial_result
        
        # Create result
        result = TuningJobResult(
            job_id=f"job_{config['experiment_name']}",
            best_trial=best_trial,
            all_trials=all_trials,
            total_time=sum(t.training_time for t in all_trials),
            convergence_history=[t.objective_value for t in all_trials],
            search_space_coverage=len(all_trials) / max(1, config.get("n_trials", 20))
        )
        
        return result

def create_tuning_config(experiment_name: str, model_name: str, 
                        search_strategy: str = "random_search",
                        n_trials: int = 20, cv_folds: int = 5,
                        objective_metric: str = "neg_mean_absolute_error",
                        parameter_spaces: Dict[str, List] = None,
                        random_state: int = 42) -> Dict[str, Any]:
    """Create a tuning configuration dictionary."""
    return {
        "experiment_name": experiment_name,
        "model_name": model_name,
        "search_strategy": search_strategy,
        "n_trials": n_trials,
        "cv_folds": cv_folds,
        "objective_metric": objective_metric,
        "parameter_spaces": parameter_spaces,
        "random_state": random_state
    }