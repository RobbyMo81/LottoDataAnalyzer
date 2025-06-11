"""
Automated Hyperparameter Tuning Wizard
-------------------------------------
Advanced hyperparameter optimization system with multiple search strategies,
comprehensive logging, and tight integration with experiment tracking.
"""

import numpy as np
import pandas as pd
import json
import time
import hashlib
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import concurrent.futures
from pathlib import Path

# ML imports
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, roc_auc_score
from sklearn.base import BaseEstimator
import joblib

# Optimization libraries
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from scipy.stats import uniform, randint
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

class SearchStrategy(Enum):
    """Available hyperparameter search strategies."""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    
class OptimizationDirection(Enum):
    """Optimization direction for metrics."""
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"

class ParameterType(Enum):
    """Types of hyperparameters."""
    FLOAT = "float"
    INTEGER = "int"
    CATEGORICAL = "categorical"
    BOOLEAN = "bool"

@dataclass
class ParameterSpace:
    """Definition of a hyperparameter search space."""
    name: str
    param_type: ParameterType
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[List[Any]] = None
    log_scale: bool = False
    step: Optional[float] = None

@dataclass
class TuningConfiguration:
    """Configuration for hyperparameter tuning job."""
    experiment_name: str
    model_class: type
    parameter_spaces: List[ParameterSpace]
    search_strategy: SearchStrategy
    objective_metric: str
    optimization_direction: OptimizationDirection
    cv_folds: int = 5
    cv_strategy: str = "kfold"  # "kfold" or "stratified"
    n_trials: int = 100
    timeout_seconds: Optional[int] = None
    n_jobs: int = 1
    random_state: int = 42
    early_stopping: bool = False
    early_stopping_rounds: int = 10

@dataclass
class TrialResult:
    """Result of a single hyperparameter trial."""
    trial_id: str
    hyperparameters: Dict[str, Any]
    objective_value: float
    cv_scores: List[float]
    cv_mean: float
    cv_std: float
    training_time: float
    model_artifact_path: Optional[str] = None
    additional_metrics: Optional[Dict[str, float]] = None
    trial_number: int = 0

@dataclass
class TuningJobResult:
    """Complete result of a hyperparameter tuning job."""
    job_id: str
    experiment_name: str
    configuration: TuningConfiguration
    best_trial: TrialResult
    all_trials: List[TrialResult]
    total_time: float
    convergence_history: List[float]
    search_space_coverage: Dict[str, Any]

class HyperparameterTuningWizard:
    """
    Advanced hyperparameter tuning system with multiple search strategies.
    Integrates with experiment tracking for comprehensive ML experiment management.
    """
    
    def __init__(self, experiment_tracker=None):
        """
        Initialize the tuning wizard.
        
        Args:
            experiment_tracker: Experiment tracking interface instance
        """
        self.experiment_tracker = experiment_tracker
        self.current_job_id = None
        self.results_cache = {}
        
    def create_parameter_space(self, name: str, param_type: str, **kwargs) -> ParameterSpace:
        """
        Create a parameter space definition.
        
        Args:
            name: Parameter name
            param_type: Type of parameter (float, int, categorical, bool)
            **kwargs: Additional parameter space configuration
            
        Returns:
            ParameterSpace object
        """
        return ParameterSpace(
            name=name,
            param_type=ParameterType(param_type),
            **kwargs
        )
    
    def _generate_grid_search_combinations(self, parameter_spaces: List[ParameterSpace]) -> List[Dict[str, Any]]:
        """Generate all combinations for grid search."""
        from itertools import product
        
        param_grids = {}
        for space in parameter_spaces:
            if space.param_type == ParameterType.FLOAT:
                if space.step:
                    param_grids[space.name] = np.arange(space.low, space.high + space.step, space.step)
                else:
                    param_grids[space.name] = np.linspace(space.low, space.high, 10)
            elif space.param_type == ParameterType.INTEGER:
                step = space.step or 1
                param_grids[space.name] = list(range(int(space.low), int(space.high) + 1, int(step)))
            elif space.param_type == ParameterType.CATEGORICAL:
                param_grids[space.name] = space.choices
            elif space.param_type == ParameterType.BOOLEAN:
                param_grids[space.name] = [True, False]
        
        # Generate all combinations
        keys = list(param_grids.keys())
        combinations = list(product(*[param_grids[key] for key in keys]))
        
        return [dict(zip(keys, combo)) for combo in combinations]
    
    def _generate_random_search_combinations(self, parameter_spaces: List[ParameterSpace], 
                                           n_trials: int, random_state: int = 42) -> List[Dict[str, Any]]:
        """Generate random combinations for random search."""
        np.random.seed(random_state)
        combinations = []
        
        for _ in range(n_trials):
            combo = {}
            for space in parameter_spaces:
                if space.param_type == ParameterType.FLOAT:
                    if space.log_scale:
                        combo[space.name] = np.exp(np.random.uniform(np.log(space.low), np.log(space.high)))
                    else:
                        combo[space.name] = np.random.uniform(space.low, space.high)
                elif space.param_type == ParameterType.INTEGER:
                    combo[space.name] = np.random.randint(space.low, space.high + 1)
                elif space.param_type == ParameterType.CATEGORICAL:
                    combo[space.name] = np.random.choice(space.choices)
                elif space.param_type == ParameterType.BOOLEAN:
                    combo[space.name] = np.random.choice([True, False])
            combinations.append(combo)
        
        return combinations
    
    def _setup_bayesian_optimization(self, parameter_spaces: List[ParameterSpace], 
                                   config: TuningConfiguration) -> Optional[Any]:
        """Setup Bayesian optimization using Optuna."""
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for Bayesian optimization. Install with: pip install optuna")
        
        # Create Optuna study
        direction = "maximize" if config.optimization_direction == OptimizationDirection.MAXIMIZE else "minimize"
        study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=config.random_state))
        
        return study
    
    def _suggest_bayesian_parameters(self, trial, parameter_spaces: List[ParameterSpace]) -> Dict[str, Any]:
        """Suggest parameters using Bayesian optimization."""
        params = {}
        for space in parameter_spaces:
            if space.param_type == ParameterType.FLOAT:
                if space.log_scale:
                    params[space.name] = trial.suggest_float(space.name, space.low, space.high, log=True)
                else:
                    params[space.name] = trial.suggest_float(space.name, space.low, space.high)
            elif space.param_type == ParameterType.INTEGER:
                params[space.name] = trial.suggest_int(space.name, int(space.low), int(space.high))
            elif space.param_type == ParameterType.CATEGORICAL:
                params[space.name] = trial.suggest_categorical(space.name, space.choices)
            elif space.param_type == ParameterType.BOOLEAN:
                params[space.name] = trial.suggest_categorical(space.name, [True, False])
        
        return params
    
    def _evaluate_hyperparameters(self, hyperparameters: Dict[str, Any], 
                                config: TuningConfiguration, X, y) -> TrialResult:
        """Evaluate a specific hyperparameter configuration."""
        start_time = time.time()
        trial_id = str(uuid.uuid4())
        
        try:
            # Create model with hyperparameters
            model = config.model_class(**hyperparameters)
            
            # Setup cross-validation
            if config.cv_strategy == "stratified" and hasattr(y, 'nunique') and y.nunique() < 20:
                cv = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
            else:
                cv = KFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
            
            # Perform cross-validation
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring=self._get_scoring_function(config.objective_metric))
            
            # Handle negative scores (some sklearn scorers return negative values)
            if config.objective_metric in ['neg_mean_squared_error', 'neg_mean_absolute_error']:
                cv_scores = -cv_scores
            
            objective_value = cv_scores.mean()
            training_time = time.time() - start_time
            
            # Train final model on full dataset for artifact storage
            model.fit(X, y)
            model_path = f"data/ml_tuning/models/trial_{trial_id}.joblib"
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, model_path)
            
            return TrialResult(
                trial_id=trial_id,
                hyperparameters=hyperparameters,
                objective_value=objective_value,
                cv_scores=cv_scores.tolist(),
                cv_mean=cv_scores.mean(),
                cv_std=cv_scores.std(),
                training_time=training_time,
                model_artifact_path=model_path
            )
            
        except Exception as e:
            # Return failed trial
            return TrialResult(
                trial_id=trial_id,
                hyperparameters=hyperparameters,
                objective_value=float('-inf') if config.optimization_direction == OptimizationDirection.MAXIMIZE else float('inf'),
                cv_scores=[],
                cv_mean=0.0,
                cv_std=0.0,
                training_time=time.time() - start_time,
                additional_metrics={"error": str(e)}
            )
    
    def _get_scoring_function(self, metric_name: str) -> str:
        """Map metric names to sklearn scoring functions."""
        metric_mapping = {
            'accuracy': 'accuracy',
            'f1': 'f1',
            'f1_weighted': 'f1_weighted',
            'roc_auc': 'roc_auc',
            'mse': 'neg_mean_squared_error',
            'mae': 'neg_mean_absolute_error',
            'r2': 'r2'
        }
        return metric_mapping.get(metric_name, metric_name)
    
    def run_tuning_job(self, config: TuningConfiguration, X, y) -> TuningJobResult:
        """
        Execute a complete hyperparameter tuning job.
        
        Args:
            config: Tuning configuration
            X: Feature matrix
            y: Target vector
            
        Returns:
            TuningJobResult with all trial results and best configuration
        """
        job_start_time = time.time()
        self.current_job_id = str(uuid.uuid4())
        
        print(f"Starting hyperparameter tuning job: {config.experiment_name}")
        print(f"Strategy: {config.search_strategy.value}")
        print(f"Model: {config.model_class.__name__}")
        print(f"Objective: {config.objective_metric} ({config.optimization_direction.value})")
        
        # Log experiment start
        if self.experiment_tracker:
            self.experiment_tracker.start_experiment(
                experiment_name=config.experiment_name,
                job_id=self.current_job_id,
                config=asdict(config)
            )
        
        all_trials = []
        convergence_history = []
        
        if config.search_strategy == SearchStrategy.GRID_SEARCH:
            # Grid search implementation
            param_combinations = self._generate_grid_search_combinations(config.parameter_spaces)
            total_trials = min(len(param_combinations), config.n_trials)
            
            for i, hyperparameters in enumerate(param_combinations[:config.n_trials]):
                print(f"Trial {i+1}/{total_trials}: {hyperparameters}")
                
                trial_result = self._evaluate_hyperparameters(hyperparameters, config, X, y)
                trial_result.trial_number = i + 1
                all_trials.append(trial_result)
                
                # Log trial to experiment tracker
                if self.experiment_tracker:
                    self.experiment_tracker.log_trial(self.current_job_id, trial_result)
                
                # Update convergence history
                if config.optimization_direction == OptimizationDirection.MAXIMIZE:
                    best_so_far = max(trial.objective_value for trial in all_trials)
                else:
                    best_so_far = min(trial.objective_value for trial in all_trials)
                convergence_history.append(best_so_far)
                
        elif config.search_strategy == SearchStrategy.RANDOM_SEARCH:
            # Random search implementation
            param_combinations = self._generate_random_search_combinations(
                config.parameter_spaces, config.n_trials, config.random_state
            )
            
            for i, hyperparameters in enumerate(param_combinations):
                print(f"Trial {i+1}/{config.n_trials}: {hyperparameters}")
                
                trial_result = self._evaluate_hyperparameters(hyperparameters, config, X, y)
                trial_result.trial_number = i + 1
                all_trials.append(trial_result)
                
                # Log trial to experiment tracker
                if self.experiment_tracker:
                    self.experiment_tracker.log_trial(self.current_job_id, trial_result)
                
                # Update convergence history
                if config.optimization_direction == OptimizationDirection.MAXIMIZE:
                    best_so_far = max(trial.objective_value for trial in all_trials)
                else:
                    best_so_far = min(trial.objective_value for trial in all_trials)
                convergence_history.append(best_so_far)
                
        elif config.search_strategy == SearchStrategy.BAYESIAN_OPTIMIZATION:
            # Bayesian optimization implementation
            study = self._setup_bayesian_optimization(config.parameter_spaces, config)
            
            def objective(trial):
                hyperparameters = self._suggest_bayesian_parameters(trial, config.parameter_spaces)
                trial_result = self._evaluate_hyperparameters(hyperparameters, config, X, y)
                trial_result.trial_number = len(all_trials) + 1
                all_trials.append(trial_result)
                
                # Log trial to experiment tracker
                if self.experiment_tracker:
                    self.experiment_tracker.log_trial(self.current_job_id, trial_result)
                
                # Update convergence history
                if config.optimization_direction == OptimizationDirection.MAXIMIZE:
                    best_so_far = max(t.objective_value for t in all_trials)
                else:
                    best_so_far = min(t.objective_value for t in all_trials)
                convergence_history.append(best_so_far)
                
                print(f"Trial {trial_result.trial_number}/{config.n_trials}: {hyperparameters} -> {trial_result.objective_value:.4f}")
                
                return trial_result.objective_value
            
            study.optimize(objective, n_trials=config.n_trials, timeout=config.timeout_seconds)
        
        # Find best trial
        if config.optimization_direction == OptimizationDirection.MAXIMIZE:
            best_trial = max(all_trials, key=lambda t: t.objective_value)
        else:
            best_trial = min(all_trials, key=lambda t: t.objective_value)
        
        total_time = time.time() - job_start_time
        
        # Create final result
        result = TuningJobResult(
            job_id=self.current_job_id,
            experiment_name=config.experiment_name,
            configuration=config,
            best_trial=best_trial,
            all_trials=all_trials,
            total_time=total_time,
            convergence_history=convergence_history,
            search_space_coverage=self._analyze_search_space_coverage(all_trials, config.parameter_spaces)
        )
        
        # Log final results
        if self.experiment_tracker:
            self.experiment_tracker.complete_experiment(self.current_job_id, result)
        
        print(f"\nTuning job completed in {total_time:.2f} seconds")
        print(f"Best {config.objective_metric}: {best_trial.objective_value:.4f}")
        print(f"Best hyperparameters: {best_trial.hyperparameters}")
        
        return result
    
    def _analyze_search_space_coverage(self, trials: List[TrialResult], 
                                     parameter_spaces: List[ParameterSpace]) -> Dict[str, Any]:
        """Analyze how well the search covered the parameter space."""
        coverage = {}
        
        for space in parameter_spaces:
            param_values = [trial.hyperparameters.get(space.name) for trial in trials if space.name in trial.hyperparameters]
            
            if space.param_type in [ParameterType.FLOAT, ParameterType.INTEGER]:
                coverage[space.name] = {
                    "min_explored": min(param_values) if param_values else None,
                    "max_explored": max(param_values) if param_values else None,
                    "range_coverage": (max(param_values) - min(param_values)) / (space.high - space.low) if param_values else 0,
                    "unique_values": len(set(param_values))
                }
            elif space.param_type == ParameterType.CATEGORICAL:
                coverage[space.name] = {
                    "explored_choices": list(set(param_values)),
                    "choice_coverage": len(set(param_values)) / len(space.choices) if space.choices else 0,
                    "choice_distribution": {choice: param_values.count(choice) for choice in set(param_values)}
                }
        
        return coverage

# Convenience functions for common use cases
def create_tuning_config(experiment_name: str, model_class: type, 
                        parameter_spaces: List[Dict], **kwargs) -> TuningConfiguration:
    """Create a tuning configuration with simplified parameter space definition."""
    spaces = []
    for space_def in parameter_spaces:
        spaces.append(ParameterSpace(**space_def))
    
    return TuningConfiguration(
        experiment_name=experiment_name,
        model_class=model_class,
        parameter_spaces=spaces,
        **kwargs
    )

def quick_random_search(model_class: type, parameter_spaces: List[Dict], 
                       X, y, n_trials: int = 50, **kwargs) -> TuningJobResult:
    """Quick random search for hyperparameter tuning."""
    wizard = HyperparameterTuningWizard()
    config = create_tuning_config(
        experiment_name=f"Quick_{model_class.__name__}_RandomSearch",
        model_class=model_class,
        parameter_spaces=parameter_spaces,
        search_strategy=SearchStrategy.RANDOM_SEARCH,
        n_trials=n_trials,
        **kwargs
    )
    return wizard.run_tuning_job(config, X, y)

def quick_bayesian_search(model_class: type, parameter_spaces: List[Dict], 
                         X, y, n_trials: int = 100, **kwargs) -> TuningJobResult:
    """Quick Bayesian optimization for hyperparameter tuning."""
    wizard = HyperparameterTuningWizard()
    config = create_tuning_config(
        experiment_name=f"Quick_{model_class.__name__}_BayesianSearch",
        model_class=model_class,
        parameter_spaces=parameter_spaces,
        search_strategy=SearchStrategy.BAYESIAN_OPTIMIZATION,
        n_trials=n_trials,
        **kwargs
    )
    return wizard.run_tuning_job(config, X, y)