"""
Machine Learning Experiment Tracking Interface
--------------------------------------------
Comprehensive system for logging, storing, organizing, and analyzing ML experiments.
Provides full integration with hyperparameter tuning and model training workflows.
"""

import json
import sqlite3
import pandas as pd
import numpy as np
import hashlib
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
import shutil
import pickle
import joblib

from .json_serialization_utils import safe_json_dumps, safe_json_loads, sanitize_config_for_json

@dataclass
class ExperimentRun:
    """Represents a single ML experiment run."""
    run_id: str
    experiment_name: str
    parent_job_id: Optional[str]
    timestamp: str
    status: str
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, float]
    artifacts: Dict[str, str]
    metadata: Dict[str, Any]
    tags: List[str]
    notes: str
    source_code_hash: Optional[str]
    dataset_hash: Optional[str]
    duration_seconds: float

@dataclass
class ExperimentJob:
    """Represents a hyperparameter tuning job or experiment group."""
    job_id: str
    experiment_name: str
    job_type: str
    configuration: Dict[str, Any]
    status: str
    start_time: str
    end_time: Optional[str]
    total_trials: int
    best_trial_id: Optional[str]
    metadata: Dict[str, Any]

class ExperimentTracker:
    """
    Comprehensive experiment tracking system with SQLite backend.
    Supports hierarchical experiments, artifact management, and rich querying.
    """
    
    def __init__(self, db_path: str = "data/experiment_tracking/experiments.db"):
        """
        Initialize the experiment tracker.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir = self.db_path.parent / "artifacts"
        self.artifacts_dir.mkdir(exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize the SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Experiment jobs table (parent experiments/tuning jobs)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiment_jobs (
                    job_id TEXT PRIMARY KEY,
                    experiment_name TEXT NOT NULL,
                    job_type TEXT NOT NULL,
                    configuration TEXT,
                    status TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    total_trials INTEGER DEFAULT 0,
                    best_trial_id TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Experiment runs table (individual trials/runs)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiment_runs (
                    run_id TEXT PRIMARY KEY,
                    experiment_name TEXT NOT NULL,
                    parent_job_id TEXT,
                    timestamp TEXT NOT NULL,
                    status TEXT NOT NULL,
                    hyperparameters TEXT,
                    metrics TEXT,
                    artifacts TEXT,
                    metadata TEXT,
                    tags TEXT,
                    notes TEXT,
                    source_code_hash TEXT,
                    dataset_hash TEXT,
                    duration_seconds REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (parent_job_id) REFERENCES experiment_jobs (job_id)
                )
            """)
            
            # Metrics time series table (for tracking metrics over epochs/iterations)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics_timeseries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    step INTEGER NOT NULL,
                    value REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES experiment_runs (run_id)
                )
            """)
            
            # Create indexes for better query performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_experiment ON experiment_runs(experiment_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_parent ON experiment_runs(parent_job_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_timestamp ON experiment_runs(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_run ON metrics_timeseries(run_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics_timeseries(metric_name)")
            
            conn.commit()
    
    def start_experiment(self, experiment_name: str, job_id: str, config: Dict[str, Any]) -> str:
        """
        Start a new experiment job (e.g., hyperparameter tuning job).
        
        Args:
            experiment_name: Name of the experiment
            job_id: Unique job identifier
            config: Configuration for the experiment
            
        Returns:
            Job ID
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            job = ExperimentJob(
                job_id=job_id,
                experiment_name=experiment_name,
                job_type="hyperparameter_tuning",
                configuration=config,
                status="running",
                start_time=datetime.now().isoformat(),
                end_time=None,
                total_trials=0,
                best_trial_id=None,
                metadata={}
            )
            
            # Sanitize configuration for safe JSON serialization
            serializable_config = sanitize_config_for_json(job.configuration)
            serializable_metadata = sanitize_config_for_json(job.metadata)
            
            cursor.execute("""
                INSERT INTO experiment_jobs 
                (job_id, experiment_name, job_type, configuration, status, start_time, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                job.job_id,
                job.experiment_name,
                job.job_type,
                safe_json_dumps(serializable_config),
                job.status,
                job.start_time,
                safe_json_dumps(serializable_metadata)
            ))
            
            conn.commit()
        
        return job_id
    
    def log_trial(self, parent_job_id: str, trial_result) -> str:
        """
        Log a single trial/run result.
        
        Args:
            parent_job_id: ID of the parent experiment job
            trial_result: TrialResult object from tuning wizard
            
        Returns:
            Run ID
        """
        run_id = trial_result.trial_id
        
        # Store model artifact if provided
        artifacts = {}
        if trial_result.model_artifact_path:
            artifact_name = f"model_{run_id}.joblib"
            artifact_path = self.artifacts_dir / artifact_name
            shutil.copy2(trial_result.model_artifact_path, artifact_path)
            artifacts["model"] = str(artifact_path)
        
        # Create run record
        run = ExperimentRun(
            run_id=run_id,
            experiment_name="",  # Will be filled from parent job
            parent_job_id=parent_job_id,
            timestamp=datetime.now().isoformat(),
            status="completed" if trial_result.cv_scores else "failed",
            hyperparameters=trial_result.hyperparameters,
            metrics={
                "objective_value": trial_result.objective_value,
                "cv_mean": trial_result.cv_mean,
                "cv_std": trial_result.cv_std,
                "training_time": trial_result.training_time
            },
            artifacts=artifacts,
            metadata={
                "trial_number": trial_result.trial_number,
                "cv_scores": trial_result.cv_scores
            },
            tags=[],
            notes="",
            source_code_hash=None,
            dataset_hash=None,
            duration_seconds=trial_result.training_time
        )
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get experiment name from parent job
            cursor.execute("SELECT experiment_name FROM experiment_jobs WHERE job_id = ?", (parent_job_id,))
            result = cursor.fetchone()
            if result:
                run.experiment_name = result[0]
            
            # Insert run
            cursor.execute("""
                INSERT INTO experiment_runs 
                (run_id, experiment_name, parent_job_id, timestamp, status, hyperparameters, 
                 metrics, artifacts, metadata, tags, notes, duration_seconds)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run.run_id,
                run.experiment_name,
                run.parent_job_id,
                run.timestamp,
                run.status,
                safe_json_dumps(sanitize_config_for_json(run.hyperparameters)),
                safe_json_dumps(run.metrics),
                safe_json_dumps(run.artifacts or {}),
                safe_json_dumps(run.metadata),
                safe_json_dumps(run.tags or []),
                run.notes,
                run.duration_seconds
            ))
            
            # Update parent job trial count
            cursor.execute("""
                UPDATE experiment_jobs 
                SET total_trials = total_trials + 1 
                WHERE job_id = ?
            """, (parent_job_id,))
            
            conn.commit()
        
        return run_id
    
    def complete_experiment(self, job_id: str, tuning_result) -> None:
        """
        Mark an experiment job as completed and store final results.
        
        Args:
            job_id: Job ID to complete
            tuning_result: TuningJobResult object
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE experiment_jobs 
                SET status = 'completed', 
                    end_time = ?, 
                    best_trial_id = ?,
                    metadata = ?
                WHERE job_id = ?
            """, (
                datetime.now().isoformat(),
                tuning_result.best_trial.trial_id,
                json.dumps({
                    "total_time": tuning_result.total_time,
                    "convergence_history": tuning_result.convergence_history,
                    "search_space_coverage": tuning_result.search_space_coverage
                }),
                job_id
            ))
            
            conn.commit()
    
    def log_single_run(self, experiment_name: str, hyperparameters: Dict[str, Any], 
                      metrics: Dict[str, float], artifacts: Dict[str, str] = None,
                      tags: List[str] = None, notes: str = "") -> str:
        """
        Log a single standalone experiment run (not part of a tuning job).
        
        Args:
            experiment_name: Name of the experiment
            hyperparameters: Model hyperparameters
            metrics: Performance metrics
            artifacts: Paths to saved artifacts
            tags: Experiment tags
            notes: Additional notes
            
        Returns:
            Run ID
        """
        run_id = str(uuid.uuid4())
        
        run = ExperimentRun(
            run_id=run_id,
            experiment_name=experiment_name,
            parent_job_id=None,
            timestamp=datetime.now().isoformat(),
            status="completed",
            hyperparameters=hyperparameters,
            metrics=metrics,
            artifacts=artifacts or {},
            metadata={},
            tags=tags or [],
            notes=notes,
            source_code_hash=None,
            dataset_hash=None,
            duration_seconds=0.0
        )
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO experiment_runs 
                (run_id, experiment_name, parent_job_id, timestamp, status, hyperparameters, 
                 metrics, artifacts, metadata, tags, notes, duration_seconds)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run.run_id,
                run.experiment_name,
                run.parent_job_id,
                run.timestamp,
                run.status,
                safe_json_dumps(sanitize_config_for_json(run.hyperparameters)),
                safe_json_dumps(run.metrics),
                safe_json_dumps(run.artifacts or {}),
                safe_json_dumps(run.metadata),
                safe_json_dumps(run.tags or []),
                run.notes,
                run.duration_seconds
            ))
            
            conn.commit()
        
        return run_id
    
    def get_experiment_jobs(self, experiment_name: str = None, limit: int = 100) -> List[ExperimentJob]:
        """
        Get experiment jobs, optionally filtered by experiment name.
        
        Args:
            experiment_name: Filter by experiment name (optional)
            limit: Maximum number of jobs to return
            
        Returns:
            List of ExperimentJob objects
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if experiment_name:
                cursor.execute("""
                    SELECT * FROM experiment_jobs 
                    WHERE experiment_name = ? 
                    ORDER BY start_time DESC 
                    LIMIT ?
                """, (experiment_name, limit))
            else:
                cursor.execute("""
                    SELECT * FROM experiment_jobs 
                    ORDER BY start_time DESC 
                    LIMIT ?
                """, (limit,))
            
            jobs = []
            for row in cursor.fetchall():
                job = ExperimentJob(
                    job_id=row[0],
                    experiment_name=row[1],
                    job_type=row[2],
                    configuration=json.loads(row[3]) if row[3] else {},
                    status=row[4],
                    start_time=row[5],
                    end_time=row[6],
                    total_trials=row[7],
                    best_trial_id=row[8],
                    metadata=json.loads(row[9]) if row[9] else {}
                )
                jobs.append(job)
            
            return jobs
    
    def get_experiment_runs(self, experiment_name: str = None, parent_job_id: str = None, 
                           limit: int = 100) -> List[ExperimentRun]:
        """
        Get experiment runs, optionally filtered by experiment name or parent job.
        
        Args:
            experiment_name: Filter by experiment name (optional)
            parent_job_id: Filter by parent job ID (optional)
            limit: Maximum number of runs to return
            
        Returns:
            List of ExperimentRun objects
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM experiment_runs WHERE 1=1"
            params = []
            
            if experiment_name:
                query += " AND experiment_name = ?"
                params.append(experiment_name)
            
            if parent_job_id:
                query += " AND parent_job_id = ?"
                params.append(parent_job_id)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            
            runs = []
            for row in cursor.fetchall():
                run = ExperimentRun(
                    run_id=row[0],
                    experiment_name=row[1],
                    parent_job_id=row[2],
                    timestamp=row[3],
                    status=row[4],
                    hyperparameters=json.loads(row[5]) if row[5] else {},
                    metrics=json.loads(row[6]) if row[6] else {},
                    artifacts=json.loads(row[7]) if row[7] else {},
                    metadata=json.loads(row[8]) if row[8] else {},
                    tags=json.loads(row[9]) if row[9] else [],
                    notes=row[10] or "",
                    source_code_hash=row[11],
                    dataset_hash=row[12],
                    duration_seconds=row[13] or 0.0
                )
                runs.append(run)
            
            return runs
    
    def get_run_by_id(self, run_id: str) -> Optional[ExperimentRun]:
        """Get a specific run by ID."""
        runs = self.get_experiment_runs()
        for run in runs:
            if run.run_id == run_id:
                return run
        return None
    
    def get_best_runs(self, experiment_name: str, metric_name: str, 
                     maximize: bool = True, limit: int = 10) -> List[ExperimentRun]:
        """
        Get the best runs for an experiment based on a specific metric.
        
        Args:
            experiment_name: Name of the experiment
            metric_name: Name of the metric to optimize
            maximize: Whether to maximize the metric (True) or minimize (False)
            limit: Maximum number of runs to return
            
        Returns:
            List of best ExperimentRun objects sorted by metric
        """
        runs = self.get_experiment_runs(experiment_name=experiment_name, limit=1000)
        
        # Filter runs that have the specified metric
        runs_with_metric = [run for run in runs if metric_name in run.metrics]
        
        # Sort by metric
        runs_with_metric.sort(
            key=lambda r: r.metrics[metric_name],
            reverse=maximize
        )
        
        return runs_with_metric[:limit]
    
    def compare_runs(self, run_ids: List[str]) -> pd.DataFrame:
        """
        Compare multiple runs side by side.
        
        Args:
            run_ids: List of run IDs to compare
            
        Returns:
            DataFrame with comparison data
        """
        runs = []
        for run_id in run_ids:
            run = self.get_run_by_id(run_id)
            if run:
                runs.append(run)
        
        if not runs:
            return pd.DataFrame()
        
        # Collect all unique hyperparameters and metrics
        all_params = set()
        all_metrics = set()
        
        for run in runs:
            all_params.update(run.hyperparameters.keys())
            all_metrics.update(run.metrics.keys())
        
        # Build comparison DataFrame
        data = []
        for run in runs:
            row = {"run_id": run.run_id, "experiment_name": run.experiment_name}
            
            # Add hyperparameters
            for param in sorted(all_params):
                row[f"param_{param}"] = run.hyperparameters.get(param, None)
            
            # Add metrics
            for metric in sorted(all_metrics):
                row[f"metric_{metric}"] = run.metrics.get(metric, None)
            
            row["duration"] = run.duration_seconds
            row["timestamp"] = run.timestamp
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def get_experiment_summary(self, experiment_name: str) -> Dict[str, Any]:
        """
        Get a comprehensive summary of an experiment.
        
        Args:
            experiment_name: Name of the experiment
            
        Returns:
            Dictionary with experiment summary statistics
        """
        runs = self.get_experiment_runs(experiment_name=experiment_name, limit=1000)
        jobs = self.get_experiment_jobs(experiment_name=experiment_name, limit=100)
        
        if not runs:
            return {"error": "No runs found for this experiment"}
        
        # Calculate statistics
        all_metrics = set()
        for run in runs:
            all_metrics.update(run.metrics.keys())
        
        metric_stats = {}
        for metric in all_metrics:
            values = [run.metrics[metric] for run in runs if metric in run.metrics]
            if values:
                metric_stats[metric] = {
                    "count": len(values),
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "median": np.median(values)
                }
        
        return {
            "experiment_name": experiment_name,
            "total_runs": len(runs),
            "total_jobs": len(jobs),
            "first_run": min(run.timestamp for run in runs),
            "last_run": max(run.timestamp for run in runs),
            "total_duration": sum(run.duration_seconds for run in runs),
            "metric_statistics": metric_stats,
            "unique_hyperparameters": len(set(
                tuple(sorted(run.hyperparameters.items())) for run in runs
            )),
            "status_breakdown": {
                status: len([r for r in runs if r.status == status])
                for status in set(run.status for run in runs)
            }
        }
    
    def search_runs(self, query: Dict[str, Any], limit: int = 100) -> List[ExperimentRun]:
        """
        Search for runs based on complex criteria.
        
        Args:
            query: Search criteria dictionary
            limit: Maximum number of results
            
        Returns:
            List of matching ExperimentRun objects
        """
        runs = self.get_experiment_runs(limit=10000)  # Get all runs first
        
        filtered_runs = []
        for run in runs:
            match = True
            
            # Filter by experiment name
            if "experiment_name" in query and query["experiment_name"] not in run.experiment_name:
                match = False
            
            # Filter by hyperparameters
            if "hyperparameters" in query:
                for param, value in query["hyperparameters"].items():
                    if param not in run.hyperparameters or run.hyperparameters[param] != value:
                        match = False
                        break
            
            # Filter by metric ranges
            if "metric_ranges" in query:
                for metric, (min_val, max_val) in query["metric_ranges"].items():
                    if metric not in run.metrics:
                        match = False
                        break
                    metric_val = run.metrics[metric]
                    if metric_val < min_val or metric_val > max_val:
                        match = False
                        break
            
            # Filter by tags
            if "tags" in query:
                required_tags = set(query["tags"])
                run_tags = set(run.tags)
                if not required_tags.issubset(run_tags):
                    match = False
            
            # Filter by date range
            if "date_range" in query:
                start_date, end_date = query["date_range"]
                run_date = datetime.fromisoformat(run.timestamp)
                if run_date < start_date or run_date > end_date:
                    match = False
            
            if match:
                filtered_runs.append(run)
        
        return filtered_runs[:limit]
    
    def export_experiment_data(self, experiment_name: str, format: str = "csv") -> str:
        """
        Export experiment data to various formats.
        
        Args:
            experiment_name: Name of the experiment to export
            format: Export format ("csv", "json", "excel")
            
        Returns:
            Path to exported file
        """
        runs = self.get_experiment_runs(experiment_name=experiment_name, limit=10000)
        
        if not runs:
            raise ValueError(f"No runs found for experiment: {experiment_name}")
        
        # Convert to DataFrame
        df = self.compare_runs([run.run_id for run in runs])
        
        # Export based on format
        export_dir = self.db_path.parent / "exports"
        export_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == "csv":
            filepath = export_dir / f"{experiment_name}_{timestamp}.csv"
            df.to_csv(filepath, index=False)
        elif format == "json":
            filepath = export_dir / f"{experiment_name}_{timestamp}.json"
            df.to_json(filepath, orient="records", indent=2)
        elif format == "excel":
            filepath = export_dir / f"{experiment_name}_{timestamp}.xlsx"
            df.to_excel(filepath, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return str(filepath)

# Global experiment tracker instance
_global_tracker = None

def get_experiment_tracker() -> ExperimentTracker:
    """Get the global experiment tracker instance."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = ExperimentTracker()
    return _global_tracker

# Convenience functions
def log_experiment(experiment_name: str, hyperparameters: Dict[str, Any], 
                  metrics: Dict[str, float], **kwargs) -> str:
    """Convenience function to log a single experiment run."""
    tracker = get_experiment_tracker()
    return tracker.log_single_run(experiment_name, hyperparameters, metrics, **kwargs)

def get_best_run(experiment_name: str, metric_name: str, maximize: bool = True) -> Optional[ExperimentRun]:
    """Get the best run for an experiment based on a metric."""
    tracker = get_experiment_tracker()
    best_runs = tracker.get_best_runs(experiment_name, metric_name, maximize, limit=1)
    return best_runs[0] if best_runs else None

def compare_experiments(run_ids: List[str]) -> pd.DataFrame:
    """Compare multiple experiment runs."""
    tracker = get_experiment_tracker()
    return tracker.compare_runs(run_ids)