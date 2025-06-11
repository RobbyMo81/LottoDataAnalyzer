"""
ML Model Training Memory Buffer
------------------------------
Persistent memory system for tracking training sessions of individual ML models.
Each model maintains its own training history that is only cleared when that specific model is retrained.
"""

import os
import json
import datetime
from pathlib import Path
from typing import Dict, Optional, Any
import joblib
from dataclasses import dataclass, asdict
from enum import Enum

class ModelType(Enum):
    """Supported ML model types."""
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    RIDGE_REGRESSION = "ridge_regression"

@dataclass
class TrainingSession:
    """Data structure for storing training session details."""
    model_type: str
    timestamp: str
    hyperparameters: Dict[str, Any]
    performance_metrics: Dict[str, float]
    model_artifact_id: str
    training_duration_seconds: float
    dataset_info: Dict[str, Any]
    success: bool
    notes: Optional[str] = None

class MLMemoryBuffer:
    """
    Persistent memory buffer for ML model training sessions.
    
    Design Principles:
    1. Each model type has independent memory storage
    2. Training completion only clears memory for that specific model
    3. Other models' memory remains untouched
    4. Persistent across application restarts
    """
    
    def __init__(self, memory_dir: str = "data/ml_memory"):
        """
        Initialize the memory buffer.
        
        Args:
            memory_dir: Directory to store memory files
        """
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths for each model's memory
        self.memory_files = {
            ModelType.RANDOM_FOREST: self.memory_dir / "random_forest_memory.json",
            ModelType.GRADIENT_BOOSTING: self.memory_dir / "gradient_boosting_memory.json",
            ModelType.RIDGE_REGRESSION: self.memory_dir / "ridge_regression_memory.json"
        }
    
    def store_training_session(self, session: TrainingSession) -> bool:
        """
        Store a training session for a specific model.
        This ONLY affects the memory for the specified model type.
        
        Args:
            session: TrainingSession object with training details
            
        Returns:
            True if stored successfully, False otherwise
        """
        try:
            model_type = ModelType(session.model_type)
            memory_file = self.memory_files[model_type]
            
            # Convert session to dictionary
            session_data = asdict(session)
            
            # Store to file (overwrites previous session for this model only)
            with open(memory_file, 'w') as f:
                json.dump(session_data, f, indent=2)
            
            print(f"Training session stored for {model_type.value}")
            return True
            
        except Exception as e:
            print(f"Error storing training session: {e}")
            return False
    
    def get_training_session(self, model_type: ModelType) -> Optional[TrainingSession]:
        """
        Retrieve the last training session for a specific model.
        
        Args:
            model_type: The ML model type to query
            
        Returns:
            TrainingSession object if found, None otherwise
        """
        try:
            memory_file = self.memory_files[model_type]
            
            if not memory_file.exists():
                return None
            
            with open(memory_file, 'r') as f:
                session_data = json.load(f)
            
            return TrainingSession(**session_data)
            
        except Exception as e:
            print(f"Error retrieving training session for {model_type.value}: {e}")
            return None
    
    def clear_model_memory(self, model_type: ModelType) -> bool:
        """
        Clear memory for a specific model only.
        This does NOT affect other models' memory.
        
        Args:
            model_type: The model type to clear memory for
            
        Returns:
            True if cleared successfully, False otherwise
        """
        try:
            memory_file = self.memory_files[model_type]
            
            if memory_file.exists():
                memory_file.unlink()
                print(f"Memory cleared for {model_type.value}")
            
            return True
            
        except Exception as e:
            print(f"Error clearing memory for {model_type.value}: {e}")
            return False
    
    def get_all_training_sessions(self) -> Dict[str, Optional[TrainingSession]]:
        """
        Get training sessions for all model types.
        
        Returns:
            Dictionary mapping model type names to their training sessions
        """
        sessions = {}
        
        for model_type in ModelType:
            sessions[model_type.value] = self.get_training_session(model_type)
        
        return sessions
    
    def has_training_session(self, model_type: ModelType) -> bool:
        """
        Check if a model has a stored training session.
        
        Args:
            model_type: The model type to check
            
        Returns:
            True if session exists, False otherwise
        """
        return self.memory_files[model_type].exists()
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """
        Get summary information about all stored training sessions.
        
        Returns:
            Dictionary with summary information
        """
        summary = {
            "total_models_trained": 0,
            "models_with_memory": [],
            "last_training_times": {},
            "memory_files_size_mb": {}
        }
        
        for model_type in ModelType:
            if self.has_training_session(model_type):
                summary["total_models_trained"] += 1
                summary["models_with_memory"].append(model_type.value)
                
                session = self.get_training_session(model_type)
                if session:
                    summary["last_training_times"][model_type.value] = session.timestamp
                
                # Get file size
                memory_file = self.memory_files[model_type]
                if memory_file.exists():
                    size_bytes = memory_file.stat().st_size
                    summary["memory_files_size_mb"][model_type.value] = size_bytes / (1024 * 1024)
        
        return summary

class MLTrainingTracker:
    """
    High-level interface for tracking ML training sessions.
    """
    
    def __init__(self):
        self.memory_buffer = MLMemoryBuffer()
    
    def start_training_session(self, model_type: str, hyperparameters: Dict[str, Any], 
                             dataset_info: Dict[str, Any]) -> str:
        """
        Start tracking a new training session.
        
        Args:
            model_type: Type of model being trained
            hyperparameters: Model configuration parameters
            dataset_info: Information about training dataset
            
        Returns:
            Unique session ID for this training run
        """
        session_id = f"{model_type}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Store session start info in a temporary location
        self._current_session = {
            "session_id": session_id,
            "model_type": model_type,
            "start_time": datetime.datetime.now(),
            "hyperparameters": hyperparameters,
            "dataset_info": dataset_info
        }
        
        return session_id
    
    def complete_training_session(self, session_id: str, performance_metrics: Dict[str, float],
                                model_artifact_path: str, success: bool = True, 
                                notes: Optional[str] = None) -> bool:
        """
        Mark a training session as complete and store in memory.
        This will clear any previous memory for this specific model type.
        
        Args:
            session_id: Session ID from start_training_session
            performance_metrics: Training performance results
            model_artifact_path: Path to saved model file
            success: Whether training completed successfully
            notes: Optional notes about the training session
            
        Returns:
            True if session stored successfully
        """
        if not hasattr(self, '_current_session'):
            print(f"No active session found for {session_id}")
            return False
        
        current = self._current_session
        end_time = datetime.datetime.now()
        duration = (end_time - current["start_time"]).total_seconds()
        
        # Create training session object
        session = TrainingSession(
            model_type=current["model_type"],
            timestamp=end_time.isoformat(),
            hyperparameters=current["hyperparameters"],
            performance_metrics=performance_metrics,
            model_artifact_id=model_artifact_path,
            training_duration_seconds=duration,
            dataset_info=current["dataset_info"],
            success=success,
            notes=notes
        )
        
        # Store in memory buffer (this clears previous memory for this model only)
        result = self.memory_buffer.store_training_session(session)
        
        # Clean up current session
        if hasattr(self, '_current_session'):
            delattr(self, '_current_session')
        
        return result
    
    def get_last_training_info(self, model_type: str) -> Optional[Dict[str, Any]]:
        """
        Get information about the last training session for a model.
        
        Args:
            model_type: Model type to query
            
        Returns:
            Dictionary with training information or None
        """
        try:
            model_enum = ModelType(model_type)
            session = self.memory_buffer.get_training_session(model_enum)
            
            if session is None:
                return None
            
            return {
                "timestamp": session.timestamp,
                "hyperparameters": session.hyperparameters,
                "performance_metrics": session.performance_metrics,
                "training_duration": session.training_duration_seconds,
                "success": session.success,
                "notes": session.notes
            }
            
        except ValueError:
            print(f"Unknown model type: {model_type}")
            return None

# Global instance for easy access
ml_tracker = MLTrainingTracker()

def get_ml_tracker() -> MLTrainingTracker:
    """Get the global ML training tracker instance."""
    return ml_tracker

# Convenience functions for common operations
def store_model_training(model_type: str, hyperparameters: Dict[str, Any], 
                        performance_metrics: Dict[str, float], model_path: str,
                        dataset_info: Dict[str, Any], notes: Optional[str] = None) -> bool:
    """
    Convenience function to store a completed training session.
    
    Args:
        model_type: Type of model (random_forest, gradient_boosting, ridge_regression)
        hyperparameters: Model configuration used
        performance_metrics: Training results
        model_path: Path to saved model
        dataset_info: Information about training data
        notes: Optional notes
        
    Returns:
        True if stored successfully
    """
    tracker = get_ml_tracker()
    session_id = tracker.start_training_session(model_type, hyperparameters, dataset_info)
    return tracker.complete_training_session(session_id, performance_metrics, model_path, True, notes)

def get_model_last_training(model_type: str) -> Optional[Dict[str, Any]]:
    """
    Get the last training session info for a specific model.
    
    Args:
        model_type: Model type to query
        
    Returns:
        Training information dictionary or None
    """
    return get_ml_tracker().get_last_training_info(model_type)

def clear_model_training_memory(model_type: str) -> bool:
    """
    Clear training memory for a specific model only.
    
    Args:
        model_type: Model type to clear
        
    Returns:
        True if cleared successfully
    """
    try:
        model_enum = ModelType(model_type)
        return get_ml_tracker().memory_buffer.clear_model_memory(model_enum)
    except ValueError:
        print(f"Unknown model type: {model_type}")
        return False