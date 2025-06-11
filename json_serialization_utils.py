"""
JSON Serialization Utilities
---------------------------
Comprehensive utilities for handling JSON serialization of complex data types
including boolean values, numpy types, and model configurations.
"""

import json
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Union

class SafeJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that safely handles all data types."""
    
    def default(self, o):
        """Convert non-serializable objects to serializable types."""
        
        # Handle numpy types
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, np.bool_):
            return bool(o)
        
        # Handle Python built-in types that might cause issues
        elif isinstance(o, bool):
            return o  # bool is JSON serializable in Python
        elif isinstance(o, (int, float, str)):
            return o
        elif isinstance(o, (list, tuple)):
            return list(o)
        elif isinstance(o, dict):
            return o
        
        # Handle datetime objects
        elif isinstance(o, datetime):
            return o.isoformat()
        
        # Handle class objects (model classes)
        elif hasattr(o, '__name__'):
            return o.__name__
        elif hasattr(o, '__class__') and hasattr(o.__class__, '__name__'):
            return o.__class__.__name__
        
        # Fallback to string representation
        return str(o)

def safe_json_dumps(data: Any, **kwargs) -> str:
    """Safely serialize data to JSON string."""
    return json.dumps(data, cls=SafeJSONEncoder, **kwargs)

def safe_json_loads(json_str: str) -> Any:
    """Safely deserialize JSON string to Python object."""
    return json.loads(json_str)

def sanitize_config_for_json(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize configuration dictionary for JSON serialization.
    Ensures all values are JSON-serializable.
    """
    sanitized = {}
    
    for key, value in config.items():
        # Handle different data types appropriately
        if isinstance(value, bool):
            sanitized[key] = value  # bool is JSON serializable
        elif isinstance(value, (int, float, str)):
            sanitized[key] = value
        elif isinstance(value, (list, tuple)):
            # Recursively sanitize list/tuple items
            sanitized[key] = [sanitize_value_for_json(item) for item in value]
        elif isinstance(value, dict):
            # Recursively sanitize nested dictionaries
            sanitized[key] = sanitize_config_for_json(value)
        elif hasattr(value, '__name__'):  # Class objects
            sanitized[key] = value.__name__
        elif hasattr(value, '__class__') and hasattr(value.__class__, '__name__'):
            sanitized[key] = value.__class__.__name__
        else:
            # Try to serialize, fallback to string if needed
            try:
                json.dumps(value)
                sanitized[key] = value
            except (TypeError, ValueError):
                sanitized[key] = str(value)
    
    return sanitized

def sanitize_value_for_json(value: Any) -> Any:
    """Sanitize a single value for JSON serialization."""
    if isinstance(value, bool):
        return value
    elif isinstance(value, (int, float, str)):
        return value
    elif isinstance(value, (list, tuple)):
        return [sanitize_value_for_json(item) for item in value]
    elif isinstance(value, dict):
        return sanitize_config_for_json(value)
    elif isinstance(value, np.integer):
        return int(value)
    elif isinstance(value, np.floating):
        return float(value)
    elif isinstance(value, np.bool_):
        return bool(value)
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif hasattr(value, '__name__'):
        return value.__name__
    elif hasattr(value, '__class__') and hasattr(value.__class__, '__name__'):
        return value.__class__.__name__
    else:
        try:
            json.dumps(value)
            return value
        except (TypeError, ValueError):
            return str(value)

def validate_json_serializable(data: Any, path: str = "root") -> List[str]:
    """
    Validate that data is JSON serializable and return any issues found.
    
    Args:
        data: Data to validate
        path: Current path in the data structure (for error reporting)
        
    Returns:
        List of error messages if any non-serializable data is found
    """
    errors = []
    
    try:
        json.dumps(data)
        return errors  # No errors, data is serializable
    except TypeError as e:
        if isinstance(data, dict):
            for key, value in data.items():
                sub_errors = validate_json_serializable(value, f"{path}.{key}")
                errors.extend(sub_errors)
        elif isinstance(data, (list, tuple)):
            for i, item in enumerate(data):
                sub_errors = validate_json_serializable(item, f"{path}[{i}]")
                errors.extend(sub_errors)
        else:
            errors.append(f"Non-serializable object at {path}: {type(data).__name__} - {str(e)}")
    
    return errors