# Powerball Insights Backend Architecture Report
## v0.1.1 System Analysis

---

## Executive Summary

The Powerball Insights application employs a modular backend architecture with multiple specialized systems for data management, machine learning operations, and user interface support. This report identifies key modules, their interactions, and the unified data flow mechanisms.

---

## 1. Key Backend Modules

### 1.1 Data Storage & Management Layer
- **`core/storage.py`** - Versioned data store singleton with parquet file management
- **`core/ingest.py`** - Data ingestion, CSV processing, and manual entry interface
- **`core/data_maintenance.py`** - Data cleaning, validation, and optimization agent
- **SQLite Database** (`data/model_predictions.db`) - Persistent model predictions storage

### 1.2 Machine Learning Systems
- **`core/ml_experimental.py`** - Primary ML analysis interface with model training
- **`core/prediction_system.py`** - Legacy prediction system with accuracy tracking
- **`core/persistent_model_predictions.py`** - Modern prediction storage with SQLite
- **`core/ml_memory.py`** - ML training session tracking and memory management
- **`core/automl_demo.py`** - AutoML hyperparameter tuning demonstrations

### 1.3 Prediction Management
- **`core/prediction_storage_refactor.py`** - One-prediction-per-date enforcement
- **`core/ml_prediction_interface.py`** - Enhanced prediction display with type conversion
- **Database Maintenance Manager** - Retention policies and integrity checks

### 1.4 Analytics & Visualization
- **`core/combos.py`** - Combinatorial analysis (pairs, triplets)
- **`core/dow_analysis.py`** - Day-of-week statistical analysis
- **`core/csv_formatter.py`** - Data format detection and parsing

### 1.5 Shared Infrastructure
- **JSON Serialization Utilities** - Safe handling of complex data types
- **Experiment Tracking** - Performance metrics and model comparison
- **Error Handling & Logging** - Centralized error management

---

## 2. Data Flow Architecture

### 2.1 Primary Data Sources
```
Manual Entry → core/ingest.py → CSV Storage → core/storage.py (Singleton)
External CSV Upload → core/ingest.py → Validation → Storage Layer
Powerball Scraper → tools/powerball_scraper.py → CSV Update → Storage
```

### 2.2 ML Training Pipeline
```
Storage Layer → core/ml_experimental.py → Model Training → Predictions
                     ↓
Training Session → core/ml_memory.py → Session Tracking
                     ↓
Predictions → core/persistent_model_predictions.py → SQLite Database
```

### 2.3 Unified Data Update Flow (New Implementation)
```
"Save Results & Update History" Button:
├── Update Main Dataset (CSV + Storage Singleton)
├── Update Prediction System (Legacy joblib files)  
├── Update SQLite Database (Modern persistence)
└── Clear Session Cache → UI Refresh
```

---

## 3. Inter-Module Interactions

### 3.1 Storage Layer Dependencies
- **Primary Dataset Access**: All analysis modules depend on `core/storage.get_store().latest()`
- **Data Format**: Standardized DataFrame with columns: `draw_date`, `n1-n5`, `powerball`
- **Version Control**: Automatic versioning via parquet files with timestamp naming

### 3.2 ML System Integration
- **Model Training**: `core/ml_experimental.py` → Feature engineering → Multiple models
- **Prediction Storage**: Dual system (Legacy joblib + Modern SQLite)
- **Type Conversion**: Comprehensive bytes-to-proper-types handling in display layer

### 3.3 UI Support Systems
- **Quick Model Comparison**: `core/ml_prediction_interface.py` with bytes error fixes
- **Prediction History**: Timeline visualization with proper datetime handling
- **Data Maintenance**: Real-time integrity checks and cleanup operations

---

## 4. Database Architecture

### 4.1 SQLite Schema (data/model_predictions.db)
```sql
CREATE TABLE prediction_sets (
    set_id TEXT PRIMARY KEY,
    model_name TEXT,
    created_at TEXT,
    is_current BOOLEAN,
    total_predictions INTEGER,
    training_duration REAL,
    notes TEXT
);

CREATE TABLE model_predictions (
    model_name TEXT,
    prediction_id TEXT,
    prediction_set_id TEXT,
    white_numbers TEXT,  -- JSON array
    powerball INTEGER,
    probability REAL,
    features_used TEXT,  -- JSON array
    hyperparameters TEXT,  -- JSON object
    performance_metrics TEXT,  -- JSON object
    created_at TEXT,
    version INTEGER,
    is_active BOOLEAN
);
```

### 4.2 Data Integrity Enforcement
- **One Prediction Per Date**: Automatic duplicate detection and replacement
- **Type Validation**: Comprehensive conversion from bytes to proper data types
- **Retention Policies**: Configurable limits on historical prediction storage

---

## 5. Critical System Integrations

### 5.1 Bytes Type Conversion Layer
**Problem Solved**: SQLite storage was returning bytes objects instead of integers/floats
**Solution**: Comprehensive type conversion in `core/ml_prediction_interface.py`
```python
# Powerball bytes → integer
if isinstance(pb, bytes):
    pb = int.from_bytes(pb, byteorder='little')

# Probability bytes → float  
if isinstance(prob, bytes):
    prob = struct.unpack('<d', prob)[0]
```

### 5.2 Unified Data Flow Implementation
**Problem Solved**: Disconnected data systems (CSV vs ML predictions)
**Solution**: Unified update mechanism in `core/ml_experimental.py`
- Updates both main CSV dataset and prediction system simultaneously
- Ensures UI consistency across all modules
- Clears session state cache for immediate refresh

### 5.3 Error Resolution Framework
- **UnboundLocalError Fixes**: Proper import statements for shared utilities
- **DateTime Format Standardization**: Consistent YYYY-MM-DD format across systems
- **Variable Scope Management**: Local vs module-level import organization

---

## 6. Shared Resource Utilization

### 6.1 Storage Singleton Pattern
- **Access Method**: `from core.storage import get_store`
- **Thread Safety**: Single instance across entire application
- **Data Consistency**: Automatic synchronization between modules

### 6.2 Configuration Management
- **Model Parameters**: Centralized in respective ML modules
- **Database Paths**: Configurable file locations
- **Retention Policies**: Database maintenance manager settings

### 6.3 Logging & Error Tracking
- **Centralized Logging**: Python logging module with INFO/WARNING/ERROR levels
- **Performance Metrics**: Cross-validation scores and training duration tracking
- **User Feedback**: Streamlit success/error messages with detailed context

---

## 7. System Dependencies Map

```
core/storage.py (Central Data Hub)
    ↑
    ├── core/ingest.py (Data Input)
    ├── core/ml_experimental.py (ML Training)
    ├── core/combos.py (Analysis)
    └── core/dow_analysis.py (Statistics)

core/persistent_model_predictions.py (Modern Storage)
    ↑
    ├── core/ml_experimental.py (Prediction Generation)
    ├── core/ml_prediction_interface.py (Display Layer)
    └── Database Maintenance Manager (Integrity)

core/prediction_system.py (Legacy Support)
    ↑
    ├── core/ml_experimental.py (Backward Compatibility)
    └── Historical Prediction Tracking
```

---

## 8. Recommendations for Future Development

### 8.1 Architecture Improvements
1. **Consolidate Storage Systems**: Migrate fully from legacy joblib to SQLite
2. **API Layer**: Create standardized data access interface
3. **Event System**: Implement pub/sub for cross-module notifications

### 8.2 Performance Optimizations
1. **Caching Layer**: Implement Redis for frequently accessed predictions
2. **Database Indexing**: Add indexes on model_name and created_at columns
3. **Async Processing**: Background model training for better UI responsiveness

### 8.3 Monitoring & Observability
1. **Health Checks**: Automated system status monitoring
2. **Performance Metrics**: Training time and prediction accuracy tracking
3. **Data Quality Alerts**: Automated detection of data anomalies

---

## Conclusion

The Powerball Insights backend demonstrates a solid modular architecture with clear separation of concerns. The recent fixes addressed critical issues in data flow integration and type handling, establishing a unified system for data updates across all modules. The dual storage approach (CSV + SQLite) provides both backward compatibility and modern persistence capabilities, while comprehensive error handling ensures system reliability.

The architecture successfully supports complex ML operations, real-time data updates, and interactive user interfaces while maintaining data integrity and system performance.