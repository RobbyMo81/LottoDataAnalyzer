"""
Hyperparameter Tuning Wizard UI
------------------------------
Streamlit interface for automated hyperparameter optimization with
comprehensive experiment tracking integration.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import time
from pathlib import Path

# Import our tuning and tracking systems
from .ml_tuning import (
    HyperparameterTuningWizard, TuningConfiguration, ParameterSpace,
    SearchStrategy, OptimizationDirection, ParameterType,
    create_tuning_config, quick_random_search, quick_bayesian_search
)
from .experiment_tracker import (
    ExperimentTracker, get_experiment_tracker,
    log_experiment, get_best_run, compare_experiments
)
from .storage import get_store

# ML model imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LogisticRegression, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

# Available models for tuning
AVAILABLE_MODELS = {
    "Random Forest": RandomForestRegressor,
    "Gradient Boosting": GradientBoostingRegressor, 
    "Ridge Regression": Ridge,
    "Lasso Regression": Lasso,
    "Support Vector Regression": SVR,
    "Neural Network (MLP)": MLPRegressor
}

# Predefined parameter spaces for common models
MODEL_PARAMETER_SPACES = {
    "Random Forest": [
        {"name": "n_estimators", "param_type": "int", "low": 50, "high": 500, "step": 50},
        {"name": "max_depth", "param_type": "int", "low": 5, "high": 50, "step": 5},
        {"name": "min_samples_split", "param_type": "int", "low": 2, "high": 20},
        {"name": "min_samples_leaf", "param_type": "int", "low": 1, "high": 10},
        {"name": "max_features", "param_type": "categorical", "choices": ["auto", "sqrt", "log2"]}
    ],
    "Gradient Boosting": [
        {"name": "n_estimators", "param_type": "int", "low": 50, "high": 300, "step": 50},
        {"name": "learning_rate", "param_type": "float", "low": 0.01, "high": 0.3, "log_scale": True},
        {"name": "max_depth", "param_type": "int", "low": 3, "high": 15},
        {"name": "subsample", "param_type": "float", "low": 0.6, "high": 1.0},
        {"name": "min_samples_split", "param_type": "int", "low": 2, "high": 20}
    ],
    "Ridge Regression": [
        {"name": "alpha", "param_type": "float", "low": 0.001, "high": 100.0, "log_scale": True},
        {"name": "fit_intercept", "param_type": "boolean"},
        {"name": "solver", "param_type": "categorical", "choices": ["auto", "svd", "cholesky", "lsqr", "saga"]}
    ],
    "Lasso Regression": [
        {"name": "alpha", "param_type": "float", "low": 0.001, "high": 10.0, "log_scale": True},
        {"name": "fit_intercept", "param_type": "boolean"},
        {"name": "selection", "param_type": "categorical", "choices": ["cyclic", "random"]}
    ],
    "Support Vector Regression": [
        {"name": "C", "param_type": "float", "low": 0.1, "high": 100.0, "log_scale": True},
        {"name": "gamma", "param_type": "categorical", "choices": ["scale", "auto"]},
        {"name": "kernel", "param_type": "categorical", "choices": ["rbf", "linear", "poly"]},
        {"name": "epsilon", "param_type": "float", "low": 0.01, "high": 1.0}
    ],
    "Neural Network (MLP)": [
        {"name": "hidden_layer_sizes", "param_type": "categorical", "choices": [(50,), (100,), (50, 50), (100, 50), (100, 100)]},
        {"name": "alpha", "param_type": "float", "low": 0.0001, "high": 0.1, "log_scale": True},
        {"name": "learning_rate_init", "param_type": "float", "low": 0.001, "high": 0.1, "log_scale": True},
        {"name": "activation", "param_type": "categorical", "choices": ["relu", "tanh", "logistic"]}
    ]
}

OPTIMIZATION_METRICS = {
    "Mean Absolute Error": {"name": "neg_mean_absolute_error", "direction": "maximize"},
    "Mean Squared Error": {"name": "neg_mean_squared_error", "direction": "maximize"},
    "R² Score": {"name": "r2", "direction": "maximize"},
    "Accuracy": {"name": "accuracy", "direction": "maximize"},
    "F1 Score": {"name": "f1_weighted", "direction": "maximize"},
    "ROC AUC": {"name": "roc_auc", "direction": "maximize"}
}

def render_hyperparameter_tuning_wizard():
    """Render the main hyperparameter tuning wizard interface."""
    st.header("🧪 Automated Hyperparameter Tuning Wizard")
    st.write("Advanced hyperparameter optimization with comprehensive experiment tracking")
    
    # Load data
    df_history = get_store().latest()
    if df_history.empty:
        st.warning("No data available. Please upload data first.")
        return
    
    # Prepare data for ML
    try:
        from .ml_experimental import _prep
        df, X, y, feature_cols = _prep(df_history)
        if X is None:
            st.error("Data preparation failed. Please check your dataset.")
            return
    except Exception as e:
        st.error(f"Error preparing data: {str(e)}")
        return
    
    # Create main tabs
    main_tabs = st.tabs(["🎯 Tuning Wizard", "📊 Experiment Tracking", "📈 Analysis & Comparison"])
    
    with main_tabs[0]:
        render_tuning_wizard_tab(X, y, feature_cols)
    
    with main_tabs[1]:
        render_experiment_tracking_tab()
    
    with main_tabs[2]:
        render_analysis_comparison_tab()

def render_tuning_wizard_tab(X, y, feature_cols):
    """Render the hyperparameter tuning wizard configuration and execution tab."""
    st.subheader("Configure Hyperparameter Tuning Job")
    
    # Experiment configuration
    with st.expander("📋 Experiment Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            experiment_name = st.text_input(
                "Experiment Name", 
                value=f"HPTuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                help="Unique name for this tuning experiment"
            )
            
            model_name = st.selectbox(
                "Select Model",
                options=list(AVAILABLE_MODELS.keys()),
                help="Choose the ML model to optimize"
            )
            
            search_strategy = st.selectbox(
                "Search Strategy",
                options=["Random Search", "Grid Search", "Bayesian Optimization"],
                help="Algorithm for exploring hyperparameter space"
            )
        
        with col2:
            objective_metric = st.selectbox(
                "Optimization Metric",
                options=list(OPTIMIZATION_METRICS.keys()),
                help="Metric to optimize during tuning"
            )
            
            n_trials = st.number_input(
                "Number of Trials",
                min_value=5,
                max_value=1000,
                value=50,
                help="Maximum number of hyperparameter combinations to try"
            )
            
            cv_folds = st.number_input(
                "Cross-Validation Folds",
                min_value=3,
                max_value=10,
                value=5,
                help="Number of folds for cross-validation"
            )
    
    # Parameter space configuration
    with st.expander("⚙️ Hyperparameter Space Configuration", expanded=True):
        st.write(f"Configure search space for {model_name}")
        
        # Get predefined parameter space
        predefined_params = MODEL_PARAMETER_SPACES.get(model_name, [])
        
        # Allow users to customize parameter spaces
        custom_params = st.checkbox(
            "Customize Parameter Ranges",
            help="Modify the default parameter search space"
        )
        
        parameter_spaces = []
        
        if custom_params:
            st.write("**Custom Parameter Configuration:**")
            
            for i, param_def in enumerate(predefined_params):
                st.write(f"**{param_def['name']}** ({param_def['param_type']})")
                
                param_config = param_def.copy()
                
                if param_def['param_type'] in ['int', 'float']:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        param_config['low'] = st.number_input(
                            f"Min {param_def['name']}", 
                            value=float(param_def['low']),
                            key=f"min_{i}"
                        )
                    with col2:
                        param_config['high'] = st.number_input(
                            f"Max {param_def['name']}", 
                            value=float(param_def['high']),
                            key=f"max_{i}"
                        )
                    with col3:
                        if 'step' in param_def:
                            param_config['step'] = st.number_input(
                                f"Step {param_def['name']}", 
                                value=float(param_def.get('step', 1)),
                                key=f"step_{i}"
                            )
                
                elif param_def['param_type'] == 'categorical':
                    choices_str = st.text_input(
                        f"Choices for {param_def['name']} (comma-separated)",
                        value=", ".join(str(c) for c in param_def['choices']),
                        key=f"choices_{i}"
                    )
                    param_config['choices'] = [c.strip() for c in choices_str.split(',')]
                
                parameter_spaces.append(param_config)
                st.divider()
        else:
            parameter_spaces = predefined_params
            
            # Display default parameter space
            st.write("**Default Parameter Search Space:**")
            for param in parameter_spaces:
                if param['param_type'] in ['int', 'float']:
                    st.write(f"- **{param['name']}**: {param['low']} to {param['high']}")
                elif param['param_type'] == 'categorical':
                    st.write(f"- **{param['name']}**: {param['choices']}")
                elif param['param_type'] == 'boolean':
                    st.write(f"- **{param['name']}**: True/False")
    
    # Advanced settings
    with st.expander("🔧 Advanced Settings", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            timeout_minutes = st.number_input(
                "Timeout (minutes)", 
                min_value=1, 
                max_value=120, 
                value=30,
                help="Maximum time for tuning job"
            )
            
            random_state = st.number_input(
                "Random Seed", 
                min_value=0, 
                max_value=9999, 
                value=42,
                help="Seed for reproducibility"
            )
        
        with col2:
            parallel_jobs = st.number_input(
                "Parallel Jobs", 
                min_value=1, 
                max_value=8, 
                value=1,
                help="Number of parallel jobs (-1 for all cores)"
            )
            
            early_stopping = st.checkbox(
                "Enable Early Stopping",
                help="Stop unpromising trials early to save time"
            )
    
    # Execute tuning job
    st.divider()
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if st.button("🚀 Start Hyperparameter Tuning", type="primary", use_container_width=True):
            execute_tuning_job(
                experiment_name, model_name, search_strategy, objective_metric,
                parameter_spaces, n_trials, cv_folds, timeout_minutes,
                random_state, parallel_jobs, early_stopping, X, y
            )
    
    with col2:
        if st.button("💾 Save Configuration", use_container_width=True):
            save_tuning_configuration(
                experiment_name, model_name, search_strategy, 
                objective_metric, parameter_spaces, n_trials, cv_folds
            )
    
    with col3:
        if st.button("📁 Load Configuration", use_container_width=True):
            load_tuning_configuration()

def execute_tuning_job(experiment_name, model_name, search_strategy, objective_metric,
                      parameter_spaces, n_trials, cv_folds, timeout_minutes,
                      random_state, parallel_jobs, early_stopping, X, y):
    """Execute the hyperparameter tuning job."""
    
    # Show progress
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    with progress_placeholder:
        st.info("🏗️ Setting up hyperparameter tuning job...")
    
    try:
        # Get experiment tracker
        tracker = get_experiment_tracker()
        
        # Create tuning configuration
        strategy_map = {
            "Grid Search": SearchStrategy.GRID_SEARCH,
            "Random Search": SearchStrategy.RANDOM_SEARCH,
            "Bayesian Optimization": SearchStrategy.BAYESIAN_OPTIMIZATION
        }
        
        metric_info = OPTIMIZATION_METRICS[objective_metric]
        direction = OptimizationDirection.MAXIMIZE if metric_info["direction"] == "maximize" else OptimizationDirection.MINIMIZE
        
        config = TuningConfiguration(
            experiment_name=experiment_name,
            model_class=AVAILABLE_MODELS[model_name],
            parameter_spaces=[ParameterSpace(**p) for p in parameter_spaces],
            search_strategy=strategy_map[search_strategy],
            objective_metric=metric_info["name"],
            optimization_direction=direction,
            cv_folds=cv_folds,
            n_trials=n_trials,
            timeout_seconds=timeout_minutes * 60,
            n_jobs=parallel_jobs if parallel_jobs > 0 else -1,
            random_state=random_state,
            early_stopping=early_stopping
        )
        
        # Initialize wizard with tracker
        wizard = HyperparameterTuningWizard(experiment_tracker=tracker)
        
        with progress_placeholder:
            st.info("🔬 Running hyperparameter optimization...")
            
        # Create progress bar and status updates
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Execute tuning job
        start_time = time.time()
        
        # Mock progress updates (in real implementation, this would be handled by the wizard)
        result = wizard.run_tuning_job(config, X, y)
        
        elapsed_time = time.time() - start_time
        
        # Display results
        with progress_placeholder:
            st.success(f"✅ Tuning completed in {elapsed_time:.2f} seconds!")
        
        # Show detailed results
        display_tuning_results(result)
        
    except Exception as e:
        with progress_placeholder:
            st.error(f"❌ Tuning failed: {str(e)}")
        st.exception(e)

def display_tuning_results(result):
    """Display comprehensive tuning job results."""
    st.subheader("🎯 Tuning Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Trials", len(result.all_trials))
    
    with col2:
        st.metric("Best Score", f"{result.best_trial.objective_value:.4f}")
    
    with col3:
        st.metric("Total Time", f"{result.total_time:.2f}s")
    
    with col4:
        success_rate = len([t for t in result.all_trials if t.cv_scores]) / len(result.all_trials) * 100
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    # Best hyperparameters
    st.subheader("🏆 Best Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Best Hyperparameters:**")
        for param, value in result.best_trial.hyperparameters.items():
            st.write(f"- **{param}**: {value}")
    
    with col2:
        st.write("**Performance Metrics:**")
        st.write(f"- **CV Mean**: {result.best_trial.cv_mean:.4f}")
        st.write(f"- **CV Std**: {result.best_trial.cv_std:.4f}")
        st.write(f"- **Training Time**: {result.best_trial.training_time:.2f}s")
    
    # Convergence plot
    st.subheader("📈 Optimization Progress")
    
    if result.convergence_history:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, len(result.convergence_history) + 1)),
            y=result.convergence_history,
            mode='lines+markers',
            name='Best Score',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title="Convergence History",
            xaxis_title="Trial Number",
            yaxis_title="Best Objective Value",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Trial results table
    st.subheader("📋 All Trials")
    
    # Convert trials to DataFrame
    trials_data = []
    for trial in result.all_trials:
        row = {
            "Trial": trial.trial_number,
            "Objective": trial.objective_value,
            "CV Mean": trial.cv_mean,
            "CV Std": trial.cv_std,
            "Training Time": trial.training_time
        }
        row.update(trial.hyperparameters)
        trials_data.append(row)
    
    trials_df = pd.DataFrame(trials_data)
    
    # Sort by objective value
    trials_df = trials_df.sort_values("Objective", ascending=False)
    
    st.dataframe(trials_df, use_container_width=True)
    
    # Parameter importance analysis
    if len(result.all_trials) > 10:
        st.subheader("🔍 Parameter Importance Analysis")
        analyze_parameter_importance(result.all_trials)

def analyze_parameter_importance(trials):
    """Analyze and visualize parameter importance from trials."""
    # Collect all parameter values and objectives
    param_data = {}
    objectives = []
    
    for trial in trials:
        objectives.append(trial.objective_value)
        for param, value in trial.hyperparameters.items():
            if param not in param_data:
                param_data[param] = []
            param_data[param].append(value)
    
    # Calculate correlations for numerical parameters
    correlations = {}
    for param, values in param_data.items():
        if all(isinstance(v, (int, float)) for v in values):
            corr = np.corrcoef(values, objectives)[0, 1]
            if not np.isnan(corr):
                correlations[param] = abs(corr)
    
    if correlations:
        # Plot parameter importance
        params = list(correlations.keys())
        importance = list(correlations.values())
        
        fig = go.Figure(go.Bar(
            x=importance,
            y=params,
            orientation='h',
            marker_color='lightblue'
        ))
        
        fig.update_layout(
            title="Parameter Importance (Correlation with Objective)",
            xaxis_title="Absolute Correlation",
            yaxis_title="Parameters",
            height=max(300, len(params) * 40)
        )
        
        st.plotly_chart(fig, use_container_width=True)

def render_experiment_tracking_tab():
    """Render the experiment tracking and management tab."""
    st.subheader("📊 Experiment Tracking Dashboard")
    
    tracker = get_experiment_tracker()
    
    # Dashboard overview
    with st.expander("📈 Dashboard Overview", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        
        # Get summary statistics
        all_jobs = tracker.get_experiment_jobs(limit=1000)
        all_runs = tracker.get_experiment_runs(limit=1000)
        
        with col1:
            st.metric("Total Experiments", len(set(run.experiment_name for run in all_runs)))
        
        with col2:
            st.metric("Total Tuning Jobs", len(all_jobs))
        
        with col3:
            st.metric("Total Trials", len(all_runs))
        
        with col4:
            if all_runs:
                avg_duration = np.mean([run.duration_seconds for run in all_runs if run.duration_seconds])
                st.metric("Avg Trial Time", f"{avg_duration:.2f}s")
    
    # Experiment management tabs
    tracking_tabs = st.tabs(["🔍 Browse Experiments", "📊 Job Details", "🏆 Best Runs", "🔍 Search & Filter"])
    
    with tracking_tabs[0]:
        render_browse_experiments(tracker)
    
    with tracking_tabs[1]:
        render_job_details(tracker)
    
    with tracking_tabs[2]:
        render_best_runs(tracker)
    
    with tracking_tabs[3]:
        render_search_filter(tracker)

def render_browse_experiments(tracker):
    """Render the experiment browsing interface."""
    st.write("Browse all hyperparameter tuning experiments")
    
    # Get all jobs
    jobs = tracker.get_experiment_jobs(limit=100)
    
    if not jobs:
        st.info("No experiments found. Run the tuning wizard to create experiments.")
        return
    
    # Convert to DataFrame for display
    job_data = []
    for job in jobs:
        job_data.append({
            "Experiment Name": job.experiment_name,
            "Status": job.status,
            "Start Time": job.start_time[:19] if job.start_time else "",
            "Total Trials": job.total_trials,
            "Job Type": job.job_type,
            "Job ID": job.job_id
        })
    
    jobs_df = pd.DataFrame(job_data)
    
    # Display with selection
    selected_indices = st.dataframe(
        jobs_df,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row"
    )
    
    # Show details for selected job
    if selected_indices and selected_indices.selection.rows:
        selected_idx = selected_indices.selection.rows[0]
        selected_job = jobs[selected_idx]
        
        st.subheader(f"📋 Details: {selected_job.experiment_name}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Job Information:**")
            st.write(f"- **Status**: {selected_job.status}")
            st.write(f"- **Type**: {selected_job.job_type}")
            st.write(f"- **Start Time**: {selected_job.start_time}")
            st.write(f"- **Total Trials**: {selected_job.total_trials}")
        
        with col2:
            if selected_job.configuration:
                st.write("**Configuration:**")
                config = selected_job.configuration
                if isinstance(config, str):
                    config = json.loads(config)
                
                for key, value in config.items():
                    if key not in ['parameter_spaces', 'model_class']:
                        st.write(f"- **{key}**: {value}")
        
        # Show trials for this job
        if st.button("View All Trials", key=f"view_trials_{selected_job.job_id}"):
            trials = tracker.get_experiment_runs(parent_job_id=selected_job.job_id)
            if trials:
                render_trials_table(trials)

def render_job_details(tracker):
    """Render detailed job analysis interface."""
    st.write("Analyze individual tuning jobs in detail")
    
    # Job selection
    jobs = tracker.get_experiment_jobs(limit=100)
    if not jobs:
        st.info("No experiments available.")
        return
    
    job_names = [f"{job.experiment_name} ({job.job_id[:8]})" for job in jobs]
    selected_job_name = st.selectbox("Select Experiment Job", job_names)
    
    if selected_job_name:
        # Find selected job
        job_id = selected_job_name.split('(')[-1].rstrip(')')
        selected_job = next((j for j in jobs if j.job_id.startswith(job_id)), None)
        
        if selected_job:
            # Get trials for this job
            trials = tracker.get_experiment_runs(parent_job_id=selected_job.job_id)
            
            if trials:
                st.subheader(f"📊 Analysis: {selected_job.experiment_name}")
                
                # Summary statistics
                objectives = [t.metrics.get('objective_value', 0) for t in trials]
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Trials", len(trials))
                with col2:
                    st.metric("Best Score", f"{max(objectives):.4f}")
                with col3:
                    st.metric("Mean Score", f"{np.mean(objectives):.4f}")
                with col4:
                    st.metric("Std Score", f"{np.std(objectives):.4f}")
                
                # Detailed visualizations
                render_job_visualizations(trials)

def render_job_visualizations(trials):
    """Render detailed visualizations for a tuning job."""
    # Objective value distribution
    objectives = [t.metrics.get('objective_value', 0) for t in trials]
    
    fig1 = go.Figure()
    fig1.add_trace(go.Histogram(x=objectives, nbinsx=20, name="Objective Values"))
    fig1.update_layout(title="Distribution of Objective Values", xaxis_title="Objective Value", yaxis_title="Count")
    
    st.plotly_chart(fig1, use_container_width=True)
    
    # Parameter vs Objective scatter plots
    if len(trials) > 1:
        # Get all hyperparameters
        all_params = set()
        for trial in trials:
            all_params.update(trial.hyperparameters.keys())
        
        # Create subplot for each numerical parameter
        numerical_params = []
        for param in all_params:
            values = [trial.hyperparameters.get(param) for trial in trials]
            if all(isinstance(v, (int, float)) for v in values if v is not None):
                numerical_params.append(param)
        
        if numerical_params:
            n_params = len(numerical_params)
            cols = 2
            rows = (n_params + cols - 1) // cols
            
            fig2 = make_subplots(
                rows=rows, cols=cols,
                subplot_titles=[f"{param} vs Objective" for param in numerical_params]
            )
            
            for i, param in enumerate(numerical_params):
                row = i // cols + 1
                col = i % cols + 1
                
                param_values = [trial.hyperparameters.get(param) for trial in trials]
                
                fig2.add_trace(
                    go.Scatter(
                        x=param_values,
                        y=objectives,
                        mode='markers',
                        name=param,
                        showlegend=False
                    ),
                    row=row, col=col
                )
            
            fig2.update_layout(height=300 * rows, title="Hyperparameter vs Objective Analysis")
            st.plotly_chart(fig2, use_container_width=True)

def render_best_runs(tracker):
    """Render the best runs analysis interface."""
    st.write("Analyze the best performing experiments across all jobs")
    
    # Get all experiments
    all_runs = tracker.get_experiment_runs(limit=1000)
    
    if not all_runs:
        st.info("No runs available.")
        return
    
    # Group by experiment name
    experiments = list(set(run.experiment_name for run in all_runs))
    
    selected_experiment = st.selectbox("Select Experiment", ["All Experiments"] + experiments)
    
    metric_name = st.selectbox(
        "Optimization Metric",
        ["objective_value", "cv_mean", "training_time"],
        help="Metric to rank runs by"
    )
    
    maximize = st.checkbox("Maximize Metric", value=True)
    
    # Filter and sort runs
    filtered_runs = all_runs
    if selected_experiment != "All Experiments":
        filtered_runs = [r for r in all_runs if r.experiment_name == selected_experiment]
    
    # Sort by selected metric
    runs_with_metric = [r for r in filtered_runs if metric_name in r.metrics]
    runs_with_metric.sort(key=lambda r: r.metrics[metric_name], reverse=maximize)
    
    # Display top runs
    top_n = st.slider("Number of top runs to show", 5, 50, 10)
    top_runs = runs_with_metric[:top_n]
    
    if top_runs:
        # Create comparison DataFrame
        comparison_df = tracker.compare_runs([run.run_id for run in top_runs])
        
        st.subheader(f"🏆 Top {len(top_runs)} Runs")
        st.dataframe(comparison_df, use_container_width=True)
        
        # Visualization
        if len(top_runs) > 1:
            metrics = [run.metrics[metric_name] for run in top_runs]
            run_names = [f"Run {i+1}" for i in range(len(top_runs))]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(x=run_names, y=metrics, name=metric_name))
            fig.update_layout(title=f"Top Runs by {metric_name}", xaxis_title="Run", yaxis_title=metric_name)
            
            st.plotly_chart(fig, use_container_width=True)

def render_search_filter(tracker):
    """Render the advanced search and filtering interface."""
    st.write("Search and filter experiments with advanced criteria")
    
    # Search form
    with st.form("search_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            experiment_name_filter = st.text_input(
                "Experiment Name Contains",
                help="Filter by experiment name substring"
            )
            
            start_date = st.date_input(
                "Start Date",
                value=datetime.now().date() - timedelta(days=30)
            )
        
        with col2:
            metric_filter = st.selectbox(
                "Metric to Filter",
                ["objective_value", "cv_mean", "training_time", "cv_std"]
            )
            
            end_date = st.date_input(
                "End Date",
                value=datetime.now().date()
            )
        
        metric_min = st.number_input("Minimum Metric Value", value=0.0)
        metric_max = st.number_input("Maximum Metric Value", value=1.0)
        
        search_submitted = st.form_submit_button("🔍 Search")
    
    if search_submitted:
        # Build search query
        search_query = {}
        
        if experiment_name_filter:
            search_query["experiment_name"] = experiment_name_filter
        
        if metric_filter and (metric_min != 0.0 or metric_max != 1.0):
            search_query["metric_ranges"] = {
                metric_filter: (metric_min, metric_max)
            }
        
        search_query["date_range"] = (
            datetime.combine(start_date, datetime.min.time()),
            datetime.combine(end_date, datetime.max.time())
        )
        
        # Execute search
        results = tracker.search_runs(search_query, limit=100)
        
        if results:
            st.subheader(f"🔍 Search Results ({len(results)} runs found)")
            
            # Display results table
            result_data = []
            for run in results:
                result_data.append({
                    "Experiment": run.experiment_name,
                    "Run ID": run.run_id[:8],
                    "Timestamp": run.timestamp[:19],
                    "Objective": run.metrics.get('objective_value', 'N/A'),
                    "CV Mean": run.metrics.get('cv_mean', 'N/A'),
                    "Duration": f"{run.duration_seconds:.2f}s"
                })
            
            results_df = pd.DataFrame(result_data)
            st.dataframe(results_df, use_container_width=True)
            
            # Export options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📊 Compare Selected Runs"):
                    if len(results) >= 2:
                        comparison = tracker.compare_runs([r.run_id for r in results[:10]])
                        st.dataframe(comparison)
            
            with col2:
                if st.button("📁 Export to CSV"):
                    try:
                        export_path = tracker.export_experiment_data(
                            results[0].experiment_name if results else "search_results",
                            format="csv"
                        )
                        st.success(f"Exported to: {export_path}")
                    except Exception as e:
                        st.error(f"Export failed: {str(e)}")
            
            with col3:
                if st.button("📋 Generate Report"):
                    generate_experiment_report(results)
        else:
            st.info("No runs found matching the search criteria.")

def render_analysis_comparison_tab():
    """Render the analysis and comparison tab."""
    st.subheader("📈 Advanced Analysis & Comparison")
    
    tracker = get_experiment_tracker()
    
    analysis_tabs = st.tabs(["📊 Multi-Experiment Comparison", "🔬 Statistical Analysis", "📈 Trend Analysis"])
    
    with analysis_tabs[0]:
        render_multi_experiment_comparison(tracker)
    
    with analysis_tabs[1]:
        render_statistical_analysis(tracker)
    
    with analysis_tabs[2]:
        render_trend_analysis(tracker)

def render_multi_experiment_comparison(tracker):
    """Render multi-experiment comparison interface."""
    st.write("Compare performance across different experiments and configurations")
    
    # Get all experiments
    all_runs = tracker.get_experiment_runs(limit=1000)
    experiments = list(set(run.experiment_name for run in all_runs))
    
    if len(experiments) < 2:
        st.info("Need at least 2 experiments for comparison.")
        return
    
    # Experiment selection
    selected_experiments = st.multiselect(
        "Select Experiments to Compare",
        experiments,
        default=experiments[:min(3, len(experiments))]
    )
    
    if len(selected_experiments) >= 2:
        # Comparison analysis
        comparison_data = []
        
        for exp_name in selected_experiments:
            exp_runs = [r for r in all_runs if r.experiment_name == exp_name]
            
            if exp_runs:
                objectives = [r.metrics.get('objective_value', 0) for r in exp_runs]
                
                comparison_data.append({
                    "Experiment": exp_name,
                    "Total Runs": len(exp_runs),
                    "Best Score": max(objectives),
                    "Mean Score": np.mean(objectives),
                    "Std Score": np.std(objectives),
                    "Success Rate": len([r for r in exp_runs if r.status == 'completed']) / len(exp_runs) * 100
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Display comparison table
        st.subheader("📊 Experiment Comparison")
        st.dataframe(comparison_df, use_container_width=True)
        
        # Visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Best Scores", "Mean Scores", "Total Runs", "Success Rates"),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        experiments_short = [exp[:20] + "..." if len(exp) > 20 else exp for exp in selected_experiments]
        
        fig.add_trace(go.Bar(x=experiments_short, y=comparison_df["Best Score"], name="Best Score"), row=1, col=1)
        fig.add_trace(go.Bar(x=experiments_short, y=comparison_df["Mean Score"], name="Mean Score"), row=1, col=2)
        fig.add_trace(go.Bar(x=experiments_short, y=comparison_df["Total Runs"], name="Total Runs"), row=2, col=1)
        fig.add_trace(go.Bar(x=experiments_short, y=comparison_df["Success Rate"], name="Success Rate"), row=2, col=2)
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

def render_statistical_analysis(tracker):
    """Render statistical analysis interface."""
    st.write("Perform statistical analysis on experiment results")
    
    # Get experiment data
    all_runs = tracker.get_experiment_runs(limit=1000)
    experiments = list(set(run.experiment_name for run in all_runs))
    
    if not experiments:
        st.info("No experiments available for analysis.")
        return
    
    selected_experiment = st.selectbox("Select Experiment for Analysis", experiments)
    
    if selected_experiment:
        exp_runs = [r for r in all_runs if r.experiment_name == selected_experiment]
        objectives = [r.metrics.get('objective_value', 0) for r in exp_runs]
        
        if objectives:
            # Statistical summary
            st.subheader("📊 Statistical Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Count", len(objectives))
                st.metric("Mean", f"{np.mean(objectives):.4f}")
            
            with col2:
                st.metric("Median", f"{np.median(objectives):.4f}")
                st.metric("Std Dev", f"{np.std(objectives):.4f}")
            
            with col3:
                st.metric("Min", f"{np.min(objectives):.4f}")
                st.metric("Max", f"{np.max(objectives):.4f}")
            
            with col4:
                st.metric("Range", f"{np.max(objectives) - np.min(objectives):.4f}")
                st.metric("CV", f"{np.std(objectives) / np.mean(objectives):.4f}")
            
            # Distribution analysis
            st.subheader("📈 Distribution Analysis")
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Histogram", "Box Plot")
            )
            
            fig.add_trace(go.Histogram(x=objectives, nbinsx=20, name="Distribution"), row=1, col=1)
            fig.add_trace(go.Box(y=objectives, name="Box Plot"), row=1, col=2)
            
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

def render_trend_analysis(tracker):
    """Render trend analysis interface."""
    st.write("Analyze trends and patterns over time")
    
    # Get all runs with timestamps
    all_runs = tracker.get_experiment_runs(limit=1000)
    
    if not all_runs:
        st.info("No runs available for trend analysis.")
        return
    
    # Convert to DataFrame for easier analysis
    trend_data = []
    for run in all_runs:
        try:
            timestamp = datetime.fromisoformat(run.timestamp)
            trend_data.append({
                "timestamp": timestamp,
                "experiment": run.experiment_name,
                "objective": run.metrics.get('objective_value', 0),
                "duration": run.duration_seconds
            })
        except:
            continue
    
    if not trend_data:
        st.info("No valid timestamp data available.")
        return
    
    trend_df = pd.DataFrame(trend_data)
    
    # Time-based analysis
    st.subheader("⏰ Temporal Trends")
    
    # Group by day
    trend_df['date'] = trend_df['timestamp'].dt.date
    daily_stats = trend_df.groupby('date').agg({
        'objective': ['count', 'mean', 'max'],
        'duration': 'mean'
    }).round(4)
    
    # Flatten column names
    daily_stats.columns = ['_'.join(col).strip() for col in daily_stats.columns]
    daily_stats = daily_stats.reset_index()
    
    # Plot trends
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Daily Experiment Count", "Daily Mean Performance", "Daily Best Performance", "Daily Mean Duration")
    )
    
    fig.add_trace(go.Scatter(x=daily_stats['date'], y=daily_stats['objective_count'], mode='lines+markers', name="Count"), row=1, col=1)
    fig.add_trace(go.Scatter(x=daily_stats['date'], y=daily_stats['objective_mean'], mode='lines+markers', name="Mean"), row=1, col=2)
    fig.add_trace(go.Scatter(x=daily_stats['date'], y=daily_stats['objective_max'], mode='lines+markers', name="Max"), row=2, col=1)
    fig.add_trace(go.Scatter(x=daily_stats['date'], y=daily_stats['duration_mean'], mode='lines+markers', name="Duration"), row=2, col=2)
    
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

def render_trials_table(trials):
    """Render a table of trials with sorting and filtering."""
    st.subheader("📋 Trial Details")
    
    # Convert to DataFrame
    trial_data = []
    for trial in trials:
        row = {
            "Run ID": trial.run_id[:8],
            "Status": trial.status,
            "Objective": trial.metrics.get('objective_value', 'N/A'),
            "CV Mean": trial.metrics.get('cv_mean', 'N/A'),
            "CV Std": trial.metrics.get('cv_std', 'N/A'),
            "Duration": f"{trial.duration_seconds:.2f}s",
            "Timestamp": trial.timestamp[:19]
        }
        
        # Add hyperparameters
        for param, value in trial.hyperparameters.items():
            row[f"param_{param}"] = value
        
        trial_data.append(row)
    
    trials_df = pd.DataFrame(trial_data)
    
    # Sort options
    sort_column = st.selectbox("Sort by", trials_df.columns.tolist())
    sort_ascending = st.checkbox("Ascending", value=False)
    
    sorted_df = trials_df.sort_values(sort_column, ascending=sort_ascending)
    
    st.dataframe(sorted_df, use_container_width=True)

def save_tuning_configuration(experiment_name, model_name, search_strategy, 
                            objective_metric, parameter_spaces, n_trials, cv_folds):
    """Save tuning configuration for later use."""
    config = {
        "experiment_name": experiment_name,
        "model_name": model_name,
        "search_strategy": search_strategy,
        "objective_metric": objective_metric,
        "parameter_spaces": parameter_spaces,
        "n_trials": n_trials,
        "cv_folds": cv_folds,
        "saved_at": datetime.now().isoformat()
    }
    
    # Save to file
    config_dir = Path("data/tuning_configs")
    config_dir.mkdir(parents=True, exist_ok=True)
    
    config_file = config_dir / f"{experiment_name.replace(' ', '_')}.json"
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    st.success(f"Configuration saved to {config_file}")

def load_tuning_configuration():
    """Load a previously saved tuning configuration."""
    config_dir = Path("data/tuning_configs")
    
    if not config_dir.exists():
        st.warning("No saved configurations found.")
        return
    
    config_files = list(config_dir.glob("*.json"))
    
    if not config_files:
        st.warning("No saved configurations found.")
        return
    
    selected_config = st.selectbox(
        "Select Configuration to Load",
        [f.stem for f in config_files]
    )
    
    if selected_config and st.button("Load Configuration"):
        config_file = config_dir / f"{selected_config}.json"
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        st.success("Configuration loaded! Update the form fields above.")
        st.json(config)

def generate_experiment_report(runs):
    """Generate a comprehensive experiment report."""
    st.subheader("📋 Experiment Report")
    
    if not runs:
        st.warning("No runs to analyze.")
        return
    
    # Summary statistics
    objectives = [r.metrics.get('objective_value', 0) for r in runs]
    durations = [r.duration_seconds for r in runs]
    
    report = f"""
    ## Experiment Analysis Report
    
    **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    ### Summary Statistics
    - **Total Runs:** {len(runs)}
    - **Best Performance:** {max(objectives):.4f}
    - **Average Performance:** {np.mean(objectives):.4f}
    - **Performance Std:** {np.std(objectives):.4f}
    - **Total Compute Time:** {sum(durations):.2f} seconds
    - **Average Run Time:** {np.mean(durations):.2f} seconds
    
    ### Top 5 Runs
    """
    
    # Sort runs by objective
    sorted_runs = sorted(runs, key=lambda r: r.metrics.get('objective_value', 0), reverse=True)
    
    for i, run in enumerate(sorted_runs[:5]):
        report += f"""
    **Rank {i+1}:**
    - Run ID: {run.run_id[:8]}
    - Objective: {run.metrics.get('objective_value', 'N/A'):.4f}
    - Parameters: {run.hyperparameters}
        """
    
    st.markdown(report)
    
    # Download button for report
    if st.button("💾 Download Report"):
        report_file = f"experiment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        st.download_button(
            label="Download Report",
            data=report,
            file_name=report_file,
            mime="text/markdown"
        )

# Main render function
def render_page():
    """Main render function for the hyperparameter tuning wizard."""
    render_hyperparameter_tuning_wizard()