"""
AutoML Tuning Demo Interface
---------------------------
Simplified demonstration of integrated hyperparameter tuning and experiment tracking.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
from typing import Dict, List, Any

from .ml_tuning_simplified import SimplifiedTuningWizard, create_tuning_config
from .experiment_tracker import get_experiment_tracker
from .storage import get_store

def render_page():
    """Main render function for the AutoML demo."""
    st.header("🧪 AutoML Tuning System")
    st.write("Automated hyperparameter optimization with comprehensive experiment tracking")
    
    # Load data
    df_history = get_store().latest()
    if df_history.empty:
        st.warning("No data available. Please upload data first.")
        return
    
    # Prepare data for ML
    try:
        df, X, y, feature_cols = prepare_ml_data(df_history)
        if X is None:
            st.error("Data preparation failed. Please check your dataset.")
            return
    except Exception as e:
        st.error(f"Error preparing data: {str(e)}")
        return
    
    # Create tabs for the demo
    tab1, tab2, tab3 = st.tabs(["🎯 Quick Tuning", "📊 Experiment Dashboard", "📈 Results Analysis"])
    
    with tab1:
        render_quick_tuning_tab(X, y, feature_cols)
    
    with tab2:
        render_experiment_dashboard()
    
    with tab3:
        render_results_analysis()

def prepare_ml_data(df_history):
    """Prepare data for machine learning."""
    try:
        # Use the same data preparation as ML Experimental
        from .ml_experimental import _prep
        return _prep(df_history)
    except:
        # Fallback preparation
        if 'draw_date' in df_history.columns:
            df_history['draw_date'] = pd.to_datetime(df_history['draw_date'])
            df_history = df_history.sort_values('draw_date')
        
        # Create features
        feature_cols = ['n1', 'n2', 'n3', 'n4', 'n5']
        if not all(col in df_history.columns for col in feature_cols):
            st.error("Required columns not found in dataset")
            return None, None, None, None
        
        X = df_history[feature_cols].values
        y = df_history['powerball'].values if 'powerball' in df_history.columns else df_history['n1'].values
        
        return df_history, X, y, feature_cols

def render_quick_tuning_tab(X, y, feature_cols):
    """Render the quick tuning demonstration tab."""
    st.subheader("Quick Hyperparameter Tuning Demo")
    
    with st.expander("📋 Experiment Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            experiment_name = st.text_input(
                "Experiment Name", 
                value=f"Demo_{datetime.now().strftime('%H%M%S')}",
                help="Name for this tuning experiment"
            )
            
            model_name = st.selectbox(
                "Select Model",
                options=["RandomForest", "GradientBoosting", "Ridge", "Lasso"],
                help="Choose the ML model to optimize"
            )
        
        with col2:
            search_strategy = st.selectbox(
                "Search Strategy",
                options=["random_search", "grid_search"],
                help="Algorithm for exploring hyperparameter space"
            )
            
            n_trials = st.slider(
                "Number of Trials",
                min_value=5,
                max_value=50,
                value=10,
                help="Number of hyperparameter combinations to try"
            )
    
    # Show dataset info
    st.subheader("📊 Dataset Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Samples", X.shape[0])
    with col2:
        st.metric("Features", X.shape[1])
    with col3:
        st.metric("Target Range", f"{y.min():.1f} - {y.max():.1f}")
    
    # Execute tuning
    if st.button("🚀 Start Quick Tuning Demo", type="primary", use_container_width=True):
        execute_demo_tuning(experiment_name, model_name, search_strategy, n_trials, X, y)

def execute_demo_tuning(experiment_name, model_name, search_strategy, n_trials, X, y):
    """Execute a demonstration tuning job."""
    
    # Show progress
    progress_placeholder = st.empty()
    
    with progress_placeholder:
        st.info("🏗️ Setting up hyperparameter tuning demonstration...")
    
    try:
        # Get experiment tracker
        tracker = get_experiment_tracker()
        
        # Create simplified wizard
        wizard = SimplifiedTuningWizard(experiment_tracker=tracker)
        
        # Create configuration
        config = create_tuning_config(
            experiment_name=experiment_name,
            model_name=model_name,
            search_strategy=search_strategy,
            n_trials=n_trials,
            cv_folds=3,  # Faster for demo
            objective_metric="neg_mean_absolute_error"
        )
        
        with progress_placeholder:
            st.info("🔬 Running hyperparameter optimization...")
        
        # Create progress bar
        progress_bar = st.progress(0)
        
        # Execute tuning job
        result = wizard.run_tuning_job(config, X, y)
        
        progress_bar.progress(100)
        
        # Display results
        with progress_placeholder:
            st.success(f"✅ Tuning completed! Found best configuration in {result.total_time:.2f} seconds")
        
        # Show detailed results
        display_demo_results(result)
        
    except Exception as e:
        with progress_placeholder:
            st.error(f"❌ Tuning failed: {str(e)}")

def display_demo_results(result):
    """Display comprehensive demo results."""
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
    
    # Best configuration
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
    if result.convergence_history:
        st.subheader("📈 Optimization Progress")
        
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
    trials_df = trials_df.sort_values("Objective", ascending=False)
    
    st.dataframe(trials_df, use_container_width=True)

def render_experiment_dashboard():
    """Render the experiment tracking dashboard."""
    st.subheader("📊 Experiment Tracking Dashboard")
    
    tracker = get_experiment_tracker()
    
    # Get experiment data
    all_runs = tracker.get_experiment_runs(limit=100)
    all_jobs = tracker.get_experiment_jobs(limit=50)
    
    if not all_runs:
        st.info("No experiments found yet. Run the Quick Tuning demo to create experiments.")
        return
    
    # Dashboard overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Experiments", len(set(run.experiment_name for run in all_runs)))
    
    with col2:
        st.metric("Total Jobs", len(all_jobs))
    
    with col3:
        st.metric("Total Trials", len(all_runs))
    
    with col4:
        if all_runs:
            avg_duration = np.mean([run.duration_seconds for run in all_runs if run.duration_seconds])
            st.metric("Avg Trial Time", f"{avg_duration:.2f}s")
    
    # Recent experiments
    st.subheader("🕒 Recent Experiments")
    
    if all_jobs:
        job_data = []
        for job in all_jobs[:10]:  # Show last 10
            job_data.append({
                "Experiment": job.experiment_name,
                "Status": job.status,
                "Start Time": job.start_time[:19] if job.start_time else "",
                "Trials": job.total_trials,
                "Job Type": job.job_type
            })
        
        jobs_df = pd.DataFrame(job_data)
        st.dataframe(jobs_df, use_container_width=True)
    
    # Performance trends
    if len(all_runs) > 1:
        st.subheader("📈 Performance Trends")
        
        # Create timeline of experiment performance
        run_data = []
        for run in all_runs:
            if run.metrics.get('objective_value'):
                run_data.append({
                    'timestamp': pd.to_datetime(run.timestamp),
                    'objective': run.metrics['objective_value'],
                    'experiment': run.experiment_name
                })
        
        if run_data:
            trend_df = pd.DataFrame(run_data)
            trend_df = trend_df.sort_values('timestamp')
            
            # Convert timestamp to string format for plotly
            trend_df['timestamp_str'] = trend_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            fig = px.scatter(
                trend_df, 
                x='timestamp_str', 
                y='objective',
                color='experiment',
                title="Experiment Performance Over Time",
                hover_data=['experiment']
            )
            
            fig.update_layout(
                xaxis_title="Time",
                yaxis_title="Objective Value",
                hovermode='closest'
            )
            
            st.plotly_chart(fig, use_container_width=True)

def render_results_analysis():
    """Render the results analysis tab."""
    st.subheader("📈 Experiment Results Analysis")
    
    tracker = get_experiment_tracker()
    all_runs = tracker.get_experiment_runs(limit=1000)
    
    if not all_runs:
        st.info("No experiment results available for analysis.")
        return
    
    # Experiment selection
    experiments = list(set(run.experiment_name for run in all_runs))
    
    if len(experiments) > 1:
        selected_experiment = st.selectbox("Select Experiment for Analysis", experiments)
        filtered_runs = [r for r in all_runs if r.experiment_name == selected_experiment]
    else:
        selected_experiment = experiments[0] if experiments else "No experiments"
        filtered_runs = all_runs
    
    if not filtered_runs:
        st.warning("No runs found for the selected experiment.")
        return
    
    # Analysis metrics
    objectives = [r.metrics.get('objective_value', 0) for r in filtered_runs if r.metrics.get('objective_value')]
    
    if objectives:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Runs", len(filtered_runs))
        
        with col2:
            st.metric("Best Score", f"{max(objectives):.4f}")
        
        with col3:
            st.metric("Mean Score", f"{np.mean(objectives):.4f}")
        
        with col4:
            st.metric("Std Score", f"{np.std(objectives):.4f}")
        
        # Distribution analysis
        st.subheader("📊 Score Distribution")
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Distribution", "Box Plot")
        )
        
        fig.add_trace(
            go.Histogram(x=objectives, nbinsx=15, name="Distribution"),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Box(y=objectives, name="Box Plot"),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Best runs comparison
        st.subheader("🏆 Top Performing Runs")
        
        # Sort runs by objective value
        top_runs = sorted(
            [r for r in filtered_runs if r.metrics.get('objective_value')], 
            key=lambda r: r.metrics['objective_value'], 
            reverse=True
        )[:5]
        
        if top_runs:
            comparison_data = []
            for i, run in enumerate(top_runs):
                row = {
                    "Rank": i + 1,
                    "Run ID": run.run_id[:8],
                    "Objective": run.metrics.get('objective_value', 0),
                    "CV Mean": run.metrics.get('cv_mean', 0),
                    "Duration": f"{run.duration_seconds:.2f}s"
                }
                row.update(run.hyperparameters)
                comparison_data.append(row)
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
    
    else:
        st.warning("No objective values found in the selected experiment runs.")