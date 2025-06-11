"""
ML Prediction Interface
======================
Enhanced UI for displaying and managing model-specific predictions with persistent storage.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from .persistent_model_predictions import get_prediction_manager, get_maintenance_manager

def render_prediction_display_interface():
    """Main interface for displaying model predictions with selection capabilities."""
    
    st.subheader("🎯 Enhanced Predictions")
    
    # Get prediction manager
    pm = get_prediction_manager()
    
    # Get all current predictions
    all_predictions = pm.get_all_current_predictions()
    
    if not all_predictions:
        st.info("""
        No predictions available yet. Train a model using the "Train & predict next draw" button
        to generate and store persistent predictions.
        """)
        return
    
    # Model selection interface
    available_models = list(all_predictions.keys())
    
    # Create columns for model selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_model = st.selectbox(
            "Select Model to View Predictions",
            options=available_models,
            help="Choose which model's predictions to display"
        )
    
    with col2:
        # Display last updated time for selected model
        if selected_model and selected_model in all_predictions:
            predictions = all_predictions[selected_model]
            if predictions:
                created_at = predictions[0]['created_at']
                st.metric("Last Updated", created_at.split('T')[0])
    
    # Display predictions for selected model
    if selected_model and selected_model in all_predictions:
        display_model_predictions(selected_model, all_predictions[selected_model])
    
    # Quick access tabs for all models
    st.markdown("---")
    st.subheader("Quick Model Comparison")
    
    if len(available_models) > 1:
        model_tabs = st.tabs([f"📊 {model}" for model in available_models])
        
        for i, model in enumerate(available_models):
            with model_tabs[i]:
                display_model_summary(model, all_predictions[model])
    else:
        # Single model display
        if available_models:
            display_model_summary(available_models[0], all_predictions[available_models[0]])

def display_model_predictions(model_name: str, predictions: List[Dict[str, Any]]):
    """Display detailed predictions for a specific model."""
    
    st.markdown(f"### {model_name} Predictions")
    
    # Model performance metrics
    if predictions and 'performance_metrics' in predictions[0]:
        metrics = predictions[0]['performance_metrics']
        
        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric("CV MAE Mean", f"{metrics.get('cv_mae_mean', 0):.3f}")
        with metric_cols[1]:
            st.metric("CV MAE Std", f"{metrics.get('cv_mae_std', 0):.3f}")
        with metric_cols[2]:
            st.metric("Best CV Score", f"{metrics.get('cv_score_best', 0):.3f}")
        with metric_cols[3]:
            st.metric("Worst CV Score", f"{metrics.get('cv_score_worst', 0):.3f}")
    
    # Create tabs for each prediction
    pred_tabs = st.tabs([f"Prediction {i+1}" for i in range(len(predictions))])
    
    for i, pred_tab in enumerate(pred_tabs):
        with pred_tab:
            prediction = predictions[i]
            display_individual_prediction(prediction, i+1)

def display_individual_prediction(prediction: Dict[str, Any], pred_number: int):
    """Display an individual prediction with detailed information."""
    
    white_numbers = prediction['white_numbers']
    powerball = prediction['powerball']
    probability = prediction['probability']
    
    # Main prediction display
    st.markdown(f"#### Prediction {pred_number}")
    
    # Number display
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # White numbers
        white_nums_str = ' '.join(f'`{w}`' for w in white_numbers)
        st.markdown(f"**White Numbers:** {white_nums_str}")
        
        # Powerball
        st.markdown(f"**Powerball:** `{powerball}`")
        
        # Probability
        st.markdown(f"**Probability Score:** {probability:.8f}")
    
    with col2:
        # Prediction summary
        st.metric("White Ball Range", f"{min(white_numbers)}-{max(white_numbers)}")
        st.metric("Number Sum", sum(white_numbers))
    
    # Additional details in expander
    with st.expander("Prediction Details"):
        # Hyperparameters used
        if 'hyperparameters' in prediction:
            st.subheader("Model Configuration")
            hyperparams = prediction['hyperparameters']
            
            param_df = pd.DataFrame([
                {'Parameter': k, 'Value': v} for k, v in hyperparams.items()
            ])
            st.dataframe(param_df, hide_index=True)
        
        # Features used
        if 'features_used' in prediction:
            st.subheader("Features Used in Training")
            features = prediction['features_used']
            
            # Group features by type
            feature_groups = {
                'Ordinal': [f for f in features if 'ordinal' in f.lower()],
                'Pair Features': [f for f in features if 'pair_' in f],
                'Statistical': [f for f in features if any(stat in f.lower() for stat in ['mean', 'std', 'sum', 'max', 'min'])],
                'Other': []
            }
            
            # Add remaining features to 'Other'
            used_features = set()
            for group_features in feature_groups.values():
                used_features.update(group_features)
            
            feature_groups['Other'] = [f for f in features if f not in used_features]
            
            # Display feature groups
            for group_name, group_features in feature_groups.items():
                if group_features:
                    st.write(f"**{group_name}:** {len(group_features)} features")
                    # Use details instead of nested expander
                    with st.container():
                        st.caption(f"{group_name} feature details:")
                        feature_text = ", ".join(group_features)
                        st.text(feature_text)

def display_model_summary(model_name: str, predictions: List[Dict[str, Any]]):
    """Display a summary view of model predictions."""
    
    if not predictions:
        st.info(f"No predictions available for {model_name}")
        return
    
    # Summary statistics with type conversion for bytes handling
    white_numbers_all = []
    powerballs = []
    probabilities = []
    
    for pred in predictions:
        # Handle white numbers - ensure they're integers
        white_nums = pred['white_numbers']
        if isinstance(white_nums, list):
            white_nums = [int(num) if not isinstance(num, bytes) else int.from_bytes(num, byteorder='little') for num in white_nums]
        white_numbers_all.append(white_nums)
        
        # Handle powerball - ensure it's an integer
        pb = pred['powerball']
        if isinstance(pb, bytes):
            pb = int.from_bytes(pb, byteorder='little')
        powerballs.append(int(pb))
        
        # Handle probability - ensure it's a float
        prob = pred['probability']
        if isinstance(prob, bytes):
            # For float bytes, use struct to unpack
            import struct
            prob = struct.unpack('<d', prob)[0]  # little-endian double
        probabilities.append(float(prob))
    
    # Calculate summary metrics
    all_white_numbers = [num for pred_nums in white_numbers_all for num in pred_nums]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Predictions", len(predictions))
        st.metric("Avg Probability", f"{sum(probabilities)/len(probabilities):.6f}")
    
    with col2:
        st.metric("White Number Range", f"{min(all_white_numbers)}-{max(all_white_numbers)}")
        st.metric("Powerball Range", f"{min(powerballs)}-{max(powerballs)}")
    
    with col3:
        st.metric("Most Common White", max(set(all_white_numbers), key=all_white_numbers.count))
        st.metric("Most Common Powerball", max(set(powerballs), key=powerballs.count))
    
    # Quick prediction list
    st.subheader("All Predictions")
    
    summary_data = []
    for i, (white_nums, pb, prob) in enumerate(zip(white_numbers_all, powerballs, probabilities)):
        summary_data.append({
            'Prediction': f"#{i+1}",
            'White Numbers': ' '.join(map(str, white_nums)),
            'Powerball': pb,
            'Probability': f"{prob:.6f}"
        })
    
    st.dataframe(pd.DataFrame(summary_data), hide_index=True)

def render_prediction_management_interface():
    """Interface for managing prediction storage and maintenance."""
    
    st.subheader("🔧 Prediction Management")
    
    pm = get_prediction_manager()
    mm = get_maintenance_manager()
    
    # Database statistics
    st.subheader("Database Statistics")
    stats = pm.get_database_stats()
    
    # Model statistics
    if stats['model_statistics']:
        stat_cols = st.columns(len(stats['model_statistics']))
        
        for i, (model, model_stats) in enumerate(stats['model_statistics'].items()):
            with stat_cols[i]:
                st.metric(
                    f"{model}",
                    f"Active: {model_stats['active']}",
                    f"Total: {model_stats['total']}"
                )
    
    # Overall statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Prediction Sets", stats['total_prediction_sets'])
    
    with col2:
        st.metric("Current Sets", stats['current_prediction_sets'])
    
    with col3:
        st.metric("Database Size", f"{stats['database_size_mb']} MB")
    
    # Maintenance operations
    st.subheader("Maintenance Operations")
    
    maintenance_cols = st.columns(3)
    
    with maintenance_cols[0]:
        if st.button("Run Integrity Checks"):
            with st.spinner("Running integrity checks..."):
                integrity_results = mm.run_data_integrity_checks()
                display_integrity_results(integrity_results)
    
    with maintenance_cols[1]:
        if st.button("Apply Retention Policy"):
            with st.spinner("Applying retention policy..."):
                retention_results = mm.apply_retention_policy()
                display_retention_results(retention_results)
    
    with maintenance_cols[2]:
        if st.button("Resolve Issues"):
            with st.spinner("Resolving integrity issues..."):
                resolution_results = mm.resolve_integrity_issues()
                display_resolution_results(resolution_results)
    
    # Configuration
    st.subheader("Configuration")
    
    config = mm.get_maintenance_config()
    
    col1, col2 = st.columns(2)
    
    with col1:
        current_limit = config['retention_limit']
        new_limit = st.number_input(
            "Retention Limit (prediction sets per model)",
            min_value=1,
            max_value=20,
            value=current_limit,
            help="Number of prediction sets to keep per model"
        )
        
        if new_limit != current_limit:
            if st.button("Update Retention Limit"):
                mm.update_retention_limit(new_limit)
                st.success(f"Updated retention limit to {new_limit}")
                st.rerun()
    
    with col2:
        st.info(f"""
        **Current Configuration:**
        - Retention Limit: {config['retention_limit']} sets per model
        - Database: {config['database_path']}
        """)

def display_integrity_results(results: Dict[str, Any]):
    """Display data integrity check results."""
    
    if not results['issues_found']:
        st.success("✅ Database integrity is excellent. No issues found.")
    else:
        st.warning(f"⚠️ Found {len(results['issues_found'])} integrity issues")
        
        for issue in results['issues_found']:
            severity_color = "🔴" if issue['severity'] == 'high' else "🟡"
            st.write(f"{severity_color} **{issue['type']}** - {issue.get('model', 'Unknown')}")
            
            if issue['type'] == 'multiple_current_sets':
                st.write(f"   Found {issue['count']} current sets (expected: 1)")
            elif issue['type'] == 'inconsistent_active_predictions':
                st.write(f"   Found {issue['active_count']} active predictions (expected: 5)")
    
    # Recommendations
    if results['recommendations']:
        st.subheader("Recommendations")
        for rec in results['recommendations']:
            st.info(rec)

def display_retention_results(results: Dict[str, Any]):
    """Display retention policy application results."""
    
    if results['sets_removed'] == 0:
        st.success("✅ All models within retention limit. No cleanup needed.")
    else:
        st.success(f"✅ Cleaned up {results['sets_removed']} old prediction sets")
        st.info(f"Removed {results['predictions_removed']} individual predictions")
        
        # Details by model
        for model_result in results['models_processed']:
            if model_result['sets_removed'] > 0:
                st.write(f"**{model_result['model']}:** Removed {model_result['sets_removed']} sets")

def display_resolution_results(results: Dict[str, Any]):
    """Display issue resolution results."""
    
    if not results['issues_resolved']:
        st.info("ℹ️ No integrity issues required automatic resolution.")
    else:
        st.success(f"✅ Resolved {len(results['issues_resolved'])} integrity issues")
        
        for resolution in results['issues_resolved']:
            st.write(f"**Fixed:** {resolution['type']} for {resolution.get('model', 'Unknown')}")
    
    if results['manual_intervention_required']:
        st.warning("⚠️ Some issues require manual intervention:")
        for issue in results['manual_intervention_required']:
            st.write(f"• {issue}")

def render_prediction_history_interface():
    """Interface for viewing prediction history and model performance over time."""
    
    st.subheader("📈 Prediction History")
    
    pm = get_prediction_manager()
    
    # Model selection
    models = ['Ridge Regression', 'Random Forest', 'Gradient Boosting']
    selected_model = st.selectbox("Select Model for History", models)
    
    # Get history
    history = pm.get_prediction_history(selected_model, limit=20)
    
    if not history:
        st.info(f"No prediction history available for {selected_model}")
        return
    
    # Convert to DataFrame for display
    history_df = pd.DataFrame(history)
    history_df['created_at'] = pd.to_datetime(history_df['created_at'])
    history_df = history_df.sort_values('created_at', ascending=False)
    
    # Display summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Prediction Sets", len(history))
    
    with col2:
        current_sets = len([h for h in history if h['is_current']])
        st.metric("Current Sets", current_sets)
    
    with col3:
        avg_duration = sum(h.get('training_duration', 0) for h in history) / len(history)
        st.metric("Avg Training Time", f"{avg_duration:.2f}s")
    
    # History timeline
    st.subheader("Training Timeline")
    
    # Create timeline chart
    timeline_data = []
    for hist in history:
        timeline_data.append({
            'Date': hist['created_at'],
            'Set ID': hist['set_id'],
            'Is Current': hist['is_current'],
            'Training Duration': hist.get('training_duration', 0),
            'Notes': hist.get('notes', '')
        })
    
    timeline_df = pd.DataFrame(timeline_data)
    
    if len(timeline_df) > 1:
        fig = px.scatter(
            timeline_df,
            x='Date',
            y='Training Duration',
            color='Is Current',
            size=[10 if current else 5 for current in timeline_df['Is Current']],
            hover_data=['Set ID', 'Notes'],
            title=f"{selected_model} Training History"
        )
        
        fig.update_layout(
            xaxis_title="Training Date",
            yaxis_title="Training Duration (seconds)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed history table
    st.subheader("Detailed History")
    
    display_df = timeline_df.copy()
    # Convert Date column to datetime before using .dt accessor
    display_df['Date'] = pd.to_datetime(display_df['Date'])
    display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d %H:%M')
    display_df['Is Current'] = display_df['Is Current'].apply(lambda x: '✅ Current' if x else '📊 Historical')
    display_df['Training Duration'] = display_df['Training Duration'].apply(lambda x: f"{x:.2f}s")
    
    st.dataframe(
        display_df,
        column_config={
            "Date": st.column_config.TextColumn("Training Date"),
            "Set ID": st.column_config.TextColumn("Prediction Set ID"),
            "Is Current": st.column_config.TextColumn("Status"),
            "Training Duration": st.column_config.TextColumn("Duration"),
            "Notes": st.column_config.TextColumn("Notes")
        },
        hide_index=True
    )