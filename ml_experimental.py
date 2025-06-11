# core/ml_experimental.py  ────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import math
import datetime
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score, KFold
from itertools import combinations
from .storage import get_store
from .modernized_prediction_system import get_enhanced_predictions, update_model_with_new_draw
from .ml_memory import get_ml_tracker, store_model_training, get_model_last_training
from .persistent_model_predictions import get_prediction_manager, get_maintenance_manager
from .ml_prediction_interface import (
    render_prediction_display_interface, 
    render_prediction_management_interface,
    render_prediction_history_interface
)

DISCLAIMER = (
    "🔮 **Experimental Only** – Powerball is designed to be random.  "
    "No model can *predict* winning numbers.  This page exists for educational "
    "feature-engineering and model-evaluation practice, **not** wagering advice."
)

WHITE_MIN, WHITE_MAX = 1, 69
PB_MIN, PB_MAX = 1, 26
TOP_PAIR_COUNT = 10  # number of top co-occurring pairs to include as features

# Dictionary of available models
MODELS = {
    "Random Forest": RandomForestRegressor,
    "Gradient Boosting": GradientBoostingRegressor,
    "Ridge Regression": Ridge
}

# ─────────────────────────────────────────────────────────────────────────────
def _prep(df: pd.DataFrame):
    """Return (df, X, y) where X includes ordinal and top-pair binary features."""
    try:
        df = df.sort_values("draw_date").reset_index(drop=True)
        df["ordinal"] = np.arange(len(df))

        # Identify top co-occurring white-ball pairs (more efficient approach)
        # Convert the white ball columns to a set-per-row for faster processing
        white_sets = []
        for _, row in df.iterrows():
            white_sets.append(set([row.n1, row.n2, row.n3, row.n4, row.n5]))
        
        # Find all pairs
        all_pairs = []
        for white_set in white_sets:
            all_pairs.extend(combinations(sorted(white_set), 2))
        
        pair_counts = pd.Series(all_pairs).value_counts().head(TOP_PAIR_COUNT)
        top_pairs = [tuple(pair) for pair in pair_counts.index]

        # Create binary features for each top pair (vectorized approach)
        feature_cols = ["ordinal"]
        for a, b in top_pairs:
            col = f"pair_{a}_{b}"
            # Vectorized approach for pair detection
            contains_a = df[["n1", "n2", "n3", "n4", "n5"]].eq(a).any(axis=1)
            contains_b = df[["n1", "n2", "n3", "n4", "n5"]].eq(b).any(axis=1)
            df[col] = (contains_a & contains_b).astype(int)
            feature_cols.append(col)

        # Prepare feature matrix X and target matrix y
        X = df[feature_cols].values
        y = df[["n1", "n2", "n3", "n4", "n5", "powerball"]]
        
        return df, X, y, feature_cols
        
    except Exception as e:
        st.error(f"Error in data preparation: {str(e)}")
        return None, None, None, None


def _sanitize_numbers(nums):
    """Ensure 5 unique white balls w/in 1-69 and one PB 1-26; return (white, pb)."""
    try:
        *white_floats, pb_float = nums
        white = [int(round(x)) for x in white_floats]
        white = [min(max(WHITE_MIN, w), WHITE_MAX) for w in white]
        pb = int(round(pb_float))
        pb = min(max(PB_MIN, pb), PB_MAX)
        
        # Deduplicate keeping original order as much as possible
        seen = set()
        unique_white = []
        for w in white:
            while w in seen:
                w = (w % WHITE_MAX) + 1  # Wrap around more elegantly
            seen.add(w)
            unique_white.append(w)
        
        # Ensure exactly 5 white balls
        while len(unique_white) < 5:
            for i in range(WHITE_MIN, WHITE_MAX + 1):
                if i not in seen:
                    unique_white.append(i)
                    seen.add(i)
                    break
                    
        unique_white.sort()  # Convention is to present sorted white balls
        return unique_white, pb
        
    except Exception as e:
        st.error(f"Error sanitizing numbers: {str(e)}")
        return [1, 2, 3, 4, 5], 1  # Fallback to default values





def render_page():
    st.header("🧪 Experimental ML Analysis")
    st.info(DISCLAIMER)

    # Load history
    from .storage import get_store
    df_history = get_store().latest()
    if df_history.empty:
        st.warning("No draw history available. Please ingest or upload data first.")
        return

    # --- Train & Predict + Post-Analysis ---
    # Model selection and hyperparameters in sidebar
    st.sidebar.subheader("Model Configuration")
    model_name = st.sidebar.selectbox("Select model", list(MODELS.keys()))
    
    # Dynamic hyperparameters based on selected model
    hyperparams = {}
    if model_name == "Random Forest":
        hyperparams["n_estimators"] = st.sidebar.slider("Trees (n_estimators)", 100, 1000, 400, step=100)
        hyperparams["max_depth"] = st.sidebar.slider("Max depth", 5, 50, 20)
    elif model_name == "Gradient Boosting":
        hyperparams["n_estimators"] = st.sidebar.slider("Boosting stages", 50, 500, 100, step=50)
        hyperparams["learning_rate"] = st.sidebar.slider("Learning rate", 0.01, 0.2, 0.1, step=0.01)
    elif model_name == "Ridge Regression":
        hyperparams["alpha"] = st.sidebar.slider("Regularization strength (alpha)", 0.1, 10.0, 1.0, step=0.1)
    
    # New Draw Results section for updating prediction history
    st.sidebar.divider()
    st.sidebar.subheader("New Draw Results")
    
    # Create expander to keep sidebar clean
    with st.sidebar.expander("Enter actual draw results", expanded=False):
        st.caption("Enter recent draw results to update prediction history")
        
        # White balls input (5 numbers)
        white_cols = st.columns(5)
        white_balls = []
        for i, col in enumerate(white_cols):
            with col:
                num = st.number_input(f"White #{i+1}", min_value=1, max_value=69, 
                                     value=1, step=1, key=f"actual_white_{i}")
                white_balls.append(num)
        
        # Powerball input
        pb = st.number_input("Powerball", min_value=1, max_value=26, value=1, step=1)
        
        # Draw date input
        draw_date = st.date_input("Draw Date", value=datetime.datetime.now().date())
        
        # Update button
        if st.button("Save Results & Update History"):
            try:
                # Format the draw result
                new_draw = {
                    'white_numbers': sorted(white_balls),
                    'powerball': pb,
                    'draw_date': draw_date.strftime('%Y-%m-%d')
                }
                
                # UNIFIED DATA UPDATE: Update both main dataset AND prediction system
                from .storage import get_store
                from .modernized_prediction_system import ModernizedPredictionSystem, update_model_with_new_draw
                import pandas as pd
                
                # Step 1: Update main dataset (same as manual entry in ingest.py)
                current_df = get_store().latest()
                if current_df.empty:
                    # Load from CSV if store is empty
                    from .ingest import _load_default_csv
                    current_df = _load_default_csv()
                    if current_df is None:
                        current_df = pd.DataFrame(columns=['draw_date', 'n1', 'n2', 'n3', 'n4', 'n5', 'powerball'])
                
                # Check if date already exists
                date_str = str(draw_date)
                date_exists = False
                if not current_df.empty and 'draw_date' in current_df.columns:
                    if date_str in current_df['draw_date'].astype(str).values:
                        date_exists = True
                
                dataset_updated = False
                prediction_updated = False
                
                if not date_exists:
                    # Add new row to main dataset
                    new_row = {
                        'draw_date': date_str,
                        'n1': white_balls[0],
                        'n2': white_balls[1], 
                        'n3': white_balls[2],
                        'n4': white_balls[3],
                        'n5': white_balls[4],
                        'powerball': pb
                    }
                    
                    # Add to dataframe
                    new_df = pd.concat([current_df, pd.DataFrame([new_row])], ignore_index=True)
                    
                    # Sort by date (newest first)
                    new_df['draw_date'] = pd.to_datetime(new_df['draw_date'])
                    new_df = new_df.sort_values('draw_date', ascending=False)
                    new_df['draw_date'] = new_df['draw_date'].dt.strftime('%Y-%m-%d')
                    
                    # Save to CSV and update storage
                    from .ingest import DATA_PATH
                    DATA_PATH.parent.mkdir(exist_ok=True, parents=True)
                    new_df.to_csv(DATA_PATH, index=False)
                    get_store().set_latest(new_df)
                    
                    # Update df_history for immediate use
                    df_history = new_df.copy()
                    dataset_updated = True
                
                # Step 2: Update modernized prediction system with the updated dataset
                try:
                    modernized_system = ModernizedPredictionSystem(new_df)
                    before_predictions = modernized_system.get_prediction_history()
                    before_count = len(before_predictions)
                    modernized_system.update_with_new_draw(new_draw)
                    after_predictions = modernized_system.get_prediction_history()
                    after_count = len(after_predictions)
                    
                    if after_count >= before_count:  # SQLite storage may not increment count
                        prediction_updated = True
                    
                except Exception as e1:
                    # Try alternative method
                    try:
                        update_model_with_new_draw(new_draw)
                        prediction_updated = True
                    except Exception as e2:
                        st.error(f"Failed to update prediction system: {str(e1)} | {str(e2)}")
                
                # Clear any cached data to ensure UI refreshes
                if 'df_history' in st.session_state:
                    del st.session_state['df_history']
                
                # Display comprehensive success/failure message
                if dataset_updated and prediction_updated:
                    st.success(f"""
                    ✅ **Complete Data Update Successful!**
                    
                    **Main Dataset Updated:**
                    - Added new draw result to CSV file
                    - Updated storage system
                    - All UI displays will now show the new data
                    
                    **Prediction System Updated:**
                    - ML models updated with new draw result
                    - Prediction accuracy tracking updated
                    - Model performance metrics recalculated
                    
                    **Draw Details:**
                    - White Balls: {sorted(white_balls)}
                    - Powerball: {pb}
                    - Date: {draw_date.strftime('%Y-%m-%d')}
                    
                    Navigate to other tabs to see the updated data reflected throughout the application.
                    """)
                elif dataset_updated:
                    st.success(f"""
                    ✅ **Main Dataset Updated Successfully**
                    
                    The new draw result has been added to the main dataset. 
                    All UI displays including "Last 5 Actual Draws" will now show the updated data.
                    
                    ⚠️ Prediction system update incomplete - some ML features may not reflect the new data until next model training.
                    """)
                elif prediction_updated:
                    st.success(f"""
                    ✅ **Prediction System Updated**
                    
                    The ML prediction system has been updated with the new draw result.
                    
                    ⚠️ Main dataset not updated - the "Last 5 Actual Draws" display may not show this new result.
                    """)
                elif date_exists:
                    st.warning(f"""
                    ⚠️ **Draw Date Already Exists**
                    
                    A draw result for {draw_date.strftime('%Y-%m-%d')} already exists in the dataset.
                    Please choose a different date or use the Upload/Data tab to modify existing entries.
                    """)
                else:
                    st.error(f"""
                    ❌ **Update Failed**
                    
                    Neither the main dataset nor the prediction system could be updated.
                    This may indicate a data storage issue or permission problem.
                    
                    Try using the Upload/Data → Manual Entry tab as an alternative.
                    """)
            except Exception as e:
                st.error(f"Error updating prediction history: {str(e)}")

    # Tab view for different aspects of ML analysis
    ml_tabs = st.tabs([
        "Train & Predict", 
        "Model Evaluation", 
        "Feature Importance", 
        "Prediction Management", 
        "Prediction History"
    ])
    
    # Prepare data once for all tabs
    with st.spinner("Preparing data..."):
        df, X, y, feature_cols = _prep(df_history)
    
    if X is None:
        st.error("Data preparation failed. Please check your dataset.")
        return
    
    # Tab 1: Train & Predict
    with ml_tabs[0]:
        # Performance options for prediction
        with st.expander("Performance Options"):
            use_caching = st.toggle("Enable prediction caching", value=True, 
                                   help="Caching significantly improves performance by reusing calculations")
            clear_cache = st.button("Clear prediction cache", 
                                   help="Clear the cached predictions to force recalculation")
            
            if clear_cache:
                # Create a temporary ModernizedPredictionSystem just to clear cache
                from .modernized_prediction_system import ModernizedPredictionSystem
                temp_system = ModernizedPredictionSystem()
                temp_system.clear_cache()
                st.success("Prediction cache cleared!")
        
        if st.button("Train & predict next draw"):
            with st.spinner("Training model and generating predictions..."):
                try:
                    # Start ML training session
                    import time
                    start_time = time.time()
                    
                    # Map model names to memory system identifiers
                    model_type_map = {
                        "Random Forest": "random_forest",
                        "Gradient Boosting": "gradient_boosting",
                        "Ridge Regression": "ridge_regression"
                    }
                    model_type = model_type_map[model_name]
                    
                    # Start tracking this training session
                    from .ml_memory import get_ml_tracker as ml_tracker_func
                    tracker = ml_tracker_func()
                    dataset_info = {
                        "total_draws": len(df_history),
                        "date_range": f"{df_history['draw_date'].iloc[-1]} to {df_history['draw_date'].iloc[0]}",
                        "features_count": len(feature_cols)
                    }
                    session_id = tracker.start_training_session(model_type, hyperparams, dataset_info)
                    
                    # Train the specific model with cross-validation
                    ModelClass = MODELS[model_name]
                    if model_name in ["Random Forest", "Gradient Boosting"]:
                        model = MultiOutputRegressor(ModelClass(**hyperparams))
                    else:
                        model = ModelClass(**hyperparams)
                    
                    # Perform cross-validation to get performance metrics
                    cv = KFold(n_splits=5, shuffle=True, random_state=42)
                    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
                    mae_scores = -cv_scores
                    
                    # Train on full dataset
                    model.fit(X, y)
                    
                    # Generate predictions using trained model
                    if not use_caching:
                        modernized_system = ModernizedPredictionSystem(df_history)
                        modernized_system.clear_cache()
                        predictions = modernized_system.generate_weighted_predictions(count=5)
                    else:
                        predictions = get_enhanced_predictions(df_history, count=5)
                    
                    # Calculate training metrics
                    elapsed_time = time.time() - start_time
                    performance_metrics = {
                        "cv_mae_mean": float(mae_scores.mean()),
                        "cv_mae_std": float(mae_scores.std()),
                        "cv_score_best": float(mae_scores.min()),
                        "cv_score_worst": float(mae_scores.max())
                    }
                    
                    # Store training session in memory (this clears previous memory for this model only)
                    import os
                    import joblib
                    model_path = f"data/ml_memory/models/{model_type}_{session_id}.joblib"
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    joblib.dump(model, model_path)
                    
                    success = tracker.complete_training_session(
                        session_id, 
                        performance_metrics, 
                        model_path, 
                        True,
                        f"Training completed with {len(df_history)} draws using {len(feature_cols)} features"
                    )
                    
                    # Store predictions in persistent SQLite database
                    pm = get_prediction_manager()
                    prediction_set_id = pm.store_model_predictions(
                        model_name=model_name,
                        predictions=predictions,
                        hyperparameters=hyperparams,
                        performance_metrics=performance_metrics,
                        features_used=feature_cols or [],
                        training_duration=elapsed_time,
                        notes=f"Training session {session_id} with {len(df_history)} draws"
                    )
                    
                    if success and prediction_set_id:
                        st.success(f"✅ {model_name} training completed and predictions stored!")
                        st.info(f"Prediction Set ID: {prediction_set_id}")
                    
                    st.info(f"Training and prediction completed in {elapsed_time:.2f} seconds " + 
                           f"with caching {'enabled' if use_caching else 'disabled'}")
                    
                    # Display results using the new prediction interface
                    render_prediction_display_interface()
                    
                    # Get prediction history
                    try:
                        from .prediction_system import PredictionSystem
                        prediction_system = PredictionSystem(df_history)
                        
                        # Initialize columns for comparison
                        compare_col1, compare_col2 = st.columns(2)
                        
                        with compare_col1:
                            st.markdown("### Last 5 Predictions")
                            
                            if (prediction_system.prediction_history and 
                                len(prediction_system.prediction_history.get('predictions', [])) > 0):
                                
                                # Get last 5 predictions
                                last_predictions = prediction_system.prediction_history['predictions'][-5:]
                                
                                # Display in reverse order (newest first)
                                for i, pred in enumerate(reversed(last_predictions)):
                                    if pred:
                                        # Format prediction date if available
                                        prediction_date_str = "Unknown date"
                                        if 'timestamp' in pred:
                                            try:
                                                date_obj = datetime.datetime.fromisoformat(pred['timestamp'])
                                                prediction_date_str = date_obj.strftime("%Y-%m-%d")
                                            except:
                                                prediction_date_str = pred.get('timestamp', 'Unknown date')
                                        
                                        # Format target draw date if available
                                        target_date_str = "Unknown"
                                        if 'prediction_for_date' in pred:
                                            try:
                                                target_date_obj = datetime.datetime.fromisoformat(pred['prediction_for_date'])
                                                target_date_str = target_date_obj.strftime("%Y-%m-%d")
                                            except:
                                                target_date_str = pred.get('prediction_for_date', 'Unknown')
                                        
                                        # Extract prediction numbers
                                        white_nums = pred.get('white_numbers', [])
                                        pb = pred.get('powerball', 0)
                                        
                                        # Create colorful display with prediction date and target date
                                        st.markdown(f"**Prediction {i+1}** (Made on: {prediction_date_str}, For draw: {target_date_str}):")
                                        
                                        # Create visualization
                                        balls_html = ""
                                        for num in white_nums:
                                            balls_html += f'<span style="display:inline-block; background-color:white; color:black; border:2px solid blue; border-radius:50%; width:30px; height:30px; text-align:center; line-height:30px; margin-right:5px;">{num}</span>'
                                        
                                        # Add powerball with different styling
                                        balls_html += f'<span style="display:inline-block; background-color:red; color:white; border-radius:50%; width:30px; height:30px; text-align:center; line-height:30px;">{pb}</span>'
                                        
                                        st.markdown(balls_html, unsafe_allow_html=True)
                            else:
                                st.info("No prediction history available yet. Make predictions to see them here.")
                        
                        with compare_col2:
                            st.markdown("### Last 5 Actual Draws")
                            
                            # Get last 5 historical draws
                            if not df_history.empty:
                                last_draws = df_history.sort_values('draw_date', ascending=False).head(5)
                                
                                for i, (_, row) in enumerate(last_draws.iterrows()):
                                    # Format date
                                    date_str = "Unknown date"
                                    if 'draw_date' in row:
                                        try:
                                            date_obj = pd.to_datetime(row['draw_date'])
                                            date_str = date_obj.strftime("%Y-%m-%d")
                                        except:
                                            date_str = str(row.get('draw_date', 'Unknown date'))
                                    
                                    # Get the numbers
                                    white_nums = [row.get(f'n{j}', 0) for j in range(1, 6)]
                                    pb = row.get('powerball', 0)
                                    
                                    # Display with title
                                    st.markdown(f"**Draw {i+1}** ({date_str}):")
                                    
                                    # Create visualization
                                    balls_html = ""
                                    for num in white_nums:
                                        balls_html += f'<span style="display:inline-block; background-color:white; color:black; border:2px solid blue; border-radius:50%; width:30px; height:30px; text-align:center; line-height:30px; margin-right:5px;">{num}</span>'
                                    
                                    # Add powerball with different styling
                                    balls_html += f'<span style="display:inline-block; background-color:red; color:white; border-radius:50%; width:30px; height:30px; text-align:center; line-height:30px;">{pb}</span>'
                                    
                                    st.markdown(balls_html, unsafe_allow_html=True)
                            else:
                                st.info("No historical data available. Please upload draw history data.")
                        
                        # Add match statistics if available
                        if (prediction_system.prediction_history and 
                            len(prediction_system.prediction_history.get('accuracy', [])) > 0):
                            
                            st.markdown("### Recent Match Statistics")
                            
                            # Get last 5 accuracy records
                            last_accuracy = prediction_system.prediction_history['accuracy'][-5:]
                            
                            # Calculate average matches
                            avg_white_matches = sum(acc.get('white_match_count', 0) for acc in last_accuracy) / max(len(last_accuracy), 1)
                            pb_matches = sum(1 for acc in last_accuracy if acc.get('pb_match', False))
                            pb_match_rate = pb_matches / max(len(last_accuracy), 1) * 100
                            
                            # Display statistics
                            stats_col1, stats_col2 = st.columns(2)
                            with stats_col1:
                                st.metric("Avg White Ball Matches", f"{avg_white_matches:.1f}/5")
                            with stats_col2:
                                st.metric("Powerball Match Rate", f"{pb_match_rate:.1f}%")
                    
                    except Exception as e:
                        st.error(f"Error loading comparison data: {str(e)}")
                    
                    # Display note about continuous learning
                    st.info(
                        "💡 **Learning System**: This prediction system learns from past predictions "
                        "and outcomes. As new draws occur, the system will automatically analyze "
                        "which analysis tools performed best and adjust future predictions accordingly."
                    )
                    
                except Exception as e:
                    st.error(f"Error generating predictions: {str(e)}")
    
    # Tab 2: Model Evaluation
    with ml_tabs[1]:
        # Create tabs for different types of evaluation
        eval_tabs = st.tabs(["Model Performance", "Training History", "Prediction History"])
        
        # Model Performance Tab
        with eval_tabs[0]:
            if st.button("Evaluate model performance"):
                try:
                    with st.spinner("Evaluating model performance..."):
                        # Initialize model for each target column separately
                        base_model_class = MODELS[model_name]
                        
                        # Prepare for cross-validation
                        cv = KFold(n_splits=5, shuffle=True, random_state=42)
                        results = []
                        
                        # For each target column, create and evaluate a separate model
                        for i, col in enumerate(y.columns):
                            target = y[col].values
                            
                            # Create a fresh base model for this target
                            base = base_model_class(**hyperparams, random_state=0)
                            
                            # Perform cross-validation for this target
                            try:
                                scores = cross_val_score(
                                    base, X, target, 
                                    cv=cv, 
                                    scoring='neg_mean_absolute_error'
                                )
                                mae = -scores.mean()  # Convert negative MAE to positive
                                std_dev = scores.std()
                            except Exception as inner_e:
                                st.warning(f"Skipping target {col} due to: {str(inner_e)}")
                                mae = np.nan
                                std_dev = np.nan
                                
                            results.append({
                                'Target': col,
                                'MAE': mae,
                                'Std Dev': std_dev
                            })
                        
                        # Filter out any failed evaluations
                        results_df = pd.DataFrame(results)
                        valid_results = results_df.dropna()
                        
                        if len(valid_results) > 0:
                            # Display cross-validation results
                            st.subheader("Cross-Validation Results")
                            st.dataframe(valid_results)
                            
                            # Calculate and display mean error across all targets
                            mean_mae = valid_results['MAE'].mean()
                            st.metric("Overall Mean Absolute Error", f"{mean_mae:.2f}")
                            
                            # Visual representation of errors
                            fig = px.bar(
                                valid_results, 
                                x='Target', 
                                y='MAE', 
                                error_y='Std Dev',
                                title="Prediction Error by Number Position",
                                labels={'MAE': 'Mean Absolute Error', 'Target': 'Number Position'},
                                color='MAE',
                                color_continuous_scale='Blues_r'  # Reversed blues (darker = lower error)
                            )
                            st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error during model evaluation: {str(e)}")
        
        # Training History Tab - ML Memory Buffer Display
        with eval_tabs[1]:
            st.subheader("ML Model Training History")
            st.write("Each model maintains independent training session memory that persists across application restarts.")
            
            # Display training history for each model type
            model_type_map = {
                "Random Forest": "random_forest",
                "Gradient Boosting": "gradient_boosting", 
                "Ridge Regression": "ridge_regression"
            }
            
            # Create columns for each model type
            history_cols = st.columns(3)
            
            for i, (display_name, model_type) in enumerate(model_type_map.items()):
                with history_cols[i]:
                    st.markdown(f"### {display_name}")
                    
                    # Get training session for this model
                    training_info = get_model_last_training(model_type)
                    
                    if training_info:
                        # Display training session details
                        st.success("Training session found")
                        
                        # Format timestamp
                        try:
                            timestamp = datetime.datetime.fromisoformat(training_info['timestamp'])
                            formatted_time = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                        except:
                            formatted_time = training_info['timestamp']
                        
                        st.write(f"**Last Trained:** {formatted_time}")
                        st.write(f"**Duration:** {training_info['training_duration']:.2f} seconds")
                        
                        # Display hyperparameters
                        st.write("**Hyperparameters:**")
                        for param, value in training_info['hyperparameters'].items():
                            st.write(f"- {param}: {value}")
                        
                        # Display performance metrics
                        st.write("**Performance Metrics:**")
                        metrics = training_info['performance_metrics']
                        st.metric("CV MAE Mean", f"{metrics['cv_mae_mean']:.4f}")
                        st.metric("CV MAE Std", f"{metrics['cv_mae_std']:.4f}")
                        st.metric("Best CV Score", f"{metrics['cv_score_best']:.4f}")
                        st.metric("Worst CV Score", f"{metrics['cv_score_worst']:.4f}")
                        
                        # Display notes if available
                        if training_info.get('notes'):
                            st.write(f"**Notes:** {training_info['notes']}")
                        
                        # Clear memory button for this specific model
                        if st.button(f"Clear {display_name} Memory", key=f"clear_{model_type}"):
                            from .ml_memory import clear_model_training_memory
                            if clear_model_training_memory(model_type):
                                st.success(f"Memory cleared for {display_name}")
                                st.rerun()
                            else:
                                st.error(f"Failed to clear memory for {display_name}")
                    else:
                        st.info("No training session found")
                        st.write("Train this model to see its memory here")
            
            # Memory summary section
            st.markdown("---")
            st.subheader("Memory Summary")
            
            try:
                tracker = get_ml_tracker()
                summary = tracker.memory_buffer.get_memory_summary()
                
                summary_col1, summary_col2 = st.columns(2)
                
                with summary_col1:
                    st.metric("Models with Training Memory", summary['total_models_trained'])
                    
                    if summary['models_with_memory']:
                        st.write("**Models Trained:**")
                        for model in summary['models_with_memory']:
                            display_name = next(k for k, v in model_type_map.items() if v == model)
                            st.write(f"- {display_name}")
                
                with summary_col2:
                    if summary['last_training_times']:
                        st.write("**Last Training Times:**")
                        for model, timestamp in summary['last_training_times'].items():
                            try:
                                dt = datetime.datetime.fromisoformat(timestamp)
                                formatted = dt.strftime("%m/%d %H:%M")
                            except:
                                formatted = timestamp
                            display_name = next(k for k, v in model_type_map.items() if v == model)
                            st.write(f"- {display_name}: {formatted}")
                
            except Exception as e:
                st.error(f"Error loading memory summary: {str(e)}")
            
        # Prediction History Tab
        with eval_tabs[2]:
            st.write("Track prediction system performance over time")
            
            # Initialize prediction system to access history
            try:
                from .prediction_system import PredictionSystem
                import pandas as pd  # Ensure pandas is available in this scope
                
                # Load an instance with our data
                prediction_system = PredictionSystem(df_history)
                
                # Check if we have prediction history
                if (prediction_system.prediction_history and 
                    len(prediction_system.prediction_history.get('predictions', [])) > 0 and
                    len(prediction_system.prediction_history.get('accuracy', [])) > 0):
                    
                    # Organize prediction history for display
                    history_data = []
                    
                    for i, (pred, acc) in enumerate(zip(
                        prediction_system.prediction_history['predictions'], 
                        prediction_system.prediction_history['accuracy']
                    )):
                        # Skip entries with missing data
                        if not pred or not acc:
                            continue
                            
                        # First try to get the date from the prediction data which has timestamp
                        date = None
                        if pred and 'timestamp' in pred:
                            try:
                                date_obj = datetime.datetime.fromisoformat(pred['timestamp'])
                                date = date_obj.strftime("%Y-%m-%d")
                            except:
                                date = pred.get('timestamp', '').split('T')[0] if isinstance(pred.get('timestamp', ''), str) else None
                        
                        # If no date in prediction, try the accuracy timestamp
                        if not date and acc and 'timestamp' in acc:
                            try:
                                date_obj = datetime.datetime.fromisoformat(acc['timestamp'])
                                date = date_obj.strftime("%Y-%m-%d")
                            except:
                                date = acc.get('timestamp', '').split('T')[0] if isinstance(acc.get('timestamp', ''), str) else None
                        
                        # Fallback to prediction for date if no timestamp available
                        if not date:
                            # Try to get target date
                            if pred and 'prediction_for_date' in pred:
                                try:
                                    date_obj = datetime.datetime.fromisoformat(pred['prediction_for_date'])
                                    date = f"{date_obj.strftime('%Y-%m-%d')} (target)"
                                except:
                                    pass
                            
                            # Last resort, use current date with sequence number
                            if not date:
                                current_date = datetime.datetime.now()
                                date = f"{current_date.strftime('%Y-%m-%d')} #{i+1}"
                        
                        # Get accuracy metrics
                        white_matches = acc.get('white_match_count', 0)
                        pb_match = acc.get('pb_match', False)
                        
                        # Format prediction for display
                        white_nums = pred.get('white_numbers', [])
                        white_str = ", ".join(str(n) for n in white_nums) if white_nums else "N/A"
                        pb = pred.get('powerball', 0)
                        
                        # Add to history data
                        history_data.append({
                            'Date': date,
                            'White Matches': white_matches,
                            'PB Match': 'Yes' if pb_match else 'No',
                            'White Numbers': white_str,
                            'Powerball': pb
                        })
                    
                    if history_data:
                        # Convert to DataFrame for display
                        history_df = pd.DataFrame(history_data)
                        
                        # Display history table
                        st.subheader("Prediction History")
                        st.dataframe(history_df)
                        
                        # Show accuracy trends if we have multiple entries
                        if len(history_df) > 1:
                            # White ball matches over time
                            fig = px.line(
                                history_df, 
                                x='Date', 
                                y='White Matches',
                                title='White Ball Matches Over Time',
                                markers=True
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Powerball match rate
                            pb_matches = history_df['PB Match'].value_counts()
                            pb_match_rate = pb_matches.get('Yes', 0) / len(history_df) * 100
                            
                            st.metric("Powerball Match Rate", f"{pb_match_rate:.1f}%")
                    else:
                        st.info("Prediction history data is not formatted correctly.")
                else:
                    st.info("""
                    No prediction history available yet. To build prediction history:
                    
                    1. Make predictions using the "Train & predict next draw" button
                    2. After the actual draw occurs, go to "New Draw Results" in the sidebar
                    3. Enter the actual results to update the system
                    
                    This will help track prediction accuracy over time.
                    """)
                    
                    # Add an example to make it clearer
                    with st.expander("How it works"):
                        st.markdown("""
                        **Example workflow:**
                        
                        1. Make a prediction for the upcoming draw
                        2. After the draw happens, enter the actual numbers
                        3. The system will calculate how many numbers you matched
                        4. This history builds up over time to show which prediction methods work best
                        5. The system automatically adjusts its methods based on past performance
                        """)
            except Exception as e:
                st.error(f"Error loading prediction history: {str(e)}")
    
    # Tab 3: Feature Importance
    with ml_tabs[2]:
        # Create tabs for different feature importance analyses
        fi_tabs = st.tabs(["Basic Feature Importance", "Enhanced System Analysis"])
        
        # Basic Feature Importance Tab
        with fi_tabs[0]:
            if st.button("Analyze basic feature importance"):
                try:
                    # Train model for feature importance
                    base_model_class = MODELS[model_name]
                    
                    # Only show feature importance for models that support it
                    if model_name not in ["Random Forest", "Gradient Boosting"]:
                        st.info(f"{model_name} doesn't provide native feature importance. Please select Random Forest or Gradient Boosting to see feature importance.")
                    else:
                        with st.spinner(f"Training {model_name} model for feature importance..."):
                            base = base_model_class(**hyperparams, random_state=0)
                            model = MultiOutputRegressor(base)
                            model.fit(X, y)
                            
                            # Extract feature importance
                            importances = []
                            for i, estimator in enumerate(model.estimators_):
                                target = y.columns[i]
                                for j, importance in enumerate(estimator.feature_importances_):
                                    importances.append({
                                        'Feature': feature_cols[j],
                                        'Importance': importance,
                                        'Target': target
                                    })
                            
                            # Aggregate importance across targets
                            imp_df = pd.DataFrame(importances)
                            avg_imp = imp_df.groupby('Feature')['Importance'].mean().reset_index()
                            avg_imp = avg_imp.sort_values('Importance', ascending=False)
                            
                            # Display overall feature importance
                            st.subheader("Feature Importance")
                            fig = px.bar(avg_imp, x='Feature', y='Importance',
                                        title="Average Feature Importance Across All Targets",
                                        labels={'Importance': 'Importance Score', 'Feature': 'Feature'},
                                        color='Importance',
                                        color_continuous_scale='Blues')
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Target-specific feature importance
                            st.subheader("Target-Specific Feature Importance")
                            target_select = st.selectbox("Select target number:", y.columns)
                            
                            target_imp = imp_df[imp_df['Target'] == target_select]
                            target_imp = target_imp.sort_values('Importance', ascending=False)
                            
                            fig2 = px.bar(target_imp, x='Feature', y='Importance',
                                        title=f"Feature Importance for {target_select}",
                                        labels={'Importance': 'Importance Score', 'Feature': 'Feature'},
                                        color='Importance',
                                        color_continuous_scale='Blues')
                            st.plotly_chart(fig2, use_container_width=True)
                            
                            st.caption("""
                            **Interpretation:** Higher importance scores indicate features that 
                            have more influence on the model's predictions. The ordinal feature 
                            represents the sequential nature of draws, while pair features 
                            represent the presence of specific number pairs in the draws.
                            """)
                except Exception as e:
                    st.error(f"Error analyzing feature importance: {str(e)}")
        
        # Enhanced Feature Importance Tab from Prediction System
        with fi_tabs[1]:
            st.write("This analysis uses our enhanced prediction system with more sophisticated features.")
            
            if st.button("Run enhanced feature importance analysis"):
                try:
                    with st.spinner("Generating enhanced feature importance analysis..."):
                        # Use our prediction system which has more sophisticated features
                        from .prediction_system import PredictionSystem
                        
                        # Initialize the prediction system with our dataset
                        prediction_system = PredictionSystem(df_history)
                        
                        # Train models (also generates feature importance)
                        prediction_system._train_models()
                        feature_importance = prediction_system._calculate_feature_importance()
                        
                        if (feature_importance and 
                            'white_balls' in feature_importance and 
                            feature_importance['white_balls'] is not None):
                            
                            # Create tabs for white balls and powerball feature importance
                            ball_tabs = st.tabs(["White Balls Features", "Powerball Features"])
                            
                            # White balls feature importance
                            with ball_tabs[0]:
                                df_imp = feature_importance['white_balls']
                                if df_imp is not None and not df_imp.empty:
                                    # Limit to top 15 features for better visualization
                                    df_imp = df_imp.head(15)
                                    
                                    # Create a horizontal bar chart
                                    fig = px.bar(
                                        df_imp, 
                                        x='Importance', 
                                        y='Feature', 
                                        orientation='h',
                                        title='Top 15 Features for White Ball Prediction',
                                        labels={'Importance': 'Relative Importance', 'Feature': 'Feature Name'},
                                        color='Importance',
                                        color_continuous_scale='Blues'
                                    )
                                    fig.update_layout(height=500)
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Explanation of features
                                    with st.expander("White Ball Feature Explanation"):
                                        st.markdown("""
                                        **Feature Key:**
                                        - `dow_X`: Day of week (0=Monday, 6=Sunday)
                                        - `draw_number`: Sequential draw number
                                        - `sum_mean_X`: Average sum of white balls over X draws
                                        - `sum_std_X`: Standard deviation of sums over X draws
                                        - `nX_lagY`: Number X from Y draws ago
                                        """)
                                else:
                                    st.info("Feature importance data not available for white balls.")
                            
                            # Powerball feature importance
                            with ball_tabs[1]:
                                df_imp = feature_importance['powerball']
                                if df_imp is not None and not df_imp.empty:
                                    # Limit to top 15 features for better visualization
                                    df_imp = df_imp.head(15)
                                    
                                    # Create a horizontal bar chart
                                    fig = px.bar(
                                        df_imp, 
                                        x='Importance', 
                                        y='Feature', 
                                        orientation='h',
                                        title='Top 15 Features for Powerball Prediction',
                                        labels={'Importance': 'Relative Importance', 'Feature': 'Feature Name'},
                                        color='Importance',
                                        color_continuous_scale='Reds'
                                    )
                                    fig.update_layout(height=500)
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Explanation of features
                                    with st.expander("Powerball Feature Explanation"):
                                        st.markdown("""
                                        **Feature Key:**
                                        - `dow_X`: Day of week (0=Monday, 6=Sunday)
                                        - `draw_number`: Sequential draw number
                                        - `powerball_lagY`: Powerball from Y draws ago
                                        - `sum_mean_X`: Average sum of white balls over X draws
                                        - `sum_std_X`: Standard deviation of sums over X draws
                                        """)
                                else:
                                    st.info("Feature importance data not available for Powerball.")
                        else:
                            st.info("Enhanced feature importance analysis requires at least 50 draws and successful model training.")
                            
                        # Show how feature importance is used
                        with st.expander("How Feature Importance Improves Predictions"):
                            st.markdown("""
                            ### Impact on Prediction Quality
                            
                            Feature importance analysis helps in several ways:
                            
                            1. **Better Feature Selection**: By identifying the most influential factors, we can focus on the most relevant data.
                            
                            2. **Model Tuning**: We can fine-tune our models to give more weight to important features.
                            
                            3. **Insight Generation**: Understanding what drives predictions helps interpret results.
                            
                            4. **Reduced Noise**: We can remove low-importance features that might be adding noise.
                            
                            The enhanced prediction system uses this information to create a weighted ensemble of prediction methods.
                            """)
                except Exception as e:
                    st.error(f"Error in enhanced feature importance analysis: {str(e)}")
    
    # Tab 4: Prediction Management
    with ml_tabs[3]:
        render_prediction_management_interface()
    
    # Tab 5: Prediction History  
    with ml_tabs[4]:
        render_prediction_history_interface()
