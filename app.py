import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import time
from insightAI import (detect_problem_type, prepare_data, train_model,
                   create_correlation_heatmap, create_prediction_plot, get_feature_importance,
                   create_outlier_plot)

# Page config
st.set_page_config(page_title="InsightAI", page_icon="🤖", layout="wide")
st.markdown("""
    <style>
    section[data-testid="stSidebar"] > div:first-child {
        animation: sidebarFadeSlide 1.2s cubic-bezier(.4,0,.2,1);
    }
    @keyframes sidebarFadeSlide {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    </style>
    """, unsafe_allow_html=True)

# Custom CSS for animations
st.markdown("**Upload your dataset and let AI do the analysis!**")
st.markdown("""
    <style>
    body, .main, .block-container {
        background-color: #18181b !important;
        color: #fff !important;
    }
    .fade-in-header {
        animation: fadeIn 1.2s ease;
        font-size: 2.2em;
        font-weight: 700;
        color: #fff;
        margin-bottom: 0.2em;
        letter-spacing: 1px;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .subtle-desc {
        font-size: 1.1em;
        color: #e0e0e0;
        margin-bottom: 1em;
        opacity: 0.85;
    }
    .feature-item {
        color: #fff !important;
    }
    .stApp {
        background-color: #18181b !important;
        color: #fff !important;
    }
    </style>
    """, unsafe_allow_html=True)
st.markdown('<div class="fade-in-header">InsightAI - Automated ML Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle-desc">Upload your dataset and let AI do the analysis!</div>', unsafe_allow_html=True)

# Sidebar with file upload and sample options
st.sidebar.header("📤 Upload Dataset")
choice = st.sidebar.selectbox("Select an option:",
               ["Upload your CSV", "Use Sample_insurance_data", "Use Sample_forecasting_of_Walmart"])
if choice == "Upload your CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("✅ File uploaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error reading file: {str(e)}")
elif choice == "Use Sample_insurance_data":
    try:
        uploaded_file = pd.read_csv(os.path.join("Datasets", "Sample_general_insurance_data.csv"))
        st.sidebar.success("✅ Loaded sample general dataset")
    except Exception as e:
        st.sidebar.error(f"Error loading sample data: {str(e)}")
elif choice == "Use Sample_forecasting_of_Walmart":
    try:
        uploaded_file = pd.read_csv(os.path.join("Datasets", "Sample_General_forecasting_of_Walmart.csv"))
        st.sidebar.success("✅ Loaded sample forecasting dataset")
    except Exception as e:
        st.sidebar.error(f"Error loading sample data: {str(e)}")

# Main area description when no file is loaded
if uploaded_file is None:
    st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <h2 style="color: #ffffff;">Welcome to InsightAI! 🚀</h2>
            <p style="font-size: 1.2em; color: #e0e0e0;">
                Your all-in-one platform for data analysis, visualization, and machine learning.
            </p>
        </div>
        
        <div style="background-color: #2d2d2d; padding: 20px; border-radius: 10px; margin: 20px 0;">
            <h3 style="color: #ffffff;">🎯 Getting Started</h3>
            <ol style="color: #e0e0e0;">
                <li>Upload your CSV file or select a sample dataset from the sidebar</li>
                <li>Explore your data in the Data Overview tab</li>
                <li>Generate visualizations to understand patterns</li>
                <li>Train and evaluate machine learning models</li>
                <li>Analyze clustering results and future predictions</li>
            </ol>
        </div>

        <div style="background-color: #2d2d2d; padding: 20px; border-radius: 10px; margin: 20px 0;">
            <h3 style="color: #ffffff;">✨ Features</h3>
            <ul style="color: #e0e0e0;">
                <li>📊 Automatic data analysis and statistics</li>
                <li>📈 Interactive visualizations</li>
                <li>🤖 Automated ML model training and comparison</li>
                <li>🎯 Support for both Regression and Classification</li>
                <li>🔮 Time series forecasting with Prophet</li>
                <li>🧩 K-means clustering analysis</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

if uploaded_file is not None:
    # Load data
    try:
        with st.spinner('Loading your dataset...'):
          time.sleep(0.5)
          df = pd.read_csv(uploaded_file).copy()
            
        st.success(f"Data loaded successfully! Shape: {df.shape}")
        
        # Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "|📊 Data Overview|", 
            "|📈 Visualizations|", 
            "|🤖 Supervised Learning|",
            "|🧩 Clustering|",
            "|🔮 Future Prediction|"
        ])
        
        # TAB 1: Data Overview
        with tab1:
            st.header("Dataset Overview")
            
            # Dataset summary
            st.markdown("""
            <div style="background-color: #2d2d2d; padding: 20px; border-radius: 10px; margin: 20px 0;">
                <h3 style="color: #ffffff;">📊 Dataset Summary</h3>
                <p style="color: #e0e0e0;">
                This section provides a comprehensive overview of your dataset's structure and content. 
                Understanding these basic characteristics is crucial for any data analysis project.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            rows = df.shape[0]
            cols = df.shape[1]
            missing = df.isnull().sum().sum()
            col1.metric("Rows", rows)
            col2.metric("Columns", cols)
            col3.metric("Missing Values", missing)
            
            # Key insights about the data
            st.markdown(f"""
            <div style="background-color: #2d2d2d; padding: 20px; border-radius: 10px; margin: 20px 0;">
                <h4 style="color: #ffffff;">🔍 Key Insights</h4>
                <ul style="color: #e0e0e0;">
                    <li>Your dataset contains <b>{rows:,}</b> records with <b>{cols}</b> features</li>
                    <li>There are <b>{missing:,}</b> missing values in total</li>
                    <li>Numeric columns: <b>{len(df.select_dtypes(include=['number']).columns)}</b></li>
                    <li>Categorical columns: <b>{len(df.select_dtypes(include=['object']).columns)}</b></li>
                    <li>Memory usage: <b>{df.memory_usage().sum() / 1024 / 1024:.2f} MB</b></li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

            st.subheader("First 10 Rows")
            first_rows = df.head(10)
            first_placeholder = st.empty()
            for i in range(1, len(first_rows)+1):
                first_placeholder.dataframe(first_rows.head(i))
                time.sleep(0.12)
                
            st.subheader("DataFrame Info")
            info_df = pd.DataFrame({
                'Column': df.columns,
                'Non-Null Count': df.count().values,
                'Dtype': df.dtypes.astype(str).values
            })
            st.dataframe(info_df)

            st.subheader("Statistical Summary")
            stats = df.describe()
            stats_placeholder = st.empty()
            for i in range(1, len(stats)+1):
                stats_placeholder.dataframe(stats.head(i))
                time.sleep(0.12)

            st.subheader("Last 10 Rows")
            last_rows = df.tail(10)
            last_placeholder = st.empty()
            for i in range(1, len(last_rows)+1):
                last_placeholder.dataframe(last_rows.head(i))
                time.sleep(0.12)
            
            
            
        # TAB 2: Visualizations
        with tab2:
            st.header("Data Visualizations")
            
            st.markdown("""
            <div style="background-color: #2d2d2d; padding: 20px; border-radius: 10px; margin: 20px 0;">
                <h3 style="color: #ffffff;">📈 Visualization Tools</h3>
                <p style="color: #e0e0e0;">
                This section helps you understand your data through interactive visualizations:
                </p>
                <ul style="color: #e0e0e0;">
                    <li><b>Correlation Heatmap:</b> Shows relationships between numeric variables</li>
                    <li><b>Distribution Plots:</b> Visualize the spread and shape of your data</li>
                    <li><b>Outlier Analysis:</b> Identify and analyze unusual values in your dataset</li>
                </ul>
                <p style="color: #e0e0e0;">
                Use these visualizations to spot patterns, identify outliers, and understand relationships in your data.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            if len(numeric_cols) >= 2:
                st.subheader("Correlation Heatmap")
                heatmap = create_correlation_heatmap(df)
                if heatmap:
                    st.plotly_chart(heatmap, use_container_width=True)
            
            if numeric_cols:
                st.subheader("Distribution Plots")
                selected_col = st.selectbox("Select column", numeric_cols)
                fig = px.histogram(df, x=selected_col, marginal="box", title=f"Distribution of {selected_col}")
                st.plotly_chart(fig, use_container_width=True)
            if numeric_cols:
                st.subheader("Outlier Analysis")
                col_for_outliers = st.selectbox("Select column for outlier detection", numeric_cols, key='outlier_select')
                outlier_fig, outliers = create_outlier_plot(df, col_for_outliers)
                st.plotly_chart(outlier_fig, use_container_width=True)
                if not outliers.empty:
                    st.info(f"{len(outliers)} outliers detected in {col_for_outliers}")
                    with st.expander("Show outlier statistics"):
                        st.dataframe(outliers[col_for_outliers].describe())
        
        # TAB 3: ML Models
        with tab3:
            st.header("Machine Learning Models")
            
            st.markdown("""
            <div style="background-color: #2d2d2d; padding: 20px; border-radius: 10px; margin: 20px 0;">
                <h3 style="color: #ffffff;">🤖 Automated Machine Learning</h3>
                <p style="color: #e0e0e0;">
                This section automatically trains and evaluates multiple machine learning models on your data:
                </p>
                <ul style="color: #e0e0e0;">
                    <li><b>Problem Detection:</b> Automatically identifies if it's a regression or classification task</li>
                    <li><b>Model Training:</b> Tests multiple algorithms including:
                        <ul>
                            <li>Linear/Logistic Regression</li>
                            <li>Random Forest Classifier/Regressor</li>
                            <li>Gradient Boosting for Regression</li>
                            <li>XGBoost</li>
                        </ul>
                    </li>
                    <li><b>Performance Metrics:</b> Shows R², RMSE (regression) or Accuracy, F1-score (classification)</li>
                    <li><b>Feature Importance:</b> Identifies which variables are most influential</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.subheader("Select Target Column")
            target_col = st.selectbox("Choose target variable", df.columns.tolist())
            
            if st.button("Train Models", type="primary"):
                with st.spinner("Training models... Please wait."):
                    # Detect problem type
                    problem_type = detect_problem_type(df, target_col)
                    st.info(f"**Detected Problem Type:** {problem_type}")
                    
                    # Prepare data
                    X, y = prepare_data(df, target_col)
                    
                    # Train models
                    results, scaler = train_model(X, y, problem_type)
                    
                    # Display results
                    st.subheader("Model Performance")
                    
                    if problem_type == 'Regression':
                        perf_data = []
                        for name, res in results.items():
                            perf_data.append({
                                'Model': name,
                                'R² Score': f"{res['r2']:.4f}",
                                'RMSE': f"{res['rmse']:.2f}"
                            })
                        perf_df = pd.DataFrame(perf_data)
                        st.dataframe(perf_df, use_container_width=True)
                        
                        best_model = max(results, key=lambda x: results[x]['r2'])
                        st.success(f"**Best Model:** {best_model} (R² = {results[best_model]['r2']:.4f})")
                        
                        # Prediction plot
                        st.subheader("Actual vs Predicted")
                        pred_fig = create_prediction_plot(
                            results[best_model]['actual'],
                            results[best_model]['predictions'],
                            best_model
                        )
                        st.plotly_chart(pred_fig, use_container_width=True)
                        
                    else:  # Classification
                        perf_data = []
                        for name, res in results.items():
                            perf_data.append({
                                'Model': name,
                                'Accuracy': f"{res['accuracy']:.4f}",
                                'F1 Score': f"{res['f1_score']:.4f}"
                            })
                        perf_df = pd.DataFrame(perf_data)
                        st.dataframe(perf_df, use_container_width=True)
                        
                        best_model = max(results, key=lambda x: results[x]['accuracy'])
                        st.success(f"**Best Model:** {best_model} (Accuracy = {results[best_model]['accuracy']:.4f})")
                    
                    # Feature Importance
                    feat_imp = get_feature_importance(results[best_model]['model'], X.columns)
                    if feat_imp is not None:
                        st.subheader("Feature Importance Analysis")
                        with st.spinner("Analyzing feature importance..."):
                            time.sleep(0.5)
                            fig = px.bar(feat_imp.head(10), 
                                       x='Importance', 
                                       y='Feature', 
                                       orientation='h',
                                       title="Top 10 Most Important Features",
                                       labels={'Importance': 'Relative Importance', 'Feature': 'Feature Name'},
                                       color='Importance',
                                       color_continuous_scale='blues')
                            fig.update_layout(
                                showlegend=False,
                                title_x=0.5,
                                title_font_size=18,
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)'
                            )
                            st.plotly_chart(fig, use_container_width=True)
        
        # TAB 4: Clustering (KMeans)
        with tab4:
            st.header("K-Means Clustering")
            
            st.markdown("""
            <div style="background-color: #2d2d2d; padding: 20px; border-radius: 10px; margin: 20px 0;">
                <h3 style="color: #ffffff;">🧩 Clustering Analysis</h3>
                <p style="color: #e0e0e0;">
                Clustering helps discover natural groupings in your data:
                </p>
                <ul style="color: #e0e0e0;">
                    <li><b>K-Means Algorithm:</b> Groups similar data points together</li>
                    <li><b>Interactive Selection:</b> Choose which features to include</li>
                    <li><b>Cluster Visualization:</b> See how your data groups in 2D space</li>
                    <li><b>Adjustable Clusters:</b> Experiment with different numbers of groups</li>
                </ul>
                <p style="color: #e0e0e0;">
                Use clustering to segment your data and identify patterns that might not be immediately obvious.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            if len(numeric_cols) < 2:
                st.warning("Need at least 2 numeric columns for clustering.")
            else:
                features = st.multiselect("Select features for clustering", numeric_cols, default=numeric_cols)
                n_clusters = st.slider("Number of clusters", min_value=2, max_value=10, value=3)
                if st.button("Run K-Means"):
                    from insightAI import run_kmeans
                    with st.spinner("Clustering in progress..."):
                        try:
                            labels, centroids, used_features = run_kmeans(df, n_clusters, features)
                            df_clustered = df.copy()
                            df_clustered["Cluster"] = labels
                            
                            # Count rows in each cluster
                            cluster_counts = pd.Series(labels).value_counts()
                            valid_clusters = cluster_counts[cluster_counts.index != -1]
                            missing_count = cluster_counts.get(-1, 0)
                            
                            # Success message with cluster information
                            st.success(f"Clustering complete! Found {len(valid_clusters)} clusters with {len(df_clustered) - missing_count:,} valid rows.")
                            
                            # Show cluster sizes
                            st.markdown("### Cluster Sizes")
                            for cluster, count in valid_clusters.items():
                                st.markdown(f"- Cluster {cluster}: {count:,} rows")
                            if missing_count > 0:
                                st.warning(f"⚠️ {missing_count:,} rows had missing values and were excluded from clustering")
                            
                            # Show clustered data
                            st.markdown("### Clustered Data Preview")
                            st.dataframe(df_clustered.head(20))
                            
                            # Plot clusters (first two features)
                            if len(used_features) >= 2:
                                import plotly.express as px
                                
                                # Create figure with valid clusters
                                valid_data = df_clustered[df_clustered["Cluster"] != -1]
                                fig = px.scatter(
                                    valid_data, 
                                    x=used_features[0], 
                                    y=used_features[1], 
                                    color="Cluster",
                                    title=f"K-Means Clusters Using {used_features[0]} vs {used_features[1]}",
                                    color_continuous_scale="viridis",
                                    labels={"Cluster": "Cluster Group"}
                                )
                                
                                # Add centroids to plot
                                for i in range(len(centroids)):
                                    fig.add_scatter(
                                        x=[centroids[i][0]], 
                                        y=[centroids[i][1]],
                                        mode='markers',
                                        marker=dict(symbol='x', size=15, color='red'),
                                        name=f'Centroid {i}'
                                    )
                                
                                fig.update_layout(
                                    height=600,
                                    showlegend=True,
                                    legend_title_text="Groups"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Show feature distributions by cluster
                                st.markdown("### Feature Distributions by Cluster")
                                selected_feature = st.selectbox(
                                    "Select feature to view distribution:", 
                                    features
                                )
                                fig_dist = px.box(
                                    valid_data,
                                    x="Cluster",
                                    y=selected_feature,
                                    title=f"Distribution of {selected_feature} by Cluster"
                                )
                                st.plotly_chart(fig_dist, use_container_width=True)
                                
                        except Exception as e:
                            st.error(f"Error during clustering: {str(e)}")

        # TAB 5: Future Prediction (Prophet)
        with tab5:
            st.header("Time Series Forecasting (Prophet)")
            
            st.markdown("""
            <div style="background-color: #2d2d2d; padding: 20px; border-radius: 10px; margin: 20px 0;">
                <h3 style="color: #ffffff;">🔮 Time Series Analysis & Forecasting</h3>
                <p style="color: #e0e0e0;">
                Analyze time-based patterns and predict future values using advanced techniques:
                </p>
                <ul style="color: #e0e0e0;">
                    <li><b>Trend Analysis:</b> Understand long-term movement in your data</li>
                    <li><b>Seasonality Detection:</b> Identify recurring patterns</li>
                    <li><b>Stationarity Tests:</b> Check if your time series is stable</li>
                    <li><b>Rolling Statistics:</b> View moving averages and standard deviations</li>
                    <li><b>Prophet Forecasting:</b> Generate future predictions with confidence intervals</li>
                    <li><b>Autocorrelation:</b> Discover time-based relationships in your data</li>
                </ul>
                <p style="color: #e0e0e0;">
                Select different plots to explore various aspects of your time series data and generate forecasts.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.warning("⚠️ For the future prediction model to work, your dataset must contain a valid date or datetime column. Only the selected date and target columns will be used for forecasting and analysis.")
            date_cols = df.select_dtypes(include=["datetime", "object"]).columns.tolist()
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            st.info("Select a date column and a numeric target column for forecasting.")
            date_col = st.selectbox("Date column", date_cols)
            target_col = st.selectbox("Target column", numeric_cols)
            periods = st.slider("Forecast periods into the future", min_value=7, max_value=365, value=30)
            freq = st.selectbox("Frequency", ["D", "W", "M"], index=0)

            plot_type = st.selectbox(
                "Select time series plot to view",
                [
                    "Trend",
                    "Seasonality",
                    "Stationarity",
                    "Rolling Mean & Std",
                    "Autocorrelation (ACF)",
                    "Partial Autocorrelation (PACF)",
                    "Correlation Heatmap",
                    "Prophet Forecast"
                ]
            )

            if plot_type == "Rolling Mean & Std":
                window = st.slider("Rolling window size", min_value=2, max_value=60, value=12)

            # Define df_agg here for use in all plots
            if st.button("Show Plot"):
                from insightAI import (
                    plot_trend, plot_seasonality, plot_stationarity, adf_test,
                    plot_acf, plot_pacf, plot_correlation, run_prophet
                )
                df_prophet = df[[date_col, target_col]].dropna().copy()
                try:
                    df_prophet[date_col] = pd.to_datetime(df_prophet[date_col], errors='coerce')
                except Exception:
                    st.error("Could not convert selected date column to datetime.")
                else:
                    df_agg = df_prophet.groupby(date_col)[target_col].sum().reset_index()
                    # Lags slider for ACF/PACF
                    if plot_type in ["Autocorrelation (ACF)", "Partial Autocorrelation (PACF)"]:
                        max_lags = max(1, int(len(df_agg) // 2) - 1)
                        lags = st.slider("Number of lags", min_value=1, max_value=max_lags, value=min(40, max_lags))
                    if plot_type == "Trend":
                        fig = plot_trend(df_agg, date_col, target_col)
                        st.plotly_chart(fig, use_container_width=True)
                    elif plot_type == "Seasonality":
                        fig = plot_seasonality(df_agg, date_col, target_col, freq)
                        st.plotly_chart(fig, use_container_width=True)
                    elif plot_type == "Stationarity":
                        fig = plot_stationarity(df_agg, date_col, target_col)
                        st.plotly_chart(fig, use_container_width=True)
                        adf = adf_test(df_agg, target_col)
                        st.write("ADF Statistic:", adf["ADF Statistic"])
                        st.write("p-value:", adf["p-value"])
                    elif plot_type == "Rolling Mean & Std":
                        rolling_mean = df_agg[target_col].rolling(window=window).mean()
                        rolling_std = df_agg[target_col].rolling(window=window).std()
                        import plotly.graph_objects as go
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=df_agg[date_col], y=df_agg[target_col], mode='lines', name='Original'))
                        fig.add_trace(go.Scatter(x=df_agg[date_col], y=rolling_mean, mode='lines', name=f'Rolling Mean ({window})'))
                        fig.add_trace(go.Scatter(x=df_agg[date_col], y=rolling_std, mode='lines', name=f'Rolling Std ({window})'))
                        fig.update_layout(title=f"Rolling Mean & Std (window={window})", xaxis_title=date_col, yaxis_title=target_col)
                        st.plotly_chart(fig, use_container_width=True)
                    elif plot_type == "Autocorrelation (ACF)":
                        fig = plot_acf(df_agg, target_col, lags=lags)
                        st.plotly_chart(fig, use_container_width=True)
                    elif plot_type == "Partial Autocorrelation (PACF)":
                        fig = plot_pacf(df_agg, target_col, lags=lags)
                        st.plotly_chart(fig, use_container_width=True)
                    elif plot_type == "Correlation Heatmap":
                        fig = plot_correlation(df_agg)
                        st.plotly_chart(fig, use_container_width=True)
                    elif plot_type == "Prophet Forecast":
                        # Validate date column format
                        try:
                            # Try to convert date column to datetime
                            df_agg[date_col] = pd.to_datetime(df_agg[date_col], errors='raise')
                        except (ValueError, TypeError):
                            st.error(f"""
                            ⚠️ Invalid date column format:
                            - Column '{date_col}' contains invalid date values
                            - Please ensure the column contains dates in a standard format (e.g., YYYY-MM-DD)
                            - Example valid formats: '2023-01-01', '01/01/2023', '2023-01-01 10:00:00'
                            """)
                            st.stop()
                            
                        # Data validation before running Prophet
                        valid_rows = df_agg[[date_col, target_col]].dropna()
                        if len(valid_rows) < 2:
                            st.error(f"""
                            ⚠️ Insufficient data for forecasting:
                            - Found only {len(valid_rows)} valid rows after removing missing values
                            - Need at least 2 complete rows with both date and target values
                            - Check your data for missing values in '{date_col}' and '{target_col}'
                            """)
                        else:
                            try:
                                with st.spinner("Running Prophet model..."):
                                    # Show data preparation info
                                    st.info(f"""
                                    📊 Preparing data for forecast:
                                    - Valid data points: {len(valid_rows):,}
                                    - Date range: {valid_rows[date_col].min()} to {valid_rows[date_col].max()}
                                    - Forecasting {periods} periods ahead with {freq} frequency
                                    """)
                                    
                                    model, forecast = run_prophet(df_agg, date_col, target_col, periods, freq)
                                    st.success("✨ Forecast complete!")
                                    
                                    # Show forecast summary
                                    st.subheader("Forecast Summary")
                                    forecast_summary = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods)
                                    forecast_summary.columns = ["Date", "Forecast", "Lower Bound", "Upper Bound"]
                                    st.dataframe(forecast_summary)
                                    
                                    # Create visualization
                                    import plotly.graph_objects as go
                                    
                                    # Split forecast into historical and future
                                    cutoff = df_agg[date_col].max()
                                    hist_mask = forecast["ds"] <= cutoff
                                    fut_mask = forecast["ds"] > cutoff
                                    
                                    fig = go.Figure()
                                    
                                    # Historical actuals
                                    fig.add_trace(go.Scatter(
                                        x=df_agg[date_col], 
                                        y=df_agg[target_col], 
                                        mode="lines", 
                                        name="Historical Data", 
                                        line=dict(color="#1f77b4", width=2)
                                    ))
                                    
                                    # Historical forecast
                                    fig.add_trace(go.Scatter(
                                        x=forecast["ds"][hist_mask], 
                                        y=forecast["yhat"][hist_mask], 
                                        mode="lines", 
                                        name="Model Fit", 
                                        line=dict(color="#2ca02c", width=2, dash="dot")
                                    ))
                                    
                                    # Future forecast
                                    fig.add_trace(go.Scatter(
                                        x=forecast["ds"][fut_mask], 
                                        y=forecast["yhat"][fut_mask], 
                                        mode="lines", 
                                        name="Forecast", 
                                        line=dict(color="#ff7f0e", width=2)
                                    ))
                                    
                                    # Uncertainty intervals
                                    fig.add_trace(go.Scatter(
                                        x=forecast["ds"],
                                        y=forecast["yhat_upper"],
                                        mode="lines",
                                        name="95% Confidence Interval",
                                        line=dict(width=0),
                                        showlegend=False
                                    ))
                                    fig.add_trace(go.Scatter(
                                        x=forecast["ds"],
                                        y=forecast["yhat_lower"],
                                        mode="lines",
                                        fill='tonexty',
                                        fillcolor='rgba(68, 68, 68, 0.3)',
                                        line=dict(width=0),
                                        showlegend=True,
                                        name="95% Confidence Interval"
                                    ))
                                    
                                    fig.update_layout(
                                        title=f"Prophet Forecast for {target_col}",
                                        xaxis_title="Date",
                                        yaxis_title=target_col,
                                        height=600,
                                        hovermode='x unified',
                                        legend=dict(
                                            yanchor="top",
                                            y=0.99,
                                            xanchor="left",
                                            x=0.01
                                        )
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Add forecast interpretation
                                    st.markdown("""
                                    ### 📈 Forecast Interpretation
                                    
                                    - **Blue line**: Historical data
                                    - **Green dotted line**: Model fit on historical data
                                    - **Orange line**: Future forecast
                                    - **Gray area**: 95% confidence interval
                                    
                                    The confidence interval shows the range where we expect 95% of future values to fall,
                                    based on the patterns learned from your historical data.
                                    """)
                                    
                            except Exception as e:
                                st.error(f"""
                                ⚠️ Error during forecasting: {str(e)}
                                
                                Common solutions:
                                1. Check that your date column contains valid dates
                                2. Ensure your target column contains numeric values
                                3. Remove or handle any missing values
                                4. Make sure you have enough data points
                                """)
                    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        
else:

        st.markdown("""
            <style>
            .fade-in-landing {
                animation: fadeIn 1.2s ease;
            }
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .feature-list {
                margin-top: 1em;
                margin-bottom: 1em;
            }
            .feature-item {
                font-size: 1.05em;
                color: #1a237e;
                margin-bottom: 0.5em;
                opacity: 0.92;
            }
            </style>
        """, unsafe_allow_html=True)
        st.markdown('<div class="fade-in-landing">', unsafe_allow_html=True)
        st.info("Upload a CSV file from the sidebar to get started!")
        st.markdown("<div class='feature-list'>", unsafe_allow_html=True)
        st.markdown("<div class='feature-item'>Automatic data analysis and statistics</div>", unsafe_allow_html=True)
        st.markdown("<div class='feature-item'>Beautiful interactive visualizations</div>", unsafe_allow_html=True)
        st.markdown("<div class='feature-item'>Automated ML model training and comparison</div>", unsafe_allow_html=True)
        st.markdown("<div class='feature-item'>Supports both Regression and Classification</div>", unsafe_allow_html=True)
        st.markdown("<div class='feature-item'>Easy-to-use interface</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("""
        ###  How to Use:
        1. Upload your CSV file from the sidebar
        2. Explore your data in the **Data Overview** tab
        3. Visualize patterns in the **Visualizations** tab
        4. Train models in the **ML Models** tab
        5. Compare performance and get insights!
        """)
    
        st.markdown("---")

    
st.markdown("---")







