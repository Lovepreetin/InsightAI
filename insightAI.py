import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, f1_score, mean_squared_error
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
import plotly.express as px
from xgboost import XGBRegressor
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from sklearn.cluster import KMeans
from prophet import Prophet

def analyse_data(df):
    """Analyze the dataset and return basic statistics"""
    analyse = {
        'shape': df.shape,
        'missing_values': df.isnull().sum().to_dict(),
        'dtype': df.dtypes.to_dict(),
        'columns': df.columns.tolist(),
        'numerical_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include='object').columns.tolist()
    }
    return analyse

def create_correlation_heatmap(df):
    """Create a correlation heatmap for numerical columns"""
    number_df = df.select_dtypes(include=['number'])
    if not number_df.empty:
        corr = number_df.corr()
        
        # heatmap 
        fig = go.Figure(data=go.Heatmap(
            z=corr,
            x=corr.columns,
            y=corr.columns,
            text=np.round(corr, 2),  
            texttemplate="%{text}",
            textfont={"size": 10},
            colorscale='RdBu_r',
            colorbar=dict(title='Correlation'),
            hoverongaps=False,
        ))
        
        fig.update_layout(
            title='Correlation Heatmap',
            height=700,
            width=700,
        )
        return fig
    return None

def detect_problem_type(df, target_col):
    """Detect if it's a regression or classification problem"""
    unique_values = df[target_col].nunique()
    if df[target_col].dtype == 'object' or unique_values < 10:
        return 'Classification'
    return 'Regression'

def prepare_data(df, target_col):
    """Prepare features and target for model training"""
    try:
        X = df.drop(columns=[target_col]).copy()
        y = df[target_col].copy()
        
        # Encode categorical columns
        categorical_cols = X.select_dtypes(include='object').columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype('str'))
        
        return X, y
    except Exception as e:
        raise Exception(f"Error in data preparation: {str(e)}")

def train_model(X, y, problem_type):
    """Train and evaluate models"""
    try:
        X.dropna(inplace=True)
        y.dropna(inplace=True)
    except Exception as e:
        raise Exception(f"Error in droping null values: {str(e)}")
    try:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define models based on problem type
        if problem_type == 'Classification':
            models = {
                'Logistic Regression': LogisticRegression(max_iter=1000),
                'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42)
            }
        else:  # Regression
            models = {
                'Linear Regression': LinearRegression(),
                'Random Forest': RandomForestRegressor(
                    n_estimators=300, 
                    max_depth=10, 
                    min_samples_split=5,
                    random_state=42
                ),
                'Gradient Boosting': GradientBoostingRegressor(
                    n_estimators=200, 
                    learning_rate=0.05,
                    max_depth=5,
                    random_state=42
                ),
                'XGBoost': XGBRegressor(
                    n_estimators=300,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbosity=0
                )
            }
        
        # Train and evaluate models
        results = {}
        for name, model in models.items():
            # Train model
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            if problem_type == 'Classification':
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted' if len(np.unique(y)) > 2 else 'binary')
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'predictions': y_pred,
                    'actual': y_test
                }
            else:  # Regression
                r2 = r2_score(y_test, y_pred)
                rmse = mean_squared_error(y_test, y_pred, squared=False)
                results[name] = {
                    'model': model,
                    'r2': r2,
                    'rmse': rmse,
                    'predictions': y_pred,
                    'actual': y_test
                }
        
        return results, scaler
    except Exception as e:
        raise Exception(f"Error in model training: {str(e)}")

def create_prediction_plot(actual, predicted, model_name):
    """Create scatter plot of actual vs predicted values"""
    try:
        fig = go.Figure()
        
        # Scatter plot
        fig.add_trace(go.Scatter(
            x=actual,
            y=predicted,
            mode='markers',
            name='Predictions',
            marker=dict(size=8, color='white', opacity=0.6)
        ))
        
        # Perfect prediction line
        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='green', dash='dash')
        ))
        
        fig.update_layout(
            title=f'{model_name}: Actual vs Predicted',
            xaxis_title='Actual Values',
            yaxis_title='Predicted Values',
            height=500
        )
        
        return fig
    except Exception as e:
        raise Exception(f"Error creating prediction plot: {str(e)}")

def get_feature_importance(model, feature_names):
    """Get feature importance for tree-based models"""
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            return pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
        return None
    except Exception as e:
        raise Exception(f"Error getting feature importance: {str(e)}")

def detect_outliers(df, column):
    """Detect outliers using IQR method"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

def create_outlier_plot(df, column):
    """Create box plot with outlier points highlighted"""
    outliers, lower_bound, upper_bound = detect_outliers(df, column)
    
    fig = go.Figure()
    
    # box plot
    fig.add_trace(go.Box(
        y=df[column],
        name=column,
        boxpoints=False,
        line_color='blue'
    ))
    
    # scatter plot for outliers
    if not outliers.empty:
        fig.add_trace(go.Scatter(
            y=outliers[column],
            mode='markers',
            name='Outliers',
            marker=dict(
                color='white',
                size=8,
                symbol='circle'
            )
        ))
    
    fig.update_layout(
        title=f'Box Plot with Outliers: {column}',
        yaxis_title=column,
        height=500,
        showlegend=True
    )
    
    return fig, outliers

# --- KMeans Clustering ---
def run_kmeans(df, n_clusters=3, features=None):
    """Run KMeans clustering and return cluster labels and centroids."""
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()
    
    X = df[features].copy()
    
    missing_rows = X.isnull().any(axis=1)
    
    X['original_index'] = df.index
    
    X_clean = X.dropna()
    
    if len(X_clean) == 0:
        raise ValueError("No complete rows found after removing missing values. Please handle missing values first.")
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean[features])
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    centroids = kmeans.cluster_centers_
    
    full_labels = pd.Series(index=df.index, dtype=float)
    full_labels.loc[X_clean['original_index']] = labels
    full_labels = full_labels.fillna(-1).astype(int)
    
    return full_labels, centroids, features

# --- Prophet Time Series Forecasting ---
def run_prophet(df, date_col, target_col, periods=30, freq='D'):
    """Run Prophet for time series forecasting."""
    data = df[[date_col, target_col]].copy()
    
    try:
        data['ds'] = pd.to_datetime(data[date_col])
    except Exception as e:
        raise ValueError(f"Could not convert {date_col} to datetime. Error: {str(e)}")
    
    data['y'] = data[target_col]
    data = data[['ds', 'y']].copy()
    
    # Drop NaN values and validate
    data = data.dropna()
    if len(data) < 2:
        raise ValueError(
            f"Insufficient data for forecasting. Found only {len(data)} valid rows after removing missing values. "
            "Need at least 2 rows with non-missing date and target values."
        )
    
    # Sort by date
    data = data.sort_values('ds')
    
    # Validate that we have enough unique timestamps
    unique_dates = data['ds'].nunique()
    if unique_dates < 2:
        raise ValueError(
            f"Need at least 2 different timestamps for forecasting. Found only {unique_dates} unique dates."
        )
    
    # Check for minimum date range
    date_range = (data['ds'].max() - data['ds'].min()).days
    if date_range < 1:
        raise ValueError(
            f"Data spans less than 1 day. Need data points from different dates for forecasting."
        )
    
    # Fit Prophet model
    try:
        model = Prophet(
            yearly_seasonality='auto',
            weekly_seasonality='auto',
            daily_seasonality='auto',
            seasonality_mode='additive'
        )
        model.fit(data)
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=periods, freq=freq)
        forecast = model.predict(future)
        return model, forecast
        
    except Exception as e:
        raise Exception(f"Error during Prophet modeling: {str(e)}")

# --- Time Series Analysis Utilities ---
def plot_trend(df, date_col, target_col):
    df_agg = df.groupby(date_col)[target_col].sum().reset_index()
    fig = px.line(df_agg, x=date_col, y=target_col, title="Trend Plot (Aggregated)")
    return fig

def plot_seasonality(df, date_col, target_col, freq='D'):
    df_agg = df.groupby(date_col)[target_col].sum().reset_index()
    result = seasonal_decompose(df_agg.set_index(date_col)[target_col], model='additive', period=7 if freq=='D' else 12)
    seasonality = result.seasonal.reset_index()
    seasonality.columns = [date_col, 'seasonal']
    fig = px.line(seasonality, x=date_col, y='seasonal', title="Seasonality Plot (Aggregated)")
    return fig

def plot_stationarity(df, date_col, target_col):
    df_agg = df.groupby(date_col)[target_col].sum().reset_index()
    rolling_mean = df_agg[target_col].rolling(window=12).mean()
    rolling_std = df_agg[target_col].rolling(window=12).std()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_agg[date_col], y=df_agg[target_col], mode='lines', name='Original'))
    fig.add_trace(go.Scatter(x=df_agg[date_col], y=rolling_mean, mode='lines', name='Rolling Mean'))
    fig.add_trace(go.Scatter(x=df_agg[date_col], y=rolling_std, mode='lines', name='Rolling Std'))
    fig.update_layout(title="Stationarity (Rolling Mean & Std, Aggregated)", xaxis_title=date_col, yaxis_title=target_col)
    return fig

def adf_test(df, target_col):
    if isinstance(df.index, pd.DatetimeIndex) or 'date' in df.columns or 'Date' in df.columns:
        if 'date' in df.columns:
            df_agg = df.groupby('date')[target_col].sum().reset_index()
        elif 'Date' in df.columns:
            df_agg = df.groupby('Date')[target_col].sum().reset_index()
        else:
            df_agg = df.copy()
        result = adfuller(df_agg[target_col].dropna())
    else:
        result = adfuller(df[target_col].dropna())
    return {'ADF Statistic': result[0], 'p-value': result[1]}

def plot_acf(df, target_col, lags=40):
    df_agg = df.copy()
    if 'date' in df_agg.columns or 'Date' in df_agg.columns:
        if 'date' in df_agg.columns:
            df_agg = df_agg.groupby('date')[target_col].sum().reset_index()
        elif 'Date' in df_agg.columns:
            df_agg = df_agg.groupby('Date')[target_col].sum().reset_index()
    acf_vals = acf(df_agg[target_col].dropna(), nlags=lags)
    fig = px.bar(x=list(range(len(acf_vals))), y=acf_vals, title="Autocorrelation (ACF, Aggregated)", labels={'x':'Lag', 'y':'ACF'})
    return fig

def plot_pacf(df, target_col, lags=40):
    df_agg = df.copy()
    if 'date' in df_agg.columns or 'Date' in df_agg.columns:
        if 'date' in df_agg.columns:
            df_agg = df_agg.groupby('date')[target_col].sum().reset_index()
        elif 'Date' in df_agg.columns:
            df_agg = df_agg.groupby('Date')[target_col].sum().reset_index()
    pacf_vals = pacf(df_agg[target_col].dropna(), nlags=lags)
    fig = px.bar(x=list(range(len(pacf_vals))), y=pacf_vals, title="Partial Autocorrelation (PACF, Aggregated)", labels={'x':'Lag', 'y':'PACF'})
    return fig

def plot_correlation(df):
    corr = df.corr()
    fig = px.imshow(
        corr,
        text=np.round(corr, 2),  
        aspect="auto",  
        color_continuous_scale='RdBu_r', 
        title="Correlation Heatmap"
    )

    fig.update_traces(texttemplate="%{text}")
    fig.update_layout(
        height=600,
        width=800,
    )

    return fig

