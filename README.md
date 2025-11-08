# InsightAI
Insight AI — An AI Powered tool that automatically explores data, visualizes insights, runs ML models, and predicts future trends using Prophet.

# 🤖 Insight AI — Automated Data Analysis & ML Web App
**Live Demo:** 

Insight AI is an end-to-end tool to upload any CSV and instantly:
- Summarize data (stats, nulls, shape)
- Visualize (correlation heatmap, boxplots, distributions, outliers)
- Auto-detect task (regression vs. classification) and select the best model
- Cluster with K-Means
- Forecast with Prophet (daily / weekly / monthly) + trend & seasonality

## ✨ Features
- **Overview:** columns, missing values, mean/median/quantiles, min/max
- **Visualization:** correlation heatmap, boxplots, distributions, outlier summary
- **Supervised (Auto-ML):** Linear/Logistic, Random Forest, XGBoost, Gradient Boost; auto-select best by metric
- **Unsupervised:** K-Means clustering with labels
- **Forecasting:** Prophet with seasonality, trend, ACF/PACF, rolling mean

## 🛠 Tech Stack
scikit-learn · XGBoost · Prophet · Pandas · NumPy · Plotly · Matplotlib · Statsmodels · Streamlit

## 🚀 Run Locally
```bash
git clone https://github.com/<your-username>/InsightAI.git
cd InsightAI
pip install -r requirements.txt
streamlit run app.py
