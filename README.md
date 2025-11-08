# InsightAI
Insight AI — An AI Powered tool that automatically explores data, visualizes insights, runs ML models, and predicts future trends using Prophet.

# 🤖 Insight AI — Automated Data Analysis & ML Web App
**Live Demo:** [https://insightai-6mzi2dcghbedqvduf4kyft.streamlit.app](https://insightai-6mzi2dcghbedqvduf4kyft.streamlit.app)

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

##🖼️ Screenshots (Preview)
<table align="center"> <tr> <td align="center"> <b>📊 Data Overview</b><br> <img src="Assests/Tab1/data_overview.png" width="450"> </td> <td align="center"> <b>📈 Visualizations</b><br> <img src="Assests/Tab 2 imgs/visualization_tab.png" width="450"> </td> </tr> <tr> <td align="center"> <b>🤖 Supervised Learning</b><br> <img src="Assests/Tab 3/supervised_learning.png" width="450"> </td> <td align="center"> <b>🧩 Clustering</b><br> <img src="Assests/Tab 4/clustering_tab.png" width="450"> </td> </tr> <tr> <td colspan="2" align="center"> <b>🔮 Future Prediction</b><br> <img src="Assests/Tab 5/forecasting_tab.png" width="600"> </td> </tr> </table>
## 🛠 Tech Stack
scikit-learn · XGBoost · Prophet · Pandas · NumPy · Plotly · Matplotlib · Statsmodels · Streamlit

## 🚀 Run Locally
```bash
git clone https://github.com/Lovepreetin/InsightAI
cd InsightAI
pip install -r requirements.txt
streamlit run app.py
