DAY 2 Milestone:
OVERVIEW :
Dataset Generation & ML Model Development
Generate and pre-process synthetic historical data for rainfall, reservoir levels,
and pressure.
Train and validate an ML model for water-level forecasting and risk classification.
Synthetic Dataset:
smart_dam_dataset_updated.xlsx
Dataset Description:
Features:
rainfall
temperature
humidity
inflow
outflow
pressure
water_level
risk
7
Smart Dam Water Level & Safety Monitoring (L&T Internship)
Machine Learning Model:
https://colab.research.google.com/drive/1Dzn0P6uV9gg8WqQk397ScB_
dRt3wOy7?sharingaction=ownershiptransfer
Exploratory Data Analysis (EDA):
Scatter plots:
Rainfall vs Water Level
Water Level vs Pressure
Inflow vs Outflow
8
Smart Dam Water Level & Safety Monitoring (L&T Internship)
Smart Dam Water Level & Safety Monitoring (L&T Internship)
9
Model Implementation :
Multiple machine learning models were implemented to improve prediction
accuracy and reliability.
1. LightGBM Model
Used for regression (water level prediction)
Handles non-linear relationships well
Provides high accuracy
Use: Predict future water level
2. XGBoost Model
10
Smart Dam Water Level & Safety Monitoring (L&T Internship)
Faster and efficient gradient boosting model
Works well with large datasets
Use: Risk classification
3. LSTM Model (Time-Series)
Deep learning model for time-series data
Captures sequential patterns
Architecture:
LSTM Layer 50 units)
Dense Output Layer
Use: Forecast water levels over time
We combined outputs of multiple models:
Hybrid Model (Combined Approach)
Initially used a simple average of predictions from:
XGBoost
LightGBM
LSTM
After evaluation, optimized the hybrid model by selecting best-performing
models
Final Hybrid Model:
XGBoost 
→
 60%
LightGBM 
→
 40%
Used weighted average to generate final prediction
Purpose:
Predict water level in the dam system
11
Smart Dam Water Level & Safety Monitoring (L&T Internship)
Classify risk levels based on predictions
Improve accuracy using multiple ML models
Models used:
XGBoost (tree-based)
LightGBM (tree-based)
LSTM (deep learning)
Model Comparison
Evaluation metric used: RMSE Root Mean Squared Error)
Model
RMSE
Hybrid 60% XGBoost + 40% LightGBM 0.237
LightGBM
XGBoost
Performance
Best
0.316
0.327
LSTM
Output:
114.092
Good
Good
Poor (after rescaling)
12
Smart Dam Water Level & Safety Monitoring (L&T Internship)
