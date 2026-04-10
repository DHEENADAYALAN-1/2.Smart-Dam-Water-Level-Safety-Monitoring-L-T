import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("processed_data.csv")

# Rename columns to match script
df = df.rename(columns={'in_flow': 'inflow', 'out_flow': 'outflow', 'created_at': 'timestamp'})

# Remove unwanted columns (if any)
df = df.dropna(subset=['rainfall', 'water_level', 'pressure'])

# Convert timestamp if exists
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'])

# Features and target
X = df[['rainfall', 'pressure']]  # Exclude water_level to avoid data leakage
y = df['water_level']   # forecasting water level

print("\nData Types:\n", df.dtypes)

print("\nCleaned Data Preview:\n", df.head())

# df.to_csv("cleaned_dataset.csv", index=False)

print("\nData Cleaning Completed Successfully!")

plt.figure()
plt.scatter(df['rainfall'], df['water_level'])
plt.xlabel("Rainfall")
plt.ylabel("Water Level")
plt.title("Rainfall vs Water Level")
plt.savefig("rainfall_vs_water_level.png")
plt.close()

plt.figure()
plt.scatter(df['water_level'], df['pressure'])
plt.xlabel("Water Level")
plt.ylabel("Pressure")
plt.title("Water Level vs Pressure")
plt.savefig("water_level_vs_pressure.png")
plt.close()

plt.figure()
plt.scatter(df['inflow'], df['outflow'])
plt.xlabel("Inflow")
plt.ylabel("Outflow")
plt.title("Inflow vs Outflow")
plt.savefig("inflow_vs_outflow.png")
plt.close()

plt.figure()
df['risk'].value_counts().plot(kind='bar')
plt.xlabel("Risk Level")
plt.ylabel("Count")
plt.title("Risk Distribution")
plt.savefig("risk_distribution.png")
plt.close()

plt.figure()
# Assuming risk is categorical, but for scatter, perhaps map to numbers
risk_map = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2}
df['risk_num'] = df['risk'].map(risk_map)
plt.scatter(df['rainfall'], df['risk_num'])
plt.xlabel("Rainfall")
plt.ylabel("Risk (0=LOW,1=MEDIUM,2=HIGH)")
plt.title("Rainfall vs Risk Level")
plt.savefig("rainfall_vs_risk.png")
plt.close()

# ML MODEL BUILDING

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

xgb_model = XGBRegressor()
xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)

print("XGBoost RMSE:", np.sqrt(mean_squared_error(y_test, xgb_pred)))

lgb_model = LGBMRegressor()
lgb_model.fit(X_train, y_train)

lgb_pred = lgb_model.predict(X_test)

print("LightGBM RMSE:", np.sqrt(mean_squared_error(y_test, lgb_pred)))

# Save models
import pickle
pickle.dump(xgb_model, open("xgb_model.pkl", "wb"))
pickle.dump(lgb_model, open("lgb_model.pkl", "wb"))

print("Models saved successfully.")

# Skip LSTM for now due to TensorFlow installation issues
# Scale data
# scaler = MinMaxScaler()
# scaled_data = scaler.fit_transform(df[['rainfall', 'water_level', 'pressure']])

# Create sequences
# X_lstm = []
# y_lstm = []

# for i in range(10, len(scaled_data)):
#     X_lstm.append(scaled_data[i-10:i])
#     y_lstm.append(scaled_data[i][1])  # waterlevel

# X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

# Split
# split = int(0.8 * len(X_lstm))
# X_train_lstm, X_test_lstm = X_lstm[:split], X_lstm[split:]
# y_train_lstm, y_test_lstm = y_lstm[:split], y_lstm[split:]

# Model
# lstm_model = Sequential()
# lstm_model.add(LSTM(50, return_sequences=False, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
# lstm_model.add(Dense(1))

# lstm_model.compile(optimizer='adam', loss='mse')

# Train
# lstm_model.fit(X_train_lstm, y_train_lstm, epochs=10, batch_size=16, verbose=0)

# Predict
# lstm_pred = lstm_model.predict(X_test_lstm)

# LIGHTGBM IMPLEMENTATION

# Align predictions
min_len = min(len(xgb_pred), len(lgb_pred))  # removed lstm_pred

hybrid_pred = (
    xgb_pred[:min_len] +
    lgb_pred[:min_len]
) / 2  # simple average

y_true = y_test.values[:min_len]

print("Hybrid RMSE:", np.sqrt(mean_squared_error(y_true, hybrid_pred)))

def classify_risk(water, pressure):
    if water > 300 or pressure > 400:
        return "HIGH"
    elif water > 200:
        return "MEDIUM"
    else:
        return "LOW"

# Apply
df['risk'] = df.apply(lambda row: classify_risk(row['water_level'], row['pressure']), axis=1)

print(df[['water_level', 'pressure', 'risk']].head())

plt.figure(figsize=(10,5))
plt.plot(y_true, label="Actual")
plt.plot(hybrid_pred, label="Predicted")
plt.legend()
plt.title("Water Level Prediction")
plt.savefig("water_level_prediction.png")
plt.close()

from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

# 🔹 Calculate RMSE for each model
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
lgb_rmse = np.sqrt(mean_squared_error(y_test, lgb_pred))

# Skip LSTM
# lstm_rmse = ...

# Hybrid RMSE (already aligned earlier)
hybrid_rmse = np.sqrt(mean_squared_error(y_true, hybrid_pred))

# 🔹 Store results in a table
results = pd.DataFrame({
    "Model": ["XGBoost", "LightGBM", "Hybrid"],
    "RMSE": [xgb_rmse, lgb_rmse, hybrid_rmse]
})

# 🔹 Sort by best model
results = results.sort_values(by="RMSE")

# 🔹 Display
print(results)

# Skip LSTM rescaling
# lstm_pred_rescaled = ...

min_len = min(len(xgb_pred), len(lgb_pred))

xgb_p = xgb_pred[:min_len]
lgb_p = lgb_pred[:min_len]
# lstm_p = ...
y_true = y_test.values[:min_len]

# LSTM-(Time series) - skipped

hybrid_pred = (xgb_p + lgb_p) / 2

xgb_rmse = np.sqrt(mean_squared_error(y_true, xgb_p))
lgb_rmse = np.sqrt(mean_squared_error(y_true, lgb_p))
# lstm_rmse = ...
hybrid_rmse = np.sqrt(mean_squared_error(y_true, hybrid_pred))

results = pd.DataFrame({
    "Model": ["XGBoost", "LightGBM", "Hybrid"],
    "RMSE": [xgb_rmse, lgb_rmse, hybrid_rmse]
}).sort_values(by="RMSE")

print(results)

# Weighted hybrid
hybrid_pred = (0.6 * xgb_p) + (0.4 * lgb_p)

xgb_rmse = np.sqrt(mean_squared_error(y_true, xgb_p))
lgb_rmse = np.sqrt(mean_squared_error(y_true, lgb_p))
# lstm_rmse = ...
hybrid_rmse = np.sqrt(mean_squared_error(y_true, hybrid_pred))

results = pd.DataFrame({
    "Model": ["XGBoost", "LightGBM", "Hybrid"],
    "RMSE": [xgb_rmse, lgb_rmse, hybrid_rmse]
}).sort_values(by="RMSE")

print(results)