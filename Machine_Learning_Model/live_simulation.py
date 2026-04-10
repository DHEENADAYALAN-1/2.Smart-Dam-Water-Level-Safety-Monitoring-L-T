import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
from azure.storage.blob import BlobServiceClient
import os

# Load models
xgb_model = pickle.load(open("xgb_model.pkl", "rb"))
lgb_model = pickle.load(open("lgb_model.pkl", "rb"))

# Function to classify risk
def classify_risk(water, pressure):
    if water > 300 or pressure > 400:
        return "HIGH"
    elif water > 200:
        return "MEDIUM"
    else:
        return "LOW"

# Load existing data for simulation
df = pd.read_csv("processed_data.csv")
df = df.rename(columns={'in_flow': 'inflow', 'out_flow': 'outflow', 'created_at': 'timestamp'})
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Simulate real-time data: generate 10 new data points
last_timestamp = df['timestamp'].max()
new_data = []

for i in range(10):
    # Simulate based on last values
    last_row = df.iloc[-1]
    rainfall = max(0, last_row['rainfall'] + np.random.randint(-2, 3))
    rainfall = np.random.poisson(rainfall)
    water_level = last_row['water_level'] + rainfall * 0.6 - last_row['outflow'] * 0.1 + np.random.normal(0, 2)
    water_level = max(200, water_level)  # min level
    pressure = water_level * 1.3 + np.random.normal(0, 5)  # approximate pressure

    timestamp = last_timestamp + timedelta(minutes=i+1)

    new_row = {
        'timestamp': timestamp,
        'rainfall': rainfall,
        'water_level': water_level,
        'pressure': pressure
    }
    new_data.append(new_row)

sim_df = pd.DataFrame(new_data)

# Features for prediction
X_sim = sim_df[['rainfall', 'pressure']]  # Exclude water_level

# Predict
xgb_pred = xgb_model.predict(X_sim)
lgb_pred = lgb_model.predict(X_sim)

# Hybrid prediction (weighted)
hybrid_pred = 0.6 * xgb_pred + 0.4 * lgb_pred

# Classify risk
sim_df['predicted_water_level'] = hybrid_pred
sim_df['risk'] = sim_df.apply(lambda row: classify_risk(row['predicted_water_level'], row['pressure']), axis=1)

print("Simulated Live Data Predictions:")
print(sim_df)

# sim_df.to_csv("live_predictions.csv", index=False)

print("Live predictions generated (not saved due to permission).")

# Upload to Azure Blob Storage
# Note: Replace with actual connection string or use environment variables
connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
if connection_string:
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_name = "dam-monitoring"
    blob_name = "live_predictions.csv"

    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

    with open("live_predictions.csv", "rb") as data:
        blob_client.upload_blob(data, overwrite=True)

    print("Uploaded to Azure Blob Storage successfully.")
else:
    print("Azure connection string not found. Skipping upload. Set AZURE_STORAGE_CONNECTION_STRING environment variable.")

print("Milestone 3: Live Data Simulation & Model Integration Completed!")