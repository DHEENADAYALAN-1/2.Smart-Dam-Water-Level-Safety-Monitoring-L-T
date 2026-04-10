import requests
import pandas as pd
import numpy as np
import pickle

# ==============================
# 🔑 CONFIGURATION
# ==============================

CHANNEL_ID = "3331828"
API_KEY = "PJPJJIIT3K1PYUB0"

FETCH_URL = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json?api_key={API_KEY}&results=50"

# ==============================
# 🤖 LOAD ML MODELS
# ==============================

xgb_model = pickle.load(open("xgb_model.pkl", "rb"))
lgb_model = pickle.load(open("lgb_model.pkl", "rb"))

# ==============================
# 🚨 RISK FUNCTION
# ==============================

def classify_risk(row):
    if row['water_level'] > 300 or row['pressure'] > 400:
        return "HIGH"
    elif row['water_level'] > 200:
        return "MEDIUM"
    else:
        return "LOW"

# ==============================
# 📡 FETCH DATA FROM THINGSPEAK
# ==============================

def fetch_data():
    response = requests.get(FETCH_URL, timeout=15)
    response.raise_for_status()
    data = response.json()

    df = pd.DataFrame(data.get('feeds', []))
    if df.empty:
        raise ValueError("No data returned from ThingSpeak")

    df = df.rename(columns={
        'field1': 'rainfall',
        'field2': 'water_level',
        'field3': 'pressure',
        'created_at': 'timestamp'
    })

    df = df[['timestamp', 'rainfall', 'water_level', 'pressure']]
    df = df.dropna(subset=['timestamp', 'rainfall', 'water_level', 'pressure'])

    # Convert to numeric
    df['rainfall'] = df['rainfall'].astype(float)
    df['water_level'] = df['water_level'].astype(float)
    df['pressure'] = df['pressure'].astype(float)

    return df

# ==============================
# 🤖 PREDICTION FUNCTION
# ==============================

def apply_model(df):
    X = df[['rainfall', 'pressure']]

    xgb_pred = xgb_model.predict(X)
    lgb_pred = lgb_model.predict(X)

    df['predicted_water_level'] = (0.6 * xgb_pred) + (0.4 * lgb_pred)

    df['risk'] = df.apply(classify_risk, axis=1)

    return df

# ==============================
# 💾 APPEND TO CSV (ADD MORE ROWS)
# ==============================

if __name__ == "__main__":
    try:
        print("📡 Fetching 50 new records from ThingSpeak...")
        
        new_data = fetch_data()
        new_data = apply_model(new_data)
        
        # Load existing data
        try:
            existing_data = pd.read_csv("live_data.csv")
            print(f"✅ Loaded existing {len(existing_data)} rows")
        except FileNotFoundError:
            existing_data = pd.DataFrame()
            print("✅ Creating new CSV file")
        
        # Append new data
        updated_data = pd.concat([existing_data, new_data], ignore_index=True)
        
        # Remove duplicates based on timestamp
        updated_data = updated_data.drop_duplicates(subset=['timestamp'], keep='last')
        
        # Save updated data
        updated_data.to_csv("live_data.csv", index=False)
        
        print(f"✅ Updated live_data.csv!")
        print(f"   Total rows now: {len(updated_data)}")
        print(f"\n📊 Latest 10 records:")
        print(updated_data.tail(10).to_string())

    except Exception as e:
        print("❌ Error:", e)
