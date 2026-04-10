import requests
import pandas as pd
import numpy as np
import pickle
import time

# ==============================
# 🔑 CONFIGURATION
# ==============================

CHANNEL_ID = "3331828"
API_KEY = "PJPJJIIT3K1PYUB0"

FETCH_URL = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json?api_key={API_KEY}&results=50"

# Power BI Streaming Dataset URL (get from Power BI service)
POWER_BI_URL = "YOUR_POWER_BI_STREAMING_URL"  # e.g., https://api.powerbi.com/beta/.../datasets/.../rows?key=...

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
# 💾 SAVE TO CSV
# ==============================

def save_data(df):
    df.to_csv("live_data.csv", index=False)
    latest = df.iloc[-1]
    print(f"✅ Data updated and saved ({len(df)} rows). Latest: {latest['timestamp']}, water={latest['water_level']}, risk={latest['risk']}")

# ==============================
# 📊 PUSH TO POWER BI STREAMING
# ==============================

def push_to_power_bi(df):
    if POWER_BI_URL == "YOUR_POWER_BI_STREAMING_URL":
        print("⚠️ Power BI URL not configured - skipping push")
        return

    latest = df.iloc[-1].to_dict()
    payload = [latest]

    try:
        response = requests.post(POWER_BI_URL, json=payload, timeout=15)
        if response.status_code == 200:
            print("✅ Data pushed to Power BI!")
        else:
            print(f"❌ Power BI push failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Power BI error: {e}")

# ==============================
# 🔁 MAIN LOOP (REAL-TIME)
# ==============================

while True:
    try:
        print("📡 Fetching data from ThingSpeak...")
        
        df = fetch_data()
        df = apply_model(df)

        save_data(df)
        push_to_power_bi(df)

        print("⏳ Waiting for next update...\n")
        time.sleep(30)  # update every 30 seconds

    except Exception as e:
        print("❌ Error:", e)
        time.sleep(10)