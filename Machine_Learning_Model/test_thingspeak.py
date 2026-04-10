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
    response = requests.get(FETCH_URL)
    data = response.json()

    df = pd.DataFrame(data['feeds'])

    df = df.rename(columns={
        'field1': 'rainfall',
        'field2': 'water_level',
        'field3': 'pressure',
        'created_at': 'timestamp'
    })

    df = df[['timestamp', 'rainfall', 'water_level', 'pressure']]

    df = df.dropna()

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
    print("✅ Data updated and saved!")

# ==============================
# TEST FETCH FROM THINGSPEAK
# ==============================

if __name__ == "__main__":
    try:
        print("📡 Testing connection to ThingSpeak...")
        
        df = fetch_data()
        print(f"✅ Fetched {len(df)} records from ThingSpeak!")
        print("Sample data:")
        print(df.head())
        
        df = apply_model(df)
        save_data(df)
        
        print("✅ ML predictions applied and data saved!")
        print("Latest predictions:")
        print(df[['timestamp', 'water_level', 'predicted_water_level', 'risk']].tail())

    except Exception as e:
        print("❌ Error:", e)