import requests
import pandas as pd
import numpy as np
import pickle

# ==============================
# 🔑 CONFIGURATION
# ==============================

CHANNEL_ID = "YOUR_CHANNEL_ID"
API_KEY = "YOUR_API_KEY"

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
# 📡 FETCH DATA FROM THINGSPEAK (TEST VERSION: USE LOCAL DATA)
# ==============================

def fetch_data():
    # For testing, use local processed_data.csv instead of API
    df = pd.read_csv("processed_data.csv")
    
    # Select relevant columns
    df = df[['created_at', 'rainfall', 'water_level', 'pressure']].head(50)  # limit to 50 for test
    
    df = df.rename(columns={'created_at': 'timestamp'})
    
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
# TEST RUN (ONE TIME)
# ==============================

if __name__ == "__main__":
    try:
        print("📡 Fetching data from ThingSpeak...")
        
        df = fetch_data()
        df = apply_model(df)

        save_data(df)

        print(df.tail())
        print("✅ Test run completed successfully!")

    except Exception as e:
        print("❌ Error:", e)