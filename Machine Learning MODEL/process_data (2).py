import pandas as pd

print("Starting script...")

try:
    df = pd.read_excel("feeds.csv")  # Use read_excel as the file is Excel format
    print("File loaded successfully")
except Exception as e:
    print("Error loading file:", e)
    exit()

df.columns = df.columns.str.strip()

print("Columns:", df.columns)

df = df.rename(columns={
    "Rainfall": "rainfall",
    "Waterlevel": "water_level",
    "Force": "pressure"
})

df['prev_level'] = df['water_level'].shift(1).fillna(df['water_level'])

df['in_flow'] = df['rainfall'] * 0.6 + (df['water_level'] - df['prev_level'])

df['gate_status'] = df['water_level'].apply(lambda x: 1 if x > 220 else 0)

df['out_flow'] = df.apply(
    lambda row: row['in_flow'] * (0.8 if row['gate_status'] else 0.2),
    axis=1
)

def risk_level(w):
    if w > 230:
        return "HIGH"
    elif w > 200:
        return "MEDIUM"
    else:
        return "LOW"

df['risk'] = df['water_level'].apply(risk_level)

# Drop rows with NaN in key columns
df = df.dropna(subset=['rainfall', 'water_level', 'pressure'])

df.to_csv("processed_data.csv", index=False)

print("✅ Done! File saved as processed_data.csv")
print(df.head())

df.to_csv("processed_data.csv", index=False)

print("✅ Done! File saved as processed_data.csv")
print(df.head())