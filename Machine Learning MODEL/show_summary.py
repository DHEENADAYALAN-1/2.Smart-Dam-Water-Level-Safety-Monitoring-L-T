import pandas as pd

df = pd.read_csv('live_data.csv')

print('\n📈 FINAL DATA SUMMARY')
print('='*50)
print(f'Total rows: {len(df)}')
print(f'HIGH risk: {len(df[df["risk"] == "HIGH"])} records')
print(f'MEDIUM risk: {len(df[df["risk"] == "MEDIUM"])} records') 
print(f'LOW risk: {len(df[df["risk"] == "LOW"])} records')
print('='*50)
print('\nLatest 15 predictions:')
print(df.tail(15)[['timestamp', 'rainfall', 'water_level', 'predicted_water_level', 'risk']].to_string())
