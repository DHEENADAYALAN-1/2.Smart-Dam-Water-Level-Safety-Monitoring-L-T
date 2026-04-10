"""
Visualization and Analysis of Location1 Predictions
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load results
df = pd.read_csv('Location1_processed.csv')

print("📈 Generating visualizations...")

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
fig = plt.figure(figsize=(16, 12))

# 1. Water Level Over Time (Actual vs Predicted)
ax1 = plt.subplot(3, 3, 1)
ax1.plot(range(len(df)), df['water_level'], 'b-', linewidth=2, label='Actual', marker='o', markersize=4)
ax1.plot(range(len(df)), df['predicted_water_level'], 'r--', linewidth=2, label='Predicted', marker='s', markersize=4)
ax1.set_xlabel('Time Index')
ax1.set_ylabel('Water Level')
ax1.set_title('Water Level: Actual vs Predicted', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Prediction Error Over Time
ax2 = plt.subplot(3, 3, 2)
errors = df['water_level'] - df['predicted_water_level']
ax2.plot(range(len(df)), errors, 'g-', linewidth=2, marker='o', markersize=4)
ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
ax2.set_xlabel('Time Index')
ax2.set_ylabel('Error (Actual - Predicted)')
ax2.set_title('Prediction Error Timeline', fontweight='bold')
ax2.grid(True, alpha=0.3)

# 3. Rainfall Over Time
ax3 = plt.subplot(3, 3, 3)
colors = ['green' if x < 30 else 'orange' if x < 45 else 'red' for x in df['rainfall']]
ax3.bar(range(len(df)), df['rainfall'], color=colors, alpha=0.7)
ax3.set_xlabel('Time Index')
ax3.set_ylabel('Rainfall (mm)')
ax3.set_title('Rainfall Pattern', fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# 4. Risk Distribution
ax4 = plt.subplot(3, 3, 4)
risk_counts = df['risk_classification'].value_counts()
colors_risk = {'HIGH': 'red', 'MEDIUM': 'orange', 'LOW': 'green'}
risk_colors = [colors_risk.get(x, 'gray') for x in risk_counts.index]
ax4.bar(risk_counts.index, risk_counts.values, color=risk_colors, alpha=0.7)
ax4.set_ylabel('Count')
ax4.set_title('Risk Classification Distribution', fontweight='bold')
for i, v in enumerate(risk_counts.values):
    ax4.text(i, v + 1, str(v), ha='center', fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

# 5. Rainfall vs Water Level
ax5 = plt.subplot(3, 3, 5)
scatter = ax5.scatter(df['rainfall'], df['water_level'], c=df['pressure'], cmap='viridis', s=100, alpha=0.6)
ax5.set_xlabel('Rainfall (mm)')
ax5.set_ylabel('Water Level')
ax5.set_title('Rainfall vs Water Level (colored by Pressure)', fontweight='bold')
plt.colorbar(scatter, ax=ax5, label='Pressure')

# 6. Pressure Over Time
ax6 = plt.subplot(3, 3, 6)
ax6.plot(range(len(df)), df['pressure'], 'purple', linewidth=2, marker='o', markersize=5)
ax6.fill_between(range(len(df)), df['pressure'], alpha=0.3, color='purple')
ax6.set_xlabel('Time Index')
ax6.set_ylabel('Pressure')
ax6.set_title('Pressure Pattern', fontweight='bold')
ax6.grid(True, alpha=0.3)

# 7. Inflow vs Outflow
ax7 = plt.subplot(3, 3, 7)
x = np.arange(len(df))
width = 0.35
ax7.bar(x - width/2, df['in_flow'], width, label='Inflow', alpha=0.8, color='blue')
ax7.bar(x + width/2, df['out_flow'], width, label='Outflow', alpha=0.8, color='orange')
ax7.set_xlabel('Time Index')
ax7.set_ylabel('Flow Rate')
ax7.set_title('Inflow vs Outflow', fontweight='bold')
ax7.legend()
ax7.grid(True, alpha=0.3, axis='y')

# 8. Water Level Distribution
ax8 = plt.subplot(3, 3, 8)
ax8.hist(df['water_level'], bins=15, color='skyblue', edgecolor='black', alpha=0.7)
ax8.axvline(df['water_level'].mean(), color='r', linestyle='--', linewidth=2, label=f'Mean: {df["water_level"].mean():.1f}')
ax8.set_xlabel('Water Level')
ax8.set_ylabel('Frequency')
ax8.set_title('Water Level Distribution', fontweight='bold')
ax8.legend()
ax8.grid(True, alpha=0.3, axis='y')

# 9. Gate Status Over Time
ax9 = plt.subplot(3, 3, 9)
gate_colors = ['green' if x == 0 else 'red' for x in df['gate_status']]
ax9.bar(range(len(df)), df['gate_status'], color=gate_colors, alpha=0.6)
ax9.set_xlabel('Time Index')
ax9.set_ylabel('Gate Status (0=Closed, 1=Open)')
ax9.set_title('Gate Status Timeline', fontweight='bold')
ax9.set_ylim(-0.1, 1.1)
ax9.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('Location1_analysis.png', dpi=150, bbox_inches='tight')
print("✅ Visualization saved: Location1_analysis.png")
plt.close()

# Generate detailed statistics CSV
print("\n📊 Generating detailed statistics...")

stats_df = pd.DataFrame({
    'Metric': [
        'Total Records',
        'Time Period',
        'Rainfall Min',
        'Rainfall Max',
        'Rainfall Mean',
        'Water Level Min',
        'Water Level Max',
        'Water Level Mean',
        'Pressure Min',
        'Pressure Max',
        'Pressure Mean',
        'Prediction Error Mean',
        'Prediction Error Std',
        'HIGH Risk Count',
        'MEDIUM Risk Count',
        'LOW Risk Count',
        'Gate Open Count',
        'Avg Inflow',
        'Avg Outflow',
    ],
    'Value': [
        len(df),
        f"{df['timestamp'].min()} to {df['timestamp'].max()}",
        f"{df['rainfall'].min():.1f}",
        f"{df['rainfall'].max():.1f}",
        f"{df['rainfall'].mean():.2f}",
        f"{df['water_level'].min():.1f}",
        f"{df['water_level'].max():.1f}",
        f"{df['water_level'].mean():.2f}",
        f"{df['pressure'].min():.1f}",
        f"{df['pressure'].max():.1f}",
        f"{df['pressure'].mean():.2f}",
        f"{(df['water_level'] - df['predicted_water_level']).mean():.2f}",
        f"{(df['water_level'] - df['predicted_water_level']).std():.2f}",
        (df['risk_classification'] == 'HIGH').sum(),
        (df['risk_classification'] == 'MEDIUM').sum(),
        (df['risk_classification'] == 'LOW').sum(),
        (df['gate_status'] == 1).sum(),
        f"{df['in_flow'].mean():.2f}",
        f"{df['out_flow'].mean():.2f}",
    ]
})

stats_df.to_csv('Location1_statistics.csv', index=False)
print("✅ Statistics saved: Location1_statistics.csv")

# Create hourly summary
print("\n⏰ Creating time-based summary...")
df['timestamp'] = pd.to_datetime(df['timestamp'])
time_summary = df.groupby(pd.Grouper(key='timestamp', freq='5min')).agg({
    'rainfall': 'mean',
    'water_level': 'mean',
    'pressure': 'mean',
    'predicted_water_level': 'mean',
    'in_flow': 'mean',
    'out_flow': 'mean',
    'entry_id': 'count'
}).rename(columns={'entry_id': 'records'})

time_summary.to_csv('Location1_time_summary.csv')
print("✅ Time summary saved: Location1_time_summary.csv")

print("\n" + "="*70)
print("✅ ALL OUTPUTS GENERATED:")
print("="*70)
print("  1. Location1_processed.csv - Complete results with all calculations")
print("  2. Location1_analysis.png - 9-panel visualization dashboard")
print("  3. Location1_statistics.csv - Detailed statistics summary")
print("  4. Location1_time_summary.csv - 5-minute aggregated summary")
print("="*70)
