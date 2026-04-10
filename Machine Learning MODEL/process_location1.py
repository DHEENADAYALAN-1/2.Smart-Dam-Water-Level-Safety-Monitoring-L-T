"""
Complete Processing Pipeline for Location1.csv Sensor Data
Includes: Data loading → Processing → Predictions → Risk Classification → Outputs
"""
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from typing import Dict, List, Tuple
import os

class SmartDamProcessor:
    """Complete pipeline for processing sensor data and generating predictions"""
    
    def __init__(self, models_path: Dict[str, str]):
        """Load pre-trained models"""
        self.xgb_model = pickle.load(open(models_path.get('xgb', 'xgb_model.pkl'), 'rb'))
        self.lgb_model = pickle.load(open(models_path.get('lgb', 'lgb_model.pkl'), 'rb'))
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load CSV data"""
        try:
            df = pd.read_csv(file_path)
            print(f"✅ Data loaded: {file_path}")
            print(f"   Shape: {df.shape}")
            print(f"   Timestampe range: {df['created_at'].min()} to {df['created_at'].max()}")
            return df
        except Exception as e:
            print(f"❌ Error: {e}")
            raise
    
    def rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename sensor fields to standard names"""
        df = df.copy()
        df.columns = df.columns.str.strip()
        
        rename_map = {
            'field1': 'rainfall',
            'field2': 'water_level',
            'field3': 'pressure',
            'created_at': 'timestamp'
        }
        
        df = df.rename(columns=rename_map)
        print(f"\n✅ Columns renamed")
        print(f"   Available: {list(df.columns)}")
        return df
    
    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived features like in_flow, out_flow, gate_status"""
        df = df.copy()
        
        # Calculate inflow and outflow (as in original process_data.py)
        df['prev_level'] = df['water_level'].shift(1).fillna(df['water_level'])
        df['in_flow'] = df['rainfall'] * 0.6 + (df['water_level'] - df['prev_level'])
        df['gate_status'] = df['water_level'].apply(lambda x: 1 if x > 220 else 0)
        df['out_flow'] = df.apply(
            lambda row: row['in_flow'] * (0.8 if row['gate_status'] else 0.2),
            axis=1
        )
        
        print(f"✅ Derived features added:")
        print(f"   - inflow, outflow, gate_status")
        print(f"   - Features shape: {df[['in_flow', 'out_flow', 'gate_status']].describe()}")
        
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with missing values in critical columns"""
        initial_rows = len(df)
        df = df.dropna(subset=['rainfall', 'water_level', 'pressure'])
        removed = initial_rows - len(df)
        
        print(f"\n✅ Data cleaned:")
        print(f"   - Rows removed: {removed}")
        print(f"   - Rows remaining: {len(df)}")
        return df
    
    def predict_water_level(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions using both models"""
        X = df[['rainfall', 'pressure']]
        
        xgb_pred = self.xgb_model.predict(X)
        lgb_pred = self.lgb_model.predict(X)
        
        # Hybrid prediction
        hybrid_pred = (0.6 * xgb_pred) + (0.4 * lgb_pred)
        
        print(f"\n✅ Predictions generated:")
        print(f"   - XGBoost RMSE: {np.sqrt(np.mean((xgb_pred - df['water_level'])**2)):.3f}")
        print(f"   - LightGBM RMSE: {np.sqrt(np.mean((lgb_pred - df['water_level'])**2)):.3f}")
        print(f"   - Hybrid RMSE: {np.sqrt(np.mean((hybrid_pred - df['water_level'])**2)):.3f}")
        
        return hybrid_pred
    
    def classify_risk(self, df: pd.DataFrame) -> List[str]:
        """Classify risk based on water level and pressure"""
        risk = []
        
        for _, row in df.iterrows():
            water = row['water_level']
            pressure = row['pressure']
            
            if water > 300 or pressure > 400:
                risk.append('HIGH')
            elif water > 200:
                risk.append('MEDIUM')
            else:
                risk.append('LOW')
        
        return risk
    
    def process_and_predict(self, input_file: str, output_file: str = None) -> pd.DataFrame:
        """
        Complete pipeline:
        Load → Rename → Clean → Add Features → Predict → Classify → Save
        """
        print("\n" + "="*70)
        print("🚀 SMART DAM DATA PROCESSING PIPELINE")
        print("="*70)
        
        # Load data
        df = self.load_data(input_file)
        
        # Rename columns
        df = self.rename_columns(df)
        
        # Add derived features
        df = self.add_derived_features(df)
        
        # Clean data
        df = self.clean_data(df)
        
        # Make predictions
        predictions = self.predict_water_level(df)
        df['predicted_water_level'] = predictions
        
        # Classify risk
        df['risk_classification'] = self.classify_risk(df)
        
        # Display summary statistics
        print(f"\n📊 SUMMARY STATISTICS:")
        print(f"   - Actual water level range: {df['water_level'].min():.1f} - {df['water_level'].max():.1f}")
        print(f"   - Predicted water level range: {predictions.min():.1f} - {predictions.max():.1f}")
        print(f"\n   Risk Distribution:")
        print(df['risk_classification'].value_counts().to_string())
        
        # Save results
        if output_file is None:
            output_file = input_file.replace('.csv', '_processed.csv')
        
        df.to_csv(output_file, index=False)
        print(f"\n✅ Results saved to: {output_file}")
        
        print("\n" + "="*70)
        print("📈 PREVIEW OF RESULTS:")
        print("="*70)
        display_cols = ['timestamp', 'rainfall', 'water_level', 'pressure', 
                       'predicted_water_level', 'in_flow', 'out_flow', 'risk_classification']
        print(df[display_cols].head(15).to_string(index=False))
        
        return df


def main():
    """Main execution"""
    processor = SmartDamProcessor(models_path={
        'xgb': 'xgb_model.pkl',
        'lgb': 'lgb_model.pkl'
    })
    
    # Process Location1.csv
    results = processor.process_and_predict(
        input_file='Location1.csv',
        output_file='Location1_processed.csv'
    )
    
    # Also generate a summary report
    generate_report(results)
    
    return results


def generate_report(df: pd.DataFrame):
    """Generate detailed analysis report"""
    print("\n" + "="*70)
    print("📋 DETAILED REPORT")
    print("="*70)
    
    # Risk-wise breakdown
    print("\n🎯 Risk Classification Breakdown:")
    for risk_level in ['HIGH', 'MEDIUM', 'LOW']:
        count = (df['risk_classification'] == risk_level).sum()
        pct = count / len(df) * 100
        avg_water = df[df['risk_classification'] == risk_level]['water_level'].mean()
        print(f"   {risk_level:10} : {count:3} records ({pct:5.1f}%) | Avg Water Level: {avg_water:.1f}")
    
    # Prediction accuracy metrics
    print("\n🎯 Prediction Metrics:")
    mae = np.mean(np.abs(df['predicted_water_level'] - df['water_level']))
    rmse = np.sqrt(np.mean((df['predicted_water_level'] - df['water_level'])**2))
    print(f"   MAE (Mean Absolute Error): {mae:.3f}")
    print(f"   RMSE (Root Mean Square Error): {rmse:.3f}")
    
    # Data quality
    print("\n✅ Data Quality:")
    print(f"   Total records: {len(df)}")
    print(f"   Timestamp range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"   Rainfall range: {df['rainfall'].min():.0f} - {df['rainfall'].max():.0f} mm")
    print(f"   Water level range: {df['water_level'].min():.1f} - {df['water_level'].max():.1f}")
    print(f"   Pressure range: {df['pressure'].min():.1f} - {df['pressure'].max():.1f}")
    
    # Anomalies
    print("\n⚠️ Anomalies Detected:")
    high_risk = (df['risk_classification'] == 'HIGH').sum()
    critical_water = (df['water_level'] > 240).sum()
    high_pressure = (df['pressure'] > 410).sum()
    print(f"   HIGH risk records: {high_risk}")
    print(f"   Water level > 240: {critical_water}")
    print(f"   Pressure > 410: {high_pressure}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    results = main()
