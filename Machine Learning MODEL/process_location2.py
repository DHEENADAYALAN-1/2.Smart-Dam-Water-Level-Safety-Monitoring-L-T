"""
Complete Processing Pipeline for Location2.csv Sensor Data
"""
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from typing import Dict, List, Tuple

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
            print(f"   Timestamp range: {df['created_at'].min()} to {df['created_at'].max()}")
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
        return df
    
    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived features like in_flow, out_flow, gate_status"""
        df = df.copy()
        
        df['prev_level'] = df['water_level'].shift(1).fillna(df['water_level'])
        df['in_flow'] = df['rainfall'] * 0.6 + (df['water_level'] - df['prev_level'])
        df['gate_status'] = df['water_level'].apply(lambda x: 1 if x > 220 else 0)
        df['out_flow'] = df.apply(
            lambda row: row['in_flow'] * (0.8 if row['gate_status'] else 0.2),
            axis=1
        )
        
        print(f"✅ Derived features added")
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with missing values in critical columns"""
        initial_rows = len(df)
        df = df.dropna(subset=['rainfall', 'water_level', 'pressure'])
        removed = initial_rows - len(df)
        
        print(f"✅ Data cleaned: {removed} rows removed, {len(df)} remaining")
        return df
    
    def predict_water_level(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions using both models"""
        X = df[['rainfall', 'pressure']]
        
        xgb_pred = self.xgb_model.predict(X)
        lgb_pred = self.lgb_model.predict(X)
        
        hybrid_pred = (0.6 * xgb_pred) + (0.4 * lgb_pred)
        
        print(f"✅ Predictions generated")
        
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
        """Complete pipeline"""
        print("\n" + "="*70)
        print("🚀 SMART DAM DATA PROCESSING PIPELINE")
        print("="*70)
        
        df = self.load_data(input_file)
        df = self.rename_columns(df)
        df = self.add_derived_features(df)
        df = self.clean_data(df)
        
        predictions = self.predict_water_level(df)
        df['predicted_water_level'] = predictions
        df['risk_classification'] = self.classify_risk(df)
        
        print(f"\n📊 SUMMARY STATISTICS:")
        print(f"   - Actual water level range: {df['water_level'].min():.1f} - {df['water_level'].max():.1f}")
        print(f"   - Predicted water level range: {predictions.min():.1f} - {predictions.max():.1f}")
        print(f"\n   Risk Distribution:")
        print(df['risk_classification'].value_counts().to_string())
        
        if output_file is None:
            output_file = input_file.replace('.csv', '_processed.csv')
        
        df.to_csv(output_file, index=False)
        print(f"\n✅ Results saved to: {output_file}")
        
        print("\n" + "="*70)
        print("📈 PREVIEW OF RESULTS (First 15 rows):")
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
    
    results = processor.process_and_predict(
        input_file='Location2.csv',
        output_file='Location2_processed.csv'
    )
    
    return results


if __name__ == "__main__":
    results = main()
