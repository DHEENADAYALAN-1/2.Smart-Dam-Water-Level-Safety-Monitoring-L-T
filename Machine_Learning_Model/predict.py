"""
Simple script to make predictions on any dataset
Just configure your dataset details and run!
"""
import pandas as pd
from generic_predictor import DataProcessor

# ============ CONFIGURATION ============
# Change these based on YOUR dataset

INPUT_FILE = "Location1.csv"  # Your dataset file
FILE_TYPE = "csv"             # 'csv', 'excel', or 'json'

# Map your column names to model features
COLUMN_MAPPING = {
    "field1": "rainfall",      # old_name: new_name
    "field2": "water_level",
    "field3": "pressure"
}

# Which columns to use for predictions
FEATURE_COLUMNS = ['rainfall', 'pressure']

# Risk classification thresholds
RISK_THRESHOLDS = {
    'HIGH': 300,
    'MEDIUM': 200
}

# Model weights (must sum to 1.0)
WEIGHTS = (0.6, 0.4)  # (XGBoost weight, LightGBM weight)

OUTPUT_FILE = "Location1_predictions.csv"

# ============ RUN PREDICTION ============

def main():
    print("\n🚀 Starting Predictions...")
    
    # Initialize predictor
    predictor = DataProcessor(models_path={
        'xgb': 'xgb_model.pkl',
        'lgb': 'lgb_model.pkl'
    })
    
    # Load your dataset
    print(f"\n📂 Loading {INPUT_FILE}...")
    df = predictor.load_data(INPUT_FILE, file_type=FILE_TYPE)
    print(f"   Dataset shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    
    # Process and predict
    print("\n🔄 Processing and predicting...")
    df_result = predictor.process_and_predict(
        df,
        feature_columns=FEATURE_COLUMNS,
        column_mapping=COLUMN_MAPPING,
        drop_na_cols=FEATURE_COLUMNS + ['water_level'],
        risk_thresholds=RISK_THRESHOLDS,
        weights=WEIGHTS
    )
    
    # Display results
    print("\n📊 Predictions Summary:")
    print(df_result[FEATURE_COLUMNS + ['predicted_value', 'risk']].head(15))
    
    print("\n📈 Risk Distribution:")
    print(df_result['risk'].value_counts())
    
    # Save results
    df_result.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Predictions saved to {OUTPUT_FILE}")
    
    return df_result

if __name__ == "__main__":
    results = main()
