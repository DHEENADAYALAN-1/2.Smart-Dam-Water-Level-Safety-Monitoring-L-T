"""
Generic ML Predictor - Works with various datasets and column configurations
"""
import pandas as pd
import numpy as np
import pickle
from typing import Dict, List, Optional, Tuple
import os

class DataProcessor:
    """Flexible data processor that works with different datasets"""
    
    def __init__(self, models_path: Dict[str, str]):
        """
        Initialize with paths to trained models
        models_path: dict like {'xgb': 'xgb_model.pkl', 'lgb': 'lgb_model.pkl'}
        """
        self.xgb_model = pickle.load(open(models_path.get('xgb', 'xgb_model.pkl'), 'rb'))
        self.lgb_model = pickle.load(open(models_path.get('lgb', 'lgb_model.pkl'), 'rb'))
    
    def load_data(self, file_path: str, file_type: str = 'csv') -> pd.DataFrame:
        """
        Load data from various file formats
        file_type: 'csv', 'excel', 'json'
        """
        try:
            if file_type.lower() == 'excel':
                df = pd.read_excel(file_path)
            elif file_type.lower() == 'csv':
                df = pd.read_csv(file_path)
            elif file_type.lower() == 'json':
                df = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            print(f"✅ Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise
    
    def preprocess(self, df: pd.DataFrame, column_mapping: Dict[str, str], 
                   drop_na_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Preprocess data with flexible column mapping
        
        Args:
            df: Input dataframe
            column_mapping: Dict mapping old column names to new names
                           e.g., {'Rainfall': 'rainfall', 'Waterlevel': 'water_level'}
            drop_na_cols: Columns to check for NaN values (default: all)
        
        Returns:
            Preprocessed dataframe
        """
        df = df.copy()
        
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        
        # Apply column mapping
        df = df.rename(columns=column_mapping)
        
        # Drop NaN values
        if drop_na_cols:
            df = df.dropna(subset=drop_na_cols)
        else:
            df = df.dropna()
        
        print(f"✅ Preprocessing complete. Rows after NaN removal: {len(df)}")
        return df
    
    def predict(self, df: pd.DataFrame, feature_columns: List[str], 
                weights: Tuple[float, float] = (0.6, 0.4)) -> np.ndarray:
        """
        Make predictions using hybrid model (XGBoost + LightGBM)
        
        Args:
            df: Input dataframe with features
            feature_columns: List of column names to use as features
            weights: (xgb_weight, lgb_weight) - must sum to 1
        
        Returns:
            Prediction array
        """
        # Validate features exist
        missing_cols = [col for col in feature_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"❌ Missing columns: {missing_cols}")
        
        X = df[feature_columns]
        
        # Get predictions from both models
        xgb_pred = self.xgb_model.predict(X)
        lgb_pred = self.lgb_model.predict(X)
        
        # Weighted average
        hybrid_pred = (weights[0] * xgb_pred) + (weights[1] * lgb_pred)
        
        print(f"✅ Predictions generated. Shape: {hybrid_pred.shape}")
        return hybrid_pred
    
    def classify_risk(self, predictions: np.ndarray, 
                     thresholds: Dict[str, float]) -> List[str]:
        """
        Classify predictions into risk categories
        
        Args:
            predictions: Array of predicted values
            thresholds: Dict like {'HIGH': 300, 'MEDIUM': 200}
        
        Returns:
            List of risk classifications
        """
        risk = []
        high_threshold = thresholds.get('HIGH', 300)
        medium_threshold = thresholds.get('MEDIUM', 200)
        
        for pred in predictions:
            if pred > high_threshold:
                risk.append('HIGH')
            elif pred > medium_threshold:
                risk.append('MEDIUM')
            else:
                risk.append('LOW')
        
        return risk
    
    def process_and_predict(self, df: pd.DataFrame, 
                           feature_columns: List[str],
                           column_mapping: Optional[Dict[str, str]] = None,
                           drop_na_cols: Optional[List[str]] = None,
                           risk_thresholds: Optional[Dict[str, float]] = None,
                           weights: Tuple[float, float] = (0.6, 0.4)) -> pd.DataFrame:
        """
        End-to-end pipeline: preprocess -> predict -> classify
        """
        # Preprocess if mapping provided
        if column_mapping:
            df = self.preprocess(df, column_mapping, drop_na_cols)
        
        # Make predictions
        predictions = self.predict(df, feature_columns, weights)
        df['predicted_value'] = predictions
        
        # Classify risk if thresholds provided
        if risk_thresholds:
            df['risk'] = self.classify_risk(predictions, risk_thresholds)
        
        return df


# ============ USAGE EXAMPLES ============

if __name__ == "__main__":
    
    print("\n" + "="*60)
    print("EXAMPLE 1: Dam Dataset (Original)")
    print("="*60)
    
    predictor = DataProcessor(models_path={
        'xgb': 'xgb_model.pkl',
        'lgb': 'lgb_model.pkl'
    })
    
    # Load and process dam data
    df_dam = predictor.load_data("feeds.csv", file_type='excel')
    
    # Map columns from your original names
    column_mapping = {
        "Rainfall": "rainfall",
        "Waterlevel": "water_level",
        "Force": "pressure"
    }
    
    # Process and predict
    df_result = predictor.process_and_predict(
        df_dam,
        feature_columns=['rainfall', 'pressure'],
        column_mapping=column_mapping,
        drop_na_cols=['rainfall', 'water_level', 'pressure'],
        risk_thresholds={'HIGH': 300, 'MEDIUM': 200}
    )
    
    print("\n📊 Results:")
    print(df_result[['rainfall', 'pressure', 'predicted_value', 'risk']].head(10))
    
    # Save results
    df_result.to_csv("predictions_output.csv", index=False)
    print("✅ Results saved to predictions_output.csv")
    
    
    print("\n" + "="*60)
    print("EXAMPLE 2: User-Provided Dataset (Any format)")
    print("="*60)
    print("""
    # For a different dataset with different column names:
    
    df_custom = predictor.load_data("my_data.csv", file_type='csv')
    
    # Map YOUR column names to expected features
    custom_mapping = {
        "Feature1": "rainfall",
        "Feature2": "pressure",  # or whatever features your models expect
    }
    
    df_result = predictor.process_and_predict(
        df_custom,
        feature_columns=['rainfall', 'pressure'],
        column_mapping=custom_mapping,
        risk_thresholds={'HIGH': 300, 'MEDIUM': 200}
    )
    
    # Save and use predictions
    df_result.to_csv("custom_predictions.csv", index=False)
    """)
