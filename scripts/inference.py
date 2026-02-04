"""
================================================================================
INFERENCE SCRIPT
================================================================================
Project: Customer Spend Prediction (30-Day CLV)
Purpose: Load trained model and make predictions on new customer data

Usage:
    # As a module
    from scripts.inference import SpendPredictor
    predictor = SpendPredictor()
    prediction = predictor.predict_single(customer_data)
    predictions = predictor.predict_batch(customer_df)
    
    # From command line
    python scripts/inference.py --customer_id C0001
    python scripts/inference.py --all
================================================================================
"""

import pandas as pd
import numpy as np
import joblib
import os
import argparse
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "final_model.joblib")
FEATURES_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "customer_features.csv")


class SpendPredictor:
    """
    Customer Spend Predictor for 30-day CLV predictions.
    
    Attributes:
        model: Trained regression model
        preprocessor: Fitted ColumnTransformer for feature preprocessing
        feature_cols: List of required feature column names
        model_name: Name of the model (e.g., "Linear Regression")
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the predictor by loading the trained model.
        
        Args:
            model_path: Path to the joblib model file. Uses default if None.
        """
        if model_path is None:
            model_path = MODEL_PATH
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")
        
        # Load model package
        print(f"Loading model from: {model_path}")
        pkg = joblib.load(model_path)
        
        self.model = pkg['model']
        self.preprocessor = pkg['preprocessor']
        self.feature_cols = pkg['feature_cols']
        self.numeric_cols = pkg['numeric_cols']
        self.categorical_cols = pkg['categorical_cols']
        self.model_name = pkg['model_name']
        self.metrics = pkg['metrics']
        self.trained_date = pkg['trained_date']
        
        print(f"✓ Loaded {self.model_name} (trained: {self.trained_date})")
        print(f"  Expected features: {len(self.feature_cols)}")
        print(f"  Test MAE: ${self.metrics['Test_MAE']:.2f}")
    
    def predict_single(self, customer_data):
        """
        Predict 30-day spend for a single customer.
        
        Args:
            customer_data: dict or pd.Series with customer features
            
        Returns:
            float: Predicted spend amount (non-negative)
        """
        if isinstance(customer_data, dict):
            customer_data = pd.Series(customer_data)
        
        # Convert to DataFrame
        df = pd.DataFrame([customer_data])
        
        return self.predict_batch(df)[0]
    
    def predict_batch(self, customer_df):
        """
        Predict 30-day spend for multiple customers.
        
        Args:
            customer_df: pd.DataFrame with customer features
            
        Returns:
            np.array: Predicted spend amounts (non-negative)
        """
        # Validate columns
        missing_cols = set(self.feature_cols) - set(customer_df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Extract features in correct order
        X = customer_df[self.feature_cols].copy()
        
        # Fill missing numeric values
        X[self.numeric_cols] = X[self.numeric_cols].fillna(0)
        
        # Preprocess
        X_processed = self.preprocessor.transform(X)
        
        # Predict (clip to non-negative)
        predictions = np.clip(self.model.predict(X_processed), 0, None)
        
        return predictions
    
    def predict_with_details(self, customer_df):
        """
        Predict with additional details (confidence, category).
        
        Args:
            customer_df: pd.DataFrame with customer features
            
        Returns:
            pd.DataFrame with customer_id, predicted_spend, spend_category
        """
        predictions = self.predict_batch(customer_df)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'customer_id': customer_df['customer_id'] if 'customer_id' in customer_df.columns else range(len(predictions)),
            'predicted_spend_30d': predictions.round(2),
        })
        
        # Add spend category
        results['spend_category'] = pd.cut(
            results['predicted_spend_30d'],
            bins=[-1, 0, 25, 75, 200, float('inf')],
            labels=['Zero', 'Low', 'Medium', 'High', 'VIP']
        )
        
        # Add prediction confidence note
        mae = self.metrics['Test_MAE']
        results['prediction_range_low'] = (results['predicted_spend_30d'] - mae).clip(0)
        results['prediction_range_high'] = results['predicted_spend_30d'] + mae
        
        return results
    
    def get_model_info(self):
        """Return model information as a dictionary."""
        return {
            'model_name': self.model_name,
            'trained_date': self.trained_date,
            'num_features': len(self.feature_cols),
            'test_mae': self.metrics['Test_MAE'],
            'test_rmse': self.metrics['Test_RMSE'],
            'test_r2': self.metrics['Test_R2']
        }


def load_customer_features():
    """Load customer features from processed data."""
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"Customer features not found at: {FEATURES_PATH}")
    return pd.read_csv(FEATURES_PATH)


def main():
    """Command-line interface for making predictions."""
    parser = argparse.ArgumentParser(description='Predict 30-day customer spend')
    parser.add_argument('--customer_id', type=str, help='Customer ID to predict for')
    parser.add_argument('--all', action='store_true', help='Predict for all customers')
    parser.add_argument('--top', type=int, default=10, help='Show top N predictions')
    parser.add_argument('--output', type=str, help='Output CSV path for predictions')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = SpendPredictor()
    
    # Load customer data
    print("\nLoading customer features...")
    customers = load_customer_features()
    print(f"✓ Loaded {len(customers)} customers")
    
    if args.customer_id:
        # Single customer prediction
        customer = customers[customers['customer_id'] == args.customer_id]
        if len(customer) == 0:
            print(f"\n❌ Customer {args.customer_id} not found")
            return
        
        result = predictor.predict_with_details(customer)
        
        print(f"\n{'='*60}")
        print(f"PREDICTION FOR CUSTOMER: {args.customer_id}")
        print(f"{'='*60}")
        print(f"Predicted 30-Day Spend: ${result['predicted_spend_30d'].values[0]:.2f}")
        print(f"Spend Category: {result['spend_category'].values[0]}")
        print(f"Prediction Range: ${result['prediction_range_low'].values[0]:.2f} - ${result['prediction_range_high'].values[0]:.2f}")
        print(f"{'='*60}")
        
    elif args.all or args.output:
        # Batch prediction
        print("\nMaking predictions for all customers...")
        results = predictor.predict_with_details(customers)
        
        if args.output:
            results.to_csv(args.output, index=False)
            print(f"\n✓ Saved predictions to: {args.output}")
        
        # Show summary
        print(f"\n{'='*60}")
        print("PREDICTION SUMMARY")
        print(f"{'='*60}")
        print(f"Total Customers: {len(results)}")
        print(f"Mean Predicted Spend: ${results['predicted_spend_30d'].mean():.2f}")
        print(f"Median Predicted Spend: ${results['predicted_spend_30d'].median():.2f}")
        print(f"\nSpend Category Distribution:")
        print(results['spend_category'].value_counts().to_string())
        
        # Show top predictions
        print(f"\n{'='*60}")
        print(f"TOP {args.top} PREDICTED SPENDERS")
        print(f"{'='*60}")
        top = results.nlargest(args.top, 'predicted_spend_30d')
        for _, row in top.iterrows():
            print(f"  {row['customer_id']}: ${row['predicted_spend_30d']:.2f} ({row['spend_category']})")
    
    else:
        # Default: show sample predictions
        print("\nSample predictions (use --customer_id or --all for more):")
        sample = customers.sample(5, random_state=42)
        results = predictor.predict_with_details(sample)
        
        print(f"\n{'Customer ID':<12} {'Predicted Spend':>15} {'Category':>10}")
        print("-" * 40)
        for _, row in results.iterrows():
            print(f"{row['customer_id']:<12} ${row['predicted_spend_30d']:>13.2f} {row['spend_category']:>10}")


if __name__ == "__main__":
    main()
