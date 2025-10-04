# notebooks/02_data_preprocessing.py

import sys
import os
sys.path.append(os.path.abspath('..'))

from src.data_loader import NASADataLoader
from src.processing.data_preprocessor import DataPreprocessor
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("🚀 Starting NASA Exoplanet Data Preprocessing")
    print("=" * 60)
    
    # Load the data
    loader = NASADataLoader()
    kepler_df = loader.load_kepler_data(use_cache=True)
    
    if kepler_df is None:
        print("❌ Failed to load data")
        return
    
    # Get the feature columns from config
    feature_columns = loader.data_sources["kepler"]["feature_columns"]
    
    print("📋 Initial Data Summary:")
    print(f"   Total samples: {len(kepler_df)}")
    print(f"   Features to use: {len(feature_columns)}")
    print(f"   Target distribution:\n{kepler_df['koi_disposition'].value_counts()}")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Run preprocessing pipeline
    X, y, final_features = preprocessor.preprocess_pipeline(kepler_df, feature_columns)
    
    # Split the data
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
    
    # Save processed data
    processed_data = {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        'feature_names': final_features
    }
    
    # Create directory if it doesn't exist
    os.makedirs('../data/processed', exist_ok=True)
    
    # Save using joblib for numpy arrays
    import joblib
    joblib.dump(processed_data, '../data/processed/kepler_processed.pkl')
    
    print(f"\n💾 Processed data saved to: ../data/processed/kepler_processed.pkl")
    
    # Display feature importance hint
    print(f"\n🔍 Key Features Available:")
    for i, feature in enumerate(final_features[:10], 1):  # Show first 10
        print(f"   {i:2d}. {feature}")
    
    if len(final_features) > 10:
        print(f"   ... and {len(final_features) - 10} more features")
    
    return processed_data

if __name__ == "__main__":
    processed_data = main()
