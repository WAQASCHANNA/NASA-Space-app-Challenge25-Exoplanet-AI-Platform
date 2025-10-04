# notebooks/02_data_preprocessing_tess.py

import sys
import os
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))

from src.data_loader import NASADataLoader
from src.processing.data_preprocessor import DataPreprocessor
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("🚀 Starting TESS Data Preprocessing")
    print("=" * 60)

    # Load the data
    loader = NASADataLoader()
    tess_df = loader.load_tess_data_from_csv()

    if tess_df is None:
        print("❌ Failed to load TESS data")
        return

    # Get the feature columns from config
    feature_columns = loader.data_sources["tess"]["feature_columns"]

    print("📋 Initial TESS Data Summary:")
    print(f"   Total samples: {len(tess_df)}")
    print(f"   Features to use: {len(feature_columns)}")

    # Check target column
    target_col = loader.data_sources["tess"]["target_column"]
    if target_col in tess_df.columns:
        print(f"   Target distribution:\n{tess_df[target_col].value_counts()}")
    else:
        print(f"   Target column '{target_col}' not found. Available columns: {tess_df.columns.tolist()[:10]}")

    # Initialize preprocessor with TESS target column
    preprocessor = DataPreprocessor(target_column=target_col)

    # Run preprocessing pipeline for TESS
    X, y, final_features = preprocessor.preprocess_pipeline_tess(tess_df, feature_columns)

    # Split the data
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)

    # Save processed data
    processed_data = {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        'feature_names': final_features
    }

    # Create directory if it doesn't exist
    processed_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    # Save using joblib for numpy arrays
    import joblib
    processed_file = os.path.join(processed_dir, 'tess_processed.pkl')
    joblib.dump(processed_data, processed_file)

    print(f"File saved to: {processed_file}")
    print(f"Directory contents: {os.listdir(processed_dir)}")

    print(f"\n💾 Processed TESS data saved to: ../data/processed/tess_processed.pkl")

    # Display feature importance hint
    print(f"\n🔍 Key Features Available:")
    for i, feature in enumerate(final_features[:10], 1):  # Show first 10
        print(f"   {i:2d}. {feature}")

    if len(final_features) > 10:
        print(f"   ... and {len(final_features) - 10} more features")

    return processed_data

if __name__ == "__main__":
    processed_data = main()
