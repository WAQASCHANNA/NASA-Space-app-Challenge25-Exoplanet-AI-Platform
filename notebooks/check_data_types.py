# notebooks/check_data_types.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import NASADataLoader
import pandas as pd

def main():
    print("🔍 Checking Data Types...")
    
    # Load data
    loader = NASADataLoader()
    kepler_df = loader.load_kepler_data(use_cache=True)
    
    print(f"Dataset shape: {kepler_df.shape}")
    print(f"Columns: {len(kepler_df.columns)}")
    
    # Check data types
    print("\n📊 Data Types:")
    print(kepler_df.dtypes.value_counts())
    
    # Show non-numeric columns
    non_numeric = []
    for col in kepler_df.columns:
        if not pd.api.types.is_numeric_dtype(kepler_df[col]):
            non_numeric.append(col)
    
    print(f"\n🚫 Non-numeric columns ({len(non_numeric)}):")
    for col in non_numeric:
        print(f"   {col}: {kepler_df[col].dtype}")
        print(f"      Sample values: {kepler_df[col].head(3).tolist()}")
    
    # Show our feature columns
    feature_cols = loader.data_sources["kepler"]["feature_columns"]
    print(f"\n🎯 Our feature columns ({len(feature_cols)}):")
    for col in feature_cols:
        if col in kepler_df.columns:
            dtype = kepler_df[col].dtype
            status = "✅ NUMERIC" if pd.api.types.is_numeric_dtype(kepler_df[col]) else "❌ NON-NUMERIC"
            print(f"   {col}: {dtype} - {status}")
        else:
            print(f"   {col}: ❌ NOT FOUND")

if __name__ == "__main__":
    main()
