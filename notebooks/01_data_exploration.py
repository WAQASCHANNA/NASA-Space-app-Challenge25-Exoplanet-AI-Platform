import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import NASADataLoader
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def explore_dataset(dataset_name="kepler"):
    """
    Explore a specific dataset
    """
    print(f"\n{'='*60}")
    print(f"🚀 Exploring {dataset_name.upper()} Dataset")
    print(f"{'='*60}")
    
    # Initialize data loader
    loader = NASADataLoader()
    
    # Load data based on dataset name
    if dataset_name.lower() == "kepler":
        df = loader.load_kepler_data(use_cache=True)
    elif dataset_name.lower() == "tess":
        df = loader.load_tess_data(use_cache=True)
    elif dataset_name.lower() == "k2":
        if hasattr(loader, "load_k2_data"):
            df = loader.load_k2_data(use_cache=True)
        else:
            print("❌ NASADataLoader has no method 'load_k2_data'")
            return None
    else:
        print(f"❌ Unknown dataset: {dataset_name}")
        return None
    
    if df is not None:
        # Get dataset info
        info = loader.get_dataset_info(df, dataset_name)
        
        print("📊 Dataset Overview:")
        print(f"   Shape: {info['shape']}")
        print(f"   Total columns: {len(info['columns'])}")
        print(f"   Missing values: {info['missing_values']}")
        
        # Show first few rows
        print("\n👀 First 3 rows:")
        print(df.head(3))
        
        # Explore target variable
        target_dist = loader.explore_target_variable(df, dataset_name)
        
        # Check feature availability
        available_features, missing_features = loader.check_feature_availability(df, dataset_name)
        
        # Basic statistics for available features
        if available_features:
            print(f"\n📈 Basic statistics for available features:")
            print(df[available_features].describe())
        
        return df, available_features
    else:
        print(f"❌ Failed to load {dataset_name} data")
        return None, None

def main():
    print("🚀 Starting NASA Exoplanet Data Exploration")

    # Explore Kepler data first
    kepler_df, kepler_features = explore_dataset("kepler")

    # Optionally explore TESS data
    explore_tess = input("\n🤔 Would you like to explore TESS data as well? (y/n): ").lower().strip()
    if explore_tess == 'y':
        tess_df, tess_features = explore_dataset("tess")

    # Optionally explore K2 data
    explore_k2 = input("\n🤔 Would you like to explore K2 data as well? (y/n): ").lower().strip()
    if explore_k2 == 'y':
        k2_df, k2_features = explore_dataset("k2")

    return kepler_df

if __name__ == "__main__":
    df = main()
