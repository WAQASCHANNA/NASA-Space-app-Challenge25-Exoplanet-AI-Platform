"""
Notebook to test NEOSSat data loader and initial data exploration.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_loader_neossat import NEOSSatDataLoader
from processing.data_preprocessor_neossat import NEOSSatDataPreprocessor

def main():
    loader = NEOSSatDataLoader()
    df_raw = loader.load_all_data()
    print(f"Loaded {len(df_raw)} NEOSSat observations")
    print(df_raw.head())

    preprocessor = NEOSSatDataPreprocessor()
    df_processed = preprocessor.preprocess(df_raw)
    df_processed = preprocessor.feature_engineering(df_processed)

    print("\nProcessed Data Sample:")
    print(df_processed.head())

if __name__ == "__main__":
    main()
