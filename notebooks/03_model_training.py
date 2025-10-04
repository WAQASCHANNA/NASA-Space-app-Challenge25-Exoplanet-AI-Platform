import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from models.exoplanet_classifier import ExoplanetClassifier
from data_loader import NASADataLoader
from processing.data_preprocessor import DataPreprocessor

def main(dataset="kepler"):
    print(f"🚀 Starting Exoplanet Detection Model Training for {dataset.upper()} data")
    print("=" * 60)
    
    data_loader = NASADataLoader()
    preprocessor = DataPreprocessor()
    
    if dataset == "kepler":
        # Load Kepler data from cache or API
        df = data_loader.load_kepler_data()
        feature_columns = data_loader.data_sources["kepler"]["feature_columns"]
        preprocessor.target_column = data_loader.data_sources["kepler"]["target_column"]
        
        if df is None:
            print("❌ Failed to load Kepler data")
            return None, None, None
        
        # Preprocess data
        X, y, feature_names = preprocessor.preprocess_pipeline(df, feature_columns)
        
    elif dataset == "tess":
        # Load TESS data from local CSV
        tess_csv_path = "TOI_2025.10.03_06.01.46.csv"
        df = data_loader.load_tess_data_from_csv(tess_csv_path)
        feature_columns = data_loader.data_sources["tess"]["feature_columns"]
        preprocessor.target_column = data_loader.data_sources["tess"]["target_column"]

        if df is None:
            print("❌ Failed to load TESS data")
            return None, None, None

        # Preprocess TESS data
        X, y, feature_names = preprocessor.preprocess_pipeline_tess(df, feature_columns)

    elif dataset == "k2":
        # Load K2 data from local CSV
        k2_csv_path = "k2pandc_2025.10.03_08.39.33.csv"
        df = data_loader.load_k2_data_from_csv(k2_csv_path)
        feature_columns = data_loader.data_sources["k2"]["feature_columns"]
        preprocessor.target_column = data_loader.data_sources["k2"]["target_column"]

        if df is None:
            print("❌ Failed to load K2 data")
            return None, None, None

        # Preprocess K2 data
        X, y, feature_names = preprocessor.preprocess_pipeline_k2(df, feature_columns)

    else:
        print(f"❌ Unsupported dataset: {dataset}")
        return None, None, None
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
    
    print("📊 Data split completed:")
    print(f"   Training set: {X_train.shape}")
    print(f"   Validation set: {X_val.shape}")
    print(f"   Test set: {X_test.shape}")
    
    # Initialize and train classifier
    classifier = ExoplanetClassifier(random_state=42)
    classifier.initialize_models()
    performance = classifier.train_models(X_train, y_train, X_val, y_val)
    
    if not classifier.is_trained:
        print("\n❌ No models were trained successfully. Checking data issues...")
        non_numeric_cols = [col for col in X_train.columns if not pd.api.types.is_numeric_dtype(X_train[col])]
        if non_numeric_cols:
            print(f"   Found non-numeric columns: {non_numeric_cols}")
            print("   Please run the preprocessing again with fixed data types.")
        return None, None, None
    
    # Evaluate on test set
    test_metrics = classifier.evaluate_on_test(X_test, y_test)
    
    # Show feature importance
    feature_importance_df = classifier.get_feature_importance(feature_names, top_n=15)
    
    # Save the trained model
    os.makedirs('../models', exist_ok=True)
    model_save_path = f'../models/exoplanet_classifier_{dataset}.pkl'
    classifier.save_model(model_save_path)
    
    # Create performance comparison
    print("\n📈 Model Performance Comparison:")
    print("=" * 50)
    
    performance_df = pd.DataFrame.from_dict(classifier.model_performance, orient='index')
    performance_df = performance_df[['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']]
    performance_df = performance_df.round(4)
    
    print(performance_df.sort_values('f1_score', ascending=False))
    
    # Save performance results
    performance_csv_path = f'../models/model_performance_{dataset}.csv'
    performance_df.to_csv(performance_csv_path)
    print(f"\n💾 Performance results saved to: {performance_csv_path}")
    
    return classifier, test_metrics, feature_importance_df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train exoplanet detection model")
    parser.add_argument('--dataset', type=str, default='kepler', choices=['kepler', 'tess', 'k2'], help='Dataset to use for training')
    args = parser.parse_args()
    
    classifier, test_metrics, feature_importance = main(dataset=args.dataset)
