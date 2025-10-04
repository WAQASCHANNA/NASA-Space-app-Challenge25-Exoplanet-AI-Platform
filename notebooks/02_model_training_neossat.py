import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_loader_neossat import NEOSSatDataLoader
from processing.data_preprocessor_neossat import NEOSSatDataPreprocessor
from models.neossat_model_trainer import NEOSSatModelTrainer

def main():
    # Load and preprocess data
    loader = NEOSSatDataLoader()
    df_raw = loader.load_all_data()
    print(f"Loaded {len(df_raw)} NEOSSat observations")

    preprocessor = NEOSSatDataPreprocessor()
    df_processed = preprocessor.preprocess(df_raw)
    df_processed = preprocessor.feature_engineering(df_processed)

    print("\nProcessed Data Sample:")
    print(df_processed.head())

    # Prepare features and target
    # For demonstration, let's assume 'Image.PrimeObjectName' as target (classification)
    if 'Image.PrimeObjectName' not in df_processed.columns:
        print("Target column 'Image.PrimeObjectName' not found in data.")
        return

    X = df_processed.drop(columns=['Image.PrimeObjectName'])
    y = df_processed['Image.PrimeObjectName']

    # Encode categorical features if any
    X_encoded = pd.get_dummies(X, drop_first=True)

    # Train model
    trainer = NEOSSatModelTrainer()
    model = trainer.train(X_encoded, y)

if __name__ == "__main__":
    main()
