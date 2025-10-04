# src/processing/data_preprocessor.py - UPDATED

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self, target_column='koi_disposition', random_state=42):
        self.target_column = target_column
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_columns = None
        self.is_fitted = False
        
    def prepare_target(self, df):
        """
        Prepare the target variable for classification
        """
        print("🎯 Preparing target variable...")
        
        # Map to binary classification: CONFIRMED vs FALSE POSITIVE
        # We'll exclude CANDIDATE for initial model training
        df_clean = df[df[self.target_column].isin(['CONFIRMED', 'FALSE POSITIVE'])].copy()
        
        # Convert to binary: CONFIRMED = 1, FALSE POSITIVE = 0
        df_clean['target'] = (df_clean[self.target_column] == 'CONFIRMED').astype(int)
        
        print(f"   Original shape: {df.shape}")
        print(f"   After filtering CANDIDATE: {df_clean.shape}")
        print(f"   Target distribution:\n{df_clean['target'].value_counts()}")
        
        return df_clean
    
    def select_numerical_features(self, df, feature_columns):
        """
        Select only numerical features for model training
        """
        print("🔍 Selecting numerical features...")
        
        # Identify numerical columns only
        numerical_features = []
        for feature in feature_columns:
            if feature in df.columns:
                # Check if column is numerical
                if pd.api.types.is_numeric_dtype(df[feature]):
                    numerical_features.append(feature)
                else:
                    print(f"   ⚠️  Skipping non-numeric feature: {feature}")
            else:
                print(f"   ⚠️  Feature not found: {feature}")
        
        print(f"   Selected {len(numerical_features)} numerical features")
        return numerical_features
    
    def handle_missing_values(self, df, feature_columns, threshold=0.3):
        """
        Handle missing values in feature columns
        """
        print("🔍 Handling missing values...")
        
        # Calculate missing value percentage for each feature
        missing_percent = df[feature_columns].isnull().sum() / len(df)
        
        # Remove features with too many missing values
        features_to_keep = missing_percent[missing_percent < threshold].index.tolist()
        features_to_drop = missing_percent[missing_percent >= threshold].index.tolist()
        
        print(f"   Features kept ({len(features_to_keep)}): {features_to_keep}")
        if features_to_drop:
            print(f"   Features dropped ({len(features_to_drop)}): {features_to_drop}")
        
        # Impute remaining missing values
        df_imputed = df.copy()
        if features_to_keep:
            df_imputed[features_to_keep] = self.imputer.fit_transform(df[features_to_keep])
        
        return df_imputed, features_to_keep
    
    def remove_outliers(self, df, feature_columns, n_std=3):
        """
        Remove extreme outliers using standard deviation method
        """
        print("📊 Removing outliers...")
        
        original_size = len(df)
        
        for feature in feature_columns:
            if feature in df.columns and pd.api.types.is_numeric_dtype(df[feature]):
                mean = df[feature].mean()
                std = df[feature].std()
                
                # Keep only values within n_std standard deviations
                df = df[np.abs(df[feature] - mean) <= n_std * std]
        
        print(f"   Removed {original_size - len(df)} outliers")
        print(f"   Remaining samples: {len(df)}")
        
        return df
    
    def create_features(self, df, feature_columns):
        """
        Create additional engineered features
        """
        print("⚙️ Creating engineered features...")
        
        df_engineered = df.copy()
        
        # Feature: Transit depth to duration ratio
        if 'koi_depth' in feature_columns and 'koi_duration' in feature_columns:
            df_engineered['depth_duration_ratio'] = df_engineered['koi_depth'] / df_engineered['koi_duration']
        
        # Feature: Orbital period squared (for non-linear relationships)
        if 'koi_period' in feature_columns:
            df_engineered['period_squared'] = df_engineered['koi_period'] ** 2
        
        # Feature: Stellar density proxy
        if 'koi_slogg' in feature_columns and 'koi_steff' in feature_columns:
            df_engineered['stellar_density_proxy'] = 10 ** df_engineered['koi_slogg'] / (df_engineered['koi_steff'] / 5772) ** 4
        
        # Feature: Signal strength (depth relative to noise)
        if 'koi_depth' in feature_columns and 'koi_model_snr' in feature_columns:
            df_engineered['signal_strength'] = df_engineered['koi_depth'] * df_engineered['koi_model_snr']
        
        new_features = [col for col in df_engineered.columns if col not in df.columns]
        print(f"   Created {len(new_features)} new features: {new_features}")
        
        return df_engineered
    
    def preprocess_pipeline(self, df, feature_columns):
        """
        Complete preprocessing pipeline
        """
        print("🚀 Starting data preprocessing pipeline...")
        print("=" * 50)
        
        # Step 1: Prepare target
        df_processed = self.prepare_target(df)
        
        # Step 1.5: Select only numerical features
        numerical_features = self.select_numerical_features(df_processed, feature_columns)
        
        # Step 2: Handle missing values
        df_processed, kept_features = self.handle_missing_values(df_processed, numerical_features)
        
        # Step 3: Remove outliers
        df_processed = self.remove_outliers(df_processed, kept_features)
        
        # Step 4: Create engineered features
        df_processed = self.create_features(df_processed, kept_features)
        
        # Update feature columns to include engineered features
        all_features = kept_features + [col for col in df_processed.columns 
                                      if col not in kept_features and col not in [self.target_column, 'target']]
        
        # Final check: ensure all features are numerical
        final_numerical_features = []
        for feature in all_features:
            if pd.api.types.is_numeric_dtype(df_processed[feature]):
                final_numerical_features.append(feature)
            else:
                print(f"   ⚠️  Removing non-numeric engineered feature: {feature}")
        
        print(f"   Final numerical features: {len(final_numerical_features)}")
        
        # Step 5: Prepare final feature set
        X = df_processed[final_numerical_features]
        y = df_processed['target']
        
        print(f"\n✅ Preprocessing completed!")
        print(f"   Final feature set: {len(final_numerical_features)} features")
        print(f"   Final dataset shape: {X.shape}")
        print(f"   Target distribution: {y.value_counts().to_dict()}")
        
        self.feature_columns = final_numerical_features
        self.is_fitted = True
        
        return X, y, final_numerical_features
    
    def split_data(self, X, y, test_size=0.2, val_size=0.1):
        """
        Split data into train, validation, and test sets
        """
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Second split: separate train and validation from temp
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=self.random_state, stratify=y_temp
        )
        
        print(f"\n📊 Data Split Summary:")
        print(f"   Training set: {X_train.shape[0]} samples")
        print(f"   Validation set: {X_val.shape[0]} samples") 
        print(f"   Test set: {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
