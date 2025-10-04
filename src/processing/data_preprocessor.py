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
        Prepare the target variable for classification (Kepler default)
        """
        print("🎯 Preparing target variable for Kepler data...")
        
        # Map to binary classification: CONFIRMED vs FALSE POSITIVE
        # We'll exclude CANDIDATE for initial model training
        df_clean = df[df[self.target_column].isin(['CONFIRMED', 'FALSE POSITIVE'])].copy()
        
        # Convert to binary: CONFIRMED = 1, FALSE POSITIVE = 0
        df_clean['target'] = (df_clean[self.target_column] == 'CONFIRMED').astype(int)
        
        print(f"   Original shape: {df.shape}")
        print(f"   After filtering CANDIDATE: {df_clean.shape}")
        print(f"   Target distribution:\n{df_clean['target'].value_counts()}")
        
        return df_clean

    def prepare_target_tess(self, df):
        """
        Prepare the target variable for classification (TESS data)
        """
        print("🎯 Preparing target variable for TESS data...")
        
        # Map to binary classification: CP (Confirmed Planet) vs FP (False Positive)
        # We'll exclude KP (Known Planet) and PC (Planet Candidate) for initial model training
        valid_classes = ['CP', 'FP']
        df_clean = df[df[self.target_column].isin(valid_classes)].copy()
        
        # Convert to binary: CP = 1, FP = 0
        df_clean['target'] = (df_clean[self.target_column] == 'CP').astype(int)
        
        print(f"   Original shape: {df.shape}")
        print(f"   After filtering dispositions: {df_clean.shape}")
        print(f"   Target distribution:\n{df_clean['target'].value_counts()}")
        
        return df_clean

    def prepare_target_k2(self, df):
        """
        Prepare the target variable for classification (K2 data)
        """
        print("🎯 Preparing target variable for K2 data...")
        
        # Map to binary classification: CONFIRMED vs FALSE POSITIVE
        # We'll exclude CANDIDATE and REFUTED for initial model training
        valid_classes = ['CONFIRMED', 'FALSE POSITIVE']
        df_clean = df[df[self.target_column].isin(valid_classes)].copy()
        
        # Convert to binary: CONFIRMED = 1, FALSE POSITIVE = 0
        df_clean['target'] = (df_clean[self.target_column] == 'CONFIRMED').astype(int)
        
        print(f"   Original shape: {df.shape}")
        print(f"   After filtering dispositions: {df_clean.shape}")
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

    def handle_missing_values_tess(self, df, feature_columns):
        """
        Handle missing values specifically for TESS data
        """
        print("🔍 Handling missing values for TESS data...")

        # For TESS, fill missing numerical values with median
        df_imputed = df.copy()
        for col in feature_columns:
            if col in df_imputed.columns and pd.api.types.is_numeric_dtype(df_imputed[col]):
                median_val = df_imputed[col].median()
                df_imputed[col].fillna(median_val, inplace=True)

        return df_imputed

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

    def preprocess_pipeline_tess(self, df, feature_columns):
        """
        Improved preprocessing pipeline for TESS data with better NaN handling
        """
        print("🚀 Starting improved data preprocessing pipeline for TESS data...")
        print("=" * 60)

        # Step 1: Prepare target with better encoding
        df_processed = self.prepare_target_tess_improved(df)

        # Step 2: Select scientifically meaningful features
        scientific_features = self.select_scientific_features_tess(df_processed, feature_columns)

        # Step 3: Advanced missing value handling
        X, y = self.handle_missing_values_advanced(df_processed, scientific_features)

        # Step 4: Remove outliers
        X, y = self.remove_outliers_advanced(X, y)

        # Step 5: Create engineered features (optional, can be extended)
        X = self.create_features_tess(X)

        print(f"\n✅ Preprocessing completed for TESS data!")
        print(f"   Final feature set: {X.shape[1]} features")
        print(f"   Final dataset shape: {X.shape}")
        print(f"   Target distribution: {y.value_counts().to_dict()}")

        self.feature_columns = X.columns.tolist()
        self.is_fitted = True

        return X, y, X.columns.tolist()

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

    def prepare_target_tess_improved(self, df):
        """
        Improved target preparation for TESS data with better class mapping
        """
        print("🎯 Preparing target variable for TESS data (improved)...")

        # Better disposition mapping
        disposition_mapping = {
            'CP': 1,  # Confirmed Planet
            'PC': 1,  # Planet Candidate
            'KP': 1,  # Known Planet
            'FP': 0,  # False Positive
            'FA': 0   # False Alarm
        }

        df_processed = df.copy()
        df_processed['target'] = df_processed[self.target_column].map(disposition_mapping)
        df_processed = df_processed[df_processed['target'].notna()]

        print(f"   Original shape: {df.shape}")
        print(f"   After filtering: {df_processed.shape}")
        print(f"   Target distribution: {df_processed['target'].value_counts().to_dict()}")

        return df_processed

    def select_scientific_features_tess(self, df, feature_columns):
        """
        Select scientifically meaningful features for TESS data
        """
        print("🔍 Selecting scientifically meaningful features for TESS...")

        # Prioritize scientifically meaningful features
        scientific_features = [
            # Planet properties
            'pl_orbper', 'pl_trandurh', 'pl_trandep', 'pl_rade',
            'pl_insol', 'pl_eqt',
            # Stellar properties
            'st_tmag', 'st_dist', 'st_teff', 'st_logg', 'st_rad',
            # Measurement quality indicators (but not error columns as primary)
            'pl_trandeperr1', 'pl_radeerr1', 'pl_orbpererr1'
        ]

        # Only use available features
        available_features = [f for f in scientific_features if f in df.columns]
        print(f"   Using {len(available_features)} scientific features: {available_features}")

        return available_features

    def handle_missing_values_advanced(self, df, feature_columns):
        """
        Advanced missing value handling for TESS data
        """
        print("🔧 Advanced missing value handling...")

        # Separate features by type for different imputation strategies
        primary_features = ['pl_orbper', 'pl_trandurh', 'pl_trandep', 'pl_rade', 'st_tmag', 'st_teff']
        secondary_features = [f for f in feature_columns if f not in primary_features]

        # For primary features, use median imputation
        primary_imputer = SimpleImputer(strategy='median')

        # For secondary features, use median imputation as well
        secondary_imputer = SimpleImputer(strategy='median')

        # Apply imputation
        X_primary = primary_imputer.fit_transform(df[primary_features])
        X_primary = pd.DataFrame(X_primary, columns=primary_features, index=df.index)

        X_secondary = secondary_imputer.fit_transform(df[secondary_features])
        X_secondary = pd.DataFrame(X_secondary, columns=secondary_features, index=df.index)

        # Combine imputed features
        X_imputed = pd.concat([X_primary, X_secondary], axis=1)
        y = df['target']

        # Remove any remaining NaN values
        mask = ~X_imputed.isna().any(axis=1)
        X_clean = X_imputed[mask]
        y_clean = y[mask]

        print(f"   Final dataset: {X_clean.shape[0]} samples, {X_clean.shape[1]} features")
        print(f"   Missing values removed: {len(X_imputed) - len(X_clean)}")

        return X_clean, y_clean

    def remove_outliers_advanced(self, X, y, n_std=3):
        """
        Advanced outlier removal using IQR method
        """
        print("📊 Removing outliers using IQR method...")

        original_size = len(X)

        # Use IQR method for outlier detection
        for feature in X.columns:
            if pd.api.types.is_numeric_dtype(X[feature]):
                Q1 = X[feature].quantile(0.25)
                Q3 = X[feature].quantile(0.75)
                IQR = Q3 - Q1

                # Define bounds
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Keep only non-outlier values
                mask = (X[feature] >= lower_bound) & (X[feature] <= upper_bound)
                X = X[mask]
                y = y[mask]

        print(f"   Removed {original_size - len(X)} outliers")
        print(f"   Remaining samples: {len(X)}")

        return X, y

    def create_features_tess(self, X):
        """
        Create engineered features for TESS data
        """
        print("⚙️ Creating engineered features for TESS...")

        X_engineered = X.copy()

        # Feature: Transit depth to duration ratio (if available)
        if 'pl_trandep' in X_engineered.columns and 'pl_trandurh' in X_engineered.columns:
            X_engineered['depth_duration_ratio'] = X_engineered['pl_trandep'] / X_engineered['pl_trandurh']

        # Feature: Orbital period squared (for non-linear relationships)
        if 'pl_orbper' in X_engineered.columns:
            X_engineered['period_squared'] = X_engineered['pl_orbper'] ** 2

        # Feature: Stellar density proxy
        if 'st_logg' in X_engineered.columns and 'st_teff' in X_engineered.columns:
            X_engineered['stellar_density_proxy'] = 10 ** X_engineered['st_logg'] / (X_engineered['st_teff'] / 5772) ** 4

        new_features = [col for col in X_engineered.columns if col not in X.columns]
        print(f"   Created {len(new_features)} new features: {new_features}")

        return X_engineered

    def preprocess_pipeline_k2(self, df, feature_columns):
        """
        Complete preprocessing pipeline for K2 data
        """
        print("🚀 Starting data preprocessing pipeline for K2 data...")
        print("=" * 50)

        # Step 1: Prepare target
        df_processed = self.prepare_target_k2(df)

        # Step 1.5: Select only numerical features
        numerical_features = self.select_numerical_features(df_processed, feature_columns)

        # Step 2: Handle missing values
        df_processed = self.handle_missing_values_tess(df_processed, numerical_features)  # Reuse TESS missing value handler

        # Step 3: Remove outliers
        df_processed = self.remove_outliers(df_processed, numerical_features)

        # Step 4: Create engineered features (optional, can be extended)
        df_processed = self.create_features(df_processed, numerical_features)

        # Update feature columns to include engineered features
        all_features = numerical_features + [col for col in df_processed.columns
                                      if col not in numerical_features and col not in [self.target_column, 'target']]

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

        print(f"\n✅ Preprocessing completed for K2 data!")
        print(f"   Final feature set: {len(final_numerical_features)} features")
        print(f"   Final dataset shape: {X.shape}")
        print(f"   Target distribution: {y.value_counts().to_dict()}")

        self.feature_columns = final_numerical_features
        self.is_fitted = True

        return X, y, final_numerical_features
