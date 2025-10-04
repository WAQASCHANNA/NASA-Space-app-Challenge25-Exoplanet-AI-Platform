# src/data_loader.py - UPDATED WITH CORRECT API CALLS

import pandas as pd
import requests
import os
from tqdm import tqdm
import time
from config import DATA_SOURCES, PATHS

class NASADataLoader:
    def __init__(self):
        self.data_sources = DATA_SOURCES
        self.raw_data_path = PATHS["raw_data"]

        # Create directories if they don't exist
        os.makedirs(self.raw_data_path, exist_ok=True)

    def load_kepler_data(self, use_cache=True):
        """
        Load Kepler Objects of Interest dataset using correct API
        """
        cache_file = os.path.join(self.raw_data_path, "kepler_data.csv")

        # Use cached data if available
        if use_cache and os.path.exists(cache_file):
            print("📁 Loading cached Kepler data...")
            return pd.read_csv(cache_file)

        print("🌍 Downloading Kepler data from NASA API...")

        try:
            # Use the correct API endpoint
            kepler_url = self.data_sources["kepler"]["url"]

            # Add a user agent to avoid being blocked
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            response = requests.get(kepler_url, headers=headers, timeout=30)
            response.raise_for_status()

            # The API returns clean CSV, no need to remove comments
            df = pd.read_csv(pd.io.common.StringIO(response.text))

            # Cache the data
            df.to_csv(cache_file, index=False)
            print(f"✅ Kepler data loaded successfully! Shape: {df.shape}")

            return df

        except requests.exceptions.RequestException as e:
            print(f"❌ Network error loading Kepler data: {e}")
            return None
        except Exception as e:
            print(f"❌ Error processing Kepler data: {e}")
            return None

    def load_tess_data_from_csv(self, csv_path=None):
        """
        Load TESS data from local CSV file
        """
        if csv_path is None:
            # Default path to the downloaded TESS file
            csv_path = "TOI_2025.10.03_06.01.46.csv"

        print(f"📁 Loading TESS data from local CSV: {csv_path}")

        try:
            # Read the CSV file, skipping header rows that start with #
            df = pd.read_csv(csv_path, comment='#')

            print(f"✅ TESS data loaded successfully! Shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")

            return df

        except FileNotFoundError:
            print(f"❌ TESS file not found at: {csv_path}")
            print("Please update the file path in the code.")
            return None
        except Exception as e:
            print(f"❌ Error loading TESS data: {e}")
            return None

    def load_k2_data_from_csv(self, csv_path=None):
        """
        Load K2 data from local CSV file
        """
        if csv_path is None:
            # Default path to the downloaded K2 file
            csv_path = "k2pandc_2025.10.03_08.39.33.csv"

        print(f"📁 Loading K2 data from local CSV: {csv_path}")

        try:
            # Read the CSV file, skipping header rows that start with #
            df = pd.read_csv(csv_path, comment='#')

            print(f"✅ K2 data loaded successfully! Shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()[:10]}...")  # Show first 10 columns

            return df

        except FileNotFoundError:
            print(f"❌ K2 file not found at: {csv_path}")
            print("Please update the file path in the code.")
            return None
        except Exception as e:
            print(f"❌ Error loading K2 data: {e}")
            return None

    def load_tess_data(self, use_cache=True):
        """
        Load TESS Objects of Interest dataset (legacy method using API)
        """
        cache_file = os.path.join(self.raw_data_path, "tess_data.csv")

        if use_cache and os.path.exists(cache_file):
            print("📁 Loading cached TESS data...")
            return pd.read_csv(cache_file)

        print("🌍 Downloading TESS data from NASA API...")

        try:
            tess_url = self.data_sources["tess"]["url"]
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            response = requests.get(tess_url, headers=headers, timeout=30)
            response.raise_for_status()

            df = pd.read_csv(pd.io.common.StringIO(response.text))
            df.to_csv(cache_file, index=False)
            print(f"✅ TESS data loaded successfully! Shape: {df.shape}")

            return df

        except Exception as e:
            print(f"❌ Error loading TESS data: {e}")
            return None

    def load_k2_data(self, use_cache=True):
        """
        Load K2 Objects of Interest dataset
        Note: K2 API is currently not working due to invalid table name
        """
        if "k2" not in self.data_sources:
            print("⚠️  K2 data source not configured (API table name issues)")
            return None

        cache_file = os.path.join(self.raw_data_path, "k2_data.csv")

        if use_cache and os.path.exists(cache_file):
            print("📁 Loading cached K2 data...")
            return pd.read_csv(cache_file)

        print("🌍 Downloading K2 data from NASA API...")

        try:
            k2_url = self.data_sources["k2"]["url"]
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            response = requests.get(k2_url, headers=headers, timeout=30)
            response.raise_for_status()

            df = pd.read_csv(pd.io.common.StringIO(response.text))
            df.to_csv(cache_file, index=False)
            print(f"✅ K2 data loaded successfully! Shape: {df.shape}")

            return df

        except Exception as e:
            print(f"❌ Error loading K2 data: {e}")
            return None

    def get_dataset_info(self, df, dataset_name="Kepler"):
        """
        Get basic information about the dataset
        """
        if df is None:
            return f"No {dataset_name} data available"

        info = {
            "dataset": dataset_name,
            "shape": df.shape,
            "columns": list(df.columns),
            "missing_values": df.isnull().sum().sum(),
            "target_distribution": None
        }

        # Check if target column exists
        target_col = self.data_sources[dataset_name.lower()]["target_column"]
        if target_col in df.columns:
            info["target_distribution"] = df[target_col].value_counts().to_dict()

        return info

    def explore_target_variable(self, df, dataset_name="kepler"):
        """
        Explore the target variable distribution
        """
        target_col = self.data_sources[dataset_name]["target_column"]

        if target_col not in df.columns:
            print(f"⚠️  Target column '{target_col}' not found in {dataset_name} data")
            print(f"Available columns: {list(df.columns)[:10]}...")  # Show first 10 columns
            return None

        print(f"🎯 {dataset_name.upper()} Target Variable Distribution:")
        target_counts = df[target_col].value_counts()
        print(target_counts)

        return target_counts

    def check_feature_availability(self, df, dataset_name="kepler"):
        """
        Check which features are available in the dataset
        """
        feature_cols = self.data_sources[dataset_name]["feature_columns"]
        available_features = [col for col in feature_cols if col in df.columns]
        missing_features = [col for col in feature_cols if col not in df.columns]

        print(f"\n🔍 Feature Availability for {dataset_name.upper()}:")
        print(f"   Available: {len(available_features)}/{len(feature_cols)}")
        print("   Available features:", available_features)
        if missing_features:
            print("   Missing features:", missing_features)

        return available_features, missing_features
