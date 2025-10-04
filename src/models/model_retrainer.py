import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class ModelRetrainer:
    def __init__(self, model_path, processed_data_path):
        self.model_path = model_path
        self.processed_data_path = processed_data_path
        self.retraining_history = []

    def load_current_model(self):
        """Load the current trained model"""
        try:
            model_data = joblib.load(self.model_path)
            return model_data['model'], model_data.get('feature_importance', None)
        except Exception as e:
            print(f"Error loading model: {e}")
            return None, None

    def prepare_new_data(self, new_data_df, target_column='target'):
        """
        Prepare new data for retraining
        """
        # Validate required columns
        required_features = joblib.load(self.processed_data_path)['feature_names']

        missing_features = [f for f in required_features if f not in new_data_df.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        if target_column not in new_data_df.columns:
            raise ValueError(f"Target column '{target_column}' not found")

        X_new = new_data_df[required_features]
        y_new = new_data_df[target_column]

        return X_new, y_new

    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model performance"""
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, y_proba)
        }

        return metrics

    def retrain_model(self, new_data_df, model_type='random_forest', test_size=0.2,
                     hyperparameters=None, save_model=True):
        """
        Retrain model with new data
        """
        print("🔄 Starting model retraining...")

        # Load current data
        try:
            processed_data = joblib.load(self.processed_data_path)
            X_old = pd.concat([processed_data['X_train'], processed_data['X_val']])
            y_old = pd.concat([processed_data['y_train'], processed_data['y_val']])
        except Exception as e:
            print(f"Error loading existing data: {e}")
            X_old, y_old = None, None

        # Prepare new data
        X_new, y_new = self.prepare_new_data(new_data_df)

        # Combine old and new data
        if X_old is not None:
            X_combined = pd.concat([X_old, X_new], ignore_index=True)
            y_combined = pd.concat([y_old, y_new], ignore_index=True)
            print(f"   Combined data: {len(X_old)} old + {len(X_new)} new = {len(X_combined)} total")
        else:
            X_combined, y_combined = X_new, y_new
            print(f"   Using only new data: {len(X_combined)} samples")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y_combined, test_size=test_size, random_state=42, stratify=y_combined
        )

        # Initialize model
        if model_type == 'random_forest':
            if hyperparameters:
                model = RandomForestClassifier(**hyperparameters, random_state=42)
            else:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'xgboost':
            if hyperparameters:
                model = xgb.XGBClassifier(**hyperparameters, random_state=42)
            else:
                model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Train model
        print(f"   Training {model_type} with {len(X_train)} samples...")
        model.fit(X_train, y_train)

        # Evaluate
        test_metrics = self.evaluate_model(model, X_test, y_test)

        # Compare with old model if available
        old_model, _ = self.load_current_model()
        if old_model is not None:
            old_metrics = self.evaluate_model(old_model, X_test, y_test)
            improvement = {k: test_metrics[k] - old_metrics[k] for k in test_metrics}
        else:
            old_metrics = None
            improvement = None

        # Save retraining record
        retrain_record = {
            'timestamp': pd.Timestamp.now(),
            'model_type': model_type,
            'training_samples': len(X_train),
            'test_metrics': test_metrics,
            'old_metrics': old_metrics,
            'improvement': improvement,
            'hyperparameters': hyperparameters
        }
        self.retraining_history.append(retrain_record)

        # Save model if improvement is good
        if save_model:
            if improvement and all(v > -0.05 for v in improvement.values()):  # Allow slight degradation
                self.save_model(model, test_metrics)
                print("✅ New model saved successfully!")
            else:
                print("⚠️  New model not saved due to performance degradation")

        return model, test_metrics, improvement

    def save_model(self, model, metrics):
        """Save the retrained model"""
        model_data = {
            'model': model,
            'feature_importance': getattr(model, 'feature_importances_', None),
            'training_metrics': metrics,
            'retraining_history': self.retraining_history
        }

        joblib.dump(model_data, self.model_path)

        # Also save a backup
        backup_path = self.model_path.replace('.pkl', '_backup.pkl')
        joblib.dump(model_data, backup_path)

    def get_retraining_history(self):
        """Get retraining history as DataFrame"""
        if not self.retraining_history:
            return pd.DataFrame()

        return pd.DataFrame(self.retraining_history)

    def create_retraining_template(self):
        """Create a template for new training data"""
        processed_data = joblib.load(self.processed_data_path)
        feature_names = processed_data['feature_names']

        template_df = pd.DataFrame(columns=feature_names + ['target'])

        # Add sample row with explanations
        sample_data = {feature: f"Enter {feature} value" for feature in feature_names}
        sample_data['target'] = "Enter 1 for exoplanet, 0 for false positive"

        template_df = pd.concat([template_df, pd.DataFrame([sample_data])], ignore_index=True)

        return template_df
