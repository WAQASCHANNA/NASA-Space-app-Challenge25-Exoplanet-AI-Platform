# src/models/exoplanet_classifier.py

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

class ExoplanetClassifier:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.feature_importance = None
        self.is_trained = False
        
    def initialize_models(self):
        """
        Initialize multiple models for comparison
        """
        print("🤖 Initializing ML models...")
        
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                class_weight='balanced'
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=self.random_state
            ),
            'logistic_regression': LogisticRegression(
                random_state=self.random_state,
                class_weight='balanced',
                max_iter=1000
            ),
            'svm': SVC(
                random_state=self.random_state,
                class_weight='balanced',
                probability=True
            ),
            'xgboost': xgb.XGBClassifier(
                random_state=self.random_state,
                eval_metric='logloss',
                use_label_encoder=False
            )
        }
        
        print(f"   Initialized {len(self.models)} models")
        return self.models
    
    def train_models(self, X_train, y_train, X_val, y_val):
        """
        Train all models and evaluate on validation set
        """
        print("🚀 Training models...")
        print("=" * 50)
        
        model_performance = {}
        
        for name, model in self.models.items():
            print(f"🏃 Training {name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Predict on validation set
                y_pred = model.predict(X_val)
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                
                # Calculate metrics
                accuracy = accuracy_score(y_val, y_pred)
                precision = precision_score(y_val, y_pred)
                recall = recall_score(y_val, y_pred)
                f1 = f1_score(y_val, y_pred)
                auc_roc = roc_auc_score(y_val, y_pred_proba)
                
                model_performance[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'auc_roc': auc_roc
                }
                
                print(f"   ✅ {name}: Accuracy={accuracy:.4f}, F1={f1:.4f}, AUC={auc_roc:.4f}")
                
            except Exception as e:
                print(f"   ❌ Error training {name}: {e}")
                continue
        
        self.model_performance = model_performance
        self.select_best_model()
        self.is_trained = True
        
        return model_performance
    
    def select_best_model(self):
        """
        Select the best model based on F1 score
        """
        if not self.model_performance:
            print("❌ No models trained yet")
            return None
        
        # Find model with highest F1 score
        best_model_name = max(self.model_performance.keys(), 
                             key=lambda x: self.model_performance[x]['f1_score'])
        
        self.best_model = self.model_performance[best_model_name]['model']
        best_metrics = self.model_performance[best_model_name]
        
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"   F1 Score: {best_metrics['f1_score']:.4f}")
        print(f"   Accuracy: {best_metrics['accuracy']:.4f}")
        print(f"   AUC-ROC: {best_metrics['auc_roc']:.4f}")
        
        # Store feature importance if available
        if hasattr(self.best_model, 'feature_importances_'):
            self.feature_importance = self.best_model.feature_importances_
        
        return best_model_name
    
    def evaluate_on_test(self, X_test, y_test):
        """
        Evaluate the best model on test set
        """
        if self.best_model is None:
            print("❌ No best model selected")
            return None
        
        print("\n🧪 Evaluating on Test Set...")
        print("=" * 40)
        
        # Predictions
        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"📊 Test Set Performance:")
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1 Score:  {f1:.4f}")
        print(f"   AUC-ROC:   {auc_roc:.4f}")
        
        print(f"\n📋 Confusion Matrix:")
        print(f"   True Negatives:  {cm[0, 0]}")
        print(f"   False Positives: {cm[0, 1]}")
        print(f"   False Negatives: {cm[1, 0]}")
        print(f"   True Positives:  {cm[1, 1]}")
        
        test_metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'confusion_matrix': cm
        }
        
        return test_metrics
    
    def get_feature_importance(self, feature_names, top_n=15):
        """
        Get and display feature importance
        """
        if self.feature_importance is None:
            print("❌ No feature importance available")
            return None
        
        # Create feature importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔍 Top {top_n} Most Important Features:")
        for i, row in importance_df.head(top_n).iterrows():
            print(f"   {row['feature']}: {row['importance']:.4f}")
        
        return importance_df
    
    def predict_new_data(self, X_new):
        """
        Predict on new data
        """
        if self.best_model is None:
            print("❌ No model trained yet")
            return None
        
        predictions = self.best_model.predict(X_new)
        probabilities = self.best_model.predict_proba(X_new)
        
        return predictions, probabilities
    
    def save_model(self, filepath):
        """
        Save the trained model
        """
        if self.best_model is None:
            print("❌ No model to save")
            return False
        
        joblib.dump({
            'model': self.best_model,
            'feature_importance': self.feature_importance,
            'model_performance': self.model_performance
        }, filepath)
        
        print(f"💾 Model saved to: {filepath}")
        return True
    
    def load_model(self, filepath):
        """
        Load a trained model
        """
        try:
            loaded_data = joblib.load(filepath)
            self.best_model = loaded_data['model']
            self.feature_importance = loaded_data['feature_importance']
            self.model_performance = loaded_data['model_performance']
            self.is_trained = True
            
            print(f"📂 Model loaded from: {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
