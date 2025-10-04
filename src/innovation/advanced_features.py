import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.impute import SimpleImputer
import plotly.express as px
import plotly.graph_objects as go

class AdvancedFeatures:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names

    def calculate_discovery_confidence(self, X, predictions, probabilities):
        """
        Calculate enhanced discovery confidence score
        Combines model probability with novelty and consistency metrics
        """
        # Model confidence
        model_confidence = np.max(probabilities, axis=1)

        # Novelty score (how different from training data)
        novelty_score = self._calculate_novelty_score(X)

        # Feature consistency score
        consistency_score = self._calculate_consistency_score(X)

        # Combined confidence score
        discovery_confidence = (
            0.6 * model_confidence +  # Model prediction weight
            0.2 * (1 - novelty_score) +  # Novelty (inverse - more novel = less confident)
            0.2 * consistency_score    # Internal consistency
        )

        return discovery_confidence, {
            'model_confidence': model_confidence,
            'novelty_score': novelty_score,
            'consistency_score': consistency_score
        }

    def _calculate_novelty_score(self, X):
        """Calculate how novel each sample is compared to training distribution"""
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        novelty = np.linalg.norm(X_scaled, axis=1)
        novelty = novelty / np.max(novelty)

        return novelty

    def _calculate_consistency_score(self, X):
        """Calculate internal consistency of features"""
        scores = []

        for i in range(len(X)):
            score = 1.0

            if 'koi_period' in self.feature_names and 'koi_duration' in self.feature_names:
                period_idx = self.feature_names.index('koi_period')
                duration_idx = self.feature_names.index('koi_duration')

                period = X.iloc[i, period_idx] if hasattr(X, 'iloc') else X[i, period_idx]
                duration = X.iloc[i, duration_idx] if hasattr(X, 'iloc') else X[i, duration_idx]

                if period > 0 and duration > 0:
                    ratio = duration / (period * 24)
                    if ratio > 0.5:
                        score *= 0.5

            if 'koi_prad' in self.feature_names and 'koi_depth' in self.feature_names:
                radius_idx = self.feature_names.index('koi_prad')
                depth_idx = self.feature_names.index('koi_depth')

                radius = X.iloc[i, radius_idx] if hasattr(X, 'iloc') else X[i, radius_idx]
                depth = X.iloc[i, depth_idx] if hasattr(X, 'iloc') else X[i, depth_idx]

                if radius > 0 and depth > 0:
                    expected_depth = (radius / 10) ** 2 * 1e6
                    depth_ratio = min(depth / expected_depth, expected_depth / depth)
                    score *= 0.5 + 0.5 * depth_ratio

            scores.append(score)

        return np.array(scores)

    def detect_anomalies(self, X, contamination=0.1):
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        anomaly_labels = iso_forest.fit_predict(X)

        anomaly_scores = iso_forest.decision_function(X)
        anomaly_labels = (anomaly_labels == -1).astype(int)

        return anomaly_labels, anomaly_scores

    def find_rare_exoplanet_types(self, X, predictions, probabilities, n_clusters=5):
        exoplanet_mask = predictions == 1
        X_exoplanets = X[exoplanet_mask]

        if len(X_exoplanets) < n_clusters:
            return np.zeros(len(X)), {}

        # Handle NaN values by imputing with mean
        imputer = SimpleImputer(strategy='mean')
        X_exoplanets_imputed = imputer.fit_transform(X_exoplanets)

        clustering = DBSCAN(eps=0.5, min_samples=3)
        cluster_labels = clustering.fit_predict(X_exoplanets_imputed)

        full_cluster_labels = np.zeros(len(X)) - 1
        full_cluster_labels[exoplanet_mask] = cluster_labels

        cluster_info = {}
        for cluster_id in np.unique(cluster_labels):
            if cluster_id == -1:
                continue

            cluster_mask = cluster_labels == cluster_id
            cluster_size = np.sum(cluster_mask)

            if cluster_size > 0:
                cluster_data = X_exoplanets_imputed[cluster_mask]
                cluster_info[cluster_id] = {
                    'size': cluster_size,
                    'features_mean': cluster_data.mean(axis=0),
                    'features_std': cluster_data.std(axis=0)
                }

        return full_cluster_labels, cluster_info

    def calculate_confidence_metrics(self, X, y_pred, y_proba, y_test):
        """Calculate various confidence metrics"""
        from sklearn.metrics import brier_score_loss, log_loss

        confidences = np.max(y_proba, axis=1)

        metrics = {
            'mean_confidence': float(confidences.mean()),
            'median_confidence': float(np.median(confidences)),
            'confidence_std': float(confidences.std()),
            'high_confidence_ratio': float((confidences > 0.8).mean()),
            'low_confidence_ratio': float((confidences < 0.6).mean()),
            'brier_score': float(brier_score_loss(y_test, y_proba[:, 1])),
            'log_loss': float(log_loss(y_test, y_proba))
        }

        return metrics

    def create_confidence_calibration_plot(self, y_proba, y_test):
        """Create confidence calibration plot"""
        from sklearn.calibration import calibration_curve

        prob_true, prob_pred = calibration_curve(y_test, y_proba[:, 1], n_bins=10)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=prob_pred,
            y=prob_true,
            mode='lines+markers',
            name='Calibration curve',
            line=dict(color='blue', width=2)
        ))

        # Perfect calibration line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Perfect calibration',
            line=dict(color='red', dash='dash')
        ))

        fig.update_layout(
            title='Confidence Calibration Plot',
            xaxis_title='Predicted Probability',
            yaxis_title='True Probability',
            showlegend=True
        )

        return fig

    def analyze_feature_contributions(self, model, X, feature_names):
        """Analyze feature contributions to predictions"""
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance_df
        return None

    def create_innovation_dashboard(self, X, predictions, probabilities, feature_names):
        discovery_confidence, component_scores = self.calculate_discovery_confidence(
            X, predictions, probabilities
        )

        anomaly_labels, anomaly_scores = self.detect_anomalies(X)
        cluster_labels, cluster_info = self.find_rare_exoplanet_types(
            X, predictions, probabilities
        )

        results_df = pd.DataFrame({
            'prediction': predictions,
            'probability': probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities,
            'discovery_confidence': discovery_confidence,
            'model_confidence': component_scores['model_confidence'],
            'novelty_score': component_scores['novelty_score'],
            'consistency_score': component_scores['consistency_score'],
            'anomaly_score': anomaly_scores,
            'is_anomaly': anomaly_labels,
            'cluster_id': cluster_labels
        })

        for i, feature in enumerate(['koi_period', 'koi_depth', 'koi_prad']):
            if feature in self.feature_names:
                feature_idx = self.feature_names.index(feature)
                results_df[feature] = X[:, feature_idx] if hasattr(X, 'shape') else X[feature]

        return results_df, cluster_info
