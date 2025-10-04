# src/visualization/advanced_plots.py

import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class AdvancedVisualizations:
    def __init__(self):
        self.color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    def create_interactive_feature_importance(self, importance_df, top_n=15):
        """Create interactive feature importance plot"""
        fig = px.bar(
            importance_df.head(top_n),
            x='importance',
            y='feature',
            orientation='h',
            title='<b>Feature Importance for Exoplanet Detection</b>',
            color='importance',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            height=500,
            showlegend=False,
            font=dict(size=12)
        )
        
        return fig
    
    def create_confidence_distribution(self, probabilities, predictions, labels):
        """Create confidence distribution plot"""
        fig = go.Figure()
        
        # Exoplanet confidence
        exoplanet_conf = probabilities[predictions == 1, 1]
        false_positive_conf = probabilities[predictions == 0, 0]
        
        fig.add_trace(go.Histogram(
            x=exoplanet_conf,
            nbinsx=20,
            name='Exoplanet Confidence',
            opacity=0.7,
            marker_color='#2ca02c'
        ))
        
        fig.add_trace(go.Histogram(
            x=false_positive_conf,
            nbinsx=20,
            name='False Positive Confidence',
            opacity=0.7,
            marker_color='#d62728'
        ))
        
        fig.update_layout(
            title='<b>Model Confidence Distribution</b>',
            xaxis_title='Prediction Confidence',
            yaxis_title='Count',
            barmode='overlay',
            height=400
        )
        
        return fig
    
    def create_performance_curves(self, y_true, y_proba):
        """Create ROC and Precision-Recall curves"""
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = np.trapz(tpr, fpr)
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = np.trapz(precision, recall)
        
        fig = sp.make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                f'<b>ROC Curve (AUC = {roc_auc:.4f})</b>',
                f'<b>Precision-Recall Curve (AUC = {pr_auc:.4f})</b>'
            )
        )
        
        # ROC Curve
        fig.add_trace(
            go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', 
                      line=dict(dash='dash', color='gray')),
            row=1, col=1
        )
        
        # Precision-Recall Curve
        fig.add_trace(
            go.Scatter(x=recall, y=precision, mode='lines', 
                      name='Precision-Recall', line=dict(color='green')),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text='False Positive Rate', row=1, col=1)
        fig.update_yaxes(title_text='True Positive Rate', row=1, col=1)
        fig.update_xaxes(title_text='Recall', row=1, col=2)
        fig.update_yaxes(title_text='Precision', row=1, col=2)
        
        fig.update_layout(height=400, showlegend=False)
        
        return fig
    
    def create_parameter_space_plot(self, df, feature_x, feature_y, target_col):
        """Create interactive 2D parameter space plot"""
        fig = px.scatter(
            df, 
            x=feature_x,
            y=feature_y,
            color=target_col,
            title=f'<b>Exoplanet Distribution in {feature_x} vs {feature_y} Space</b>',
            color_discrete_map={0: '#d62728', 1: '#2ca02c'},
            labels={target_col: 'Classification'},
            hover_data=df.columns
        )
        
        fig.update_layout(height=500)
        return fig
    
    def create_transit_simulation(self, params):
        """Create simulated transit light curve"""
        time = np.linspace(-0.2, 0.2, 1000)
        
        # Simple transit model
        depth = params.get('koi_depth', 1000) / 1e6  # Convert ppm to fractional depth
        duration = params.get('koi_duration', 3) / 24  # Convert hours to days
        period = params.get('koi_period', 10)
        
        # Create transit signal
        flux = np.ones_like(time)
        transit_center = 0
        half_duration = duration / 2
        
        in_transit = np.abs(time - transit_center) < half_duration
        flux[in_transit] = 1 - depth
        
        # Add some noise
        noise_level = 0.001
        flux += np.random.normal(0, noise_level, len(flux))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=time, 
            y=flux,
            mode='lines',
            name='Simulated Light Curve',
            line=dict(color='#1f77b4')
        ))
        
        fig.update_layout(
            title='<b>Simulated Transit Light Curve</b>',
            xaxis_title='Time (days)',
            yaxis_title='Normalized Flux',
            height=400
        )
        
        return fig
    
    def create_confusion_matrix_plot(self, y_true, y_pred):
        """Create interactive confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        fig = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale='Blues',
            title='<b>Confusion Matrix</b>',
            labels=dict(x="Predicted", y="Actual", color="Count")
        )
        
        fig.update_layout(
            xaxis=dict(tickvals=[0, 1], ticktext=['False Positive', 'Exoplanet']),
            yaxis=dict(tickvals=[0, 1], ticktext=['False Positive', 'Exoplanet']),
            height=400
        )
        
        return fig
    
    def create_feature_correlation_heatmap(self, df, target_col):
        """Create feature correlation heatmap"""
        # Calculate correlations
        corr_matrix = df.corr()
        
        # Focus on correlations with target
        target_correlations = corr_matrix[target_col].sort_values(ascending=False)
        
        fig = px.imshow(
            corr_matrix,
            color_continuous_scale='RdBu_r',
            title='<b>Feature Correlation Matrix</b>',
            aspect="auto"
        )
        
        fig.update_layout(height=600)
        
        return fig, target_correlations
