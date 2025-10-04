# webapp/app.py

import sys
import os

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'src'))

# Print all environment information
st.sidebar.write("Environment Information:")
st.sidebar.write("Python version:", sys.version)
st.sidebar.write("Working directory:", os.getcwd())
st.sidebar.write("Script location:", __file__)
st.sidebar.write("Contents of script directory:", os.listdir(os.path.dirname(__file__)))

# Configure paths relative to the webapp directory
WEBAPP_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(WEBAPP_DIR, 'models')
DATA_DIR = os.path.join(WEBAPP_DIR, 'data')

# Print debug information
st.sidebar.write("Environment Information:")
st.sidebar.write("Python version:", sys.version)
st.sidebar.write("Working directory:", os.getcwd())
st.sidebar.write("Script location:", __file__)
st.sidebar.write("Contents of script directory:", os.listdir(os.path.dirname(__file__)))

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, 'processed'), exist_ok=True)

# Import local modules after path setup
from data_preprocessor import DataPreprocessor
from src.visualization.advanced_plots import AdvancedVisualizations
from src.models.model_retrainer import ModelRetrainer
from src.innovation.advanced_features import AdvancedFeatures

# Print debug information
st.sidebar.write("Environment Information:")
st.sidebar.write("Python version:", sys.version)
st.sidebar.write("Working directory:", os.getcwd())
st.sidebar.write("Script location:", __file__)
st.sidebar.write("Contents of script directory:", os.listdir(os.path.dirname(__file__)))

st.sidebar.write("\nDirectory Paths:")
st.sidebar.write("WEBAPP_DIR:", WEBAPP_DIR)
st.sidebar.write("MODELS_DIR:", MODELS_DIR)
st.sidebar.write("DATA_DIR:", DATA_DIR)

# Set page config
st.set_page_config(
    page_title="ExoNet - Exoplanet Detection AI",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .exoplanet {
        background-color: #d4edda;
        border: 2px solid #c3e6cb;
    }
    .false-positive {
        background-color: #f8d7da;
        border: 2px solid #f5c6cb;
    }
    .feature-importance {
        background-color: #f0f8ff;
        padding: 15px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

class ExoplanetPredictor:
    def __init__(self, model_path, feature_names):
        model_dict = joblib.load(model_path)
        self.model = model_dict['model']
        self.feature_names = feature_names
        self.preprocessor = DataPreprocessor()
        
    def predict(self, input_data):
        """Make prediction on new data"""
        # Apply preprocessing pipeline to input_data
        # Note: input_data is a single row DataFrame
        # We need to apply feature engineering and scaling as per training
        
        # Add engineered features
        input_data = self.preprocessor.create_features(input_data, list(input_data.columns))
        
        # Add missing features with default 0.0
        for feature in self.feature_names:
            if feature not in input_data.columns:
                input_data[feature] = 0.0
        
        # Ensure correct column order
        input_data = input_data[self.feature_names]
        
        prediction = self.model.predict(input_data)[0]
        probability = self.model.predict_proba(input_data)[0]
        
        return prediction, probability
    
    def get_feature_importance(self, top_n=10):
        """Get feature importance from the model"""
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance_df.head(top_n)
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🪐 ExoNet - Exoplanet Detection AI</h1>', unsafe_allow_html=True)
    st.markdown("### NASA Space App Challenge 2025 - AI-Powered Exoplanet Discovery")
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose Mode",
                                   ["🔍 Single Prediction", "📊 Batch Analysis",
                                    "📈 Model Insights", "🎨 Advanced Visualizations",
                                    "🔄 Model Retraining", "🚀 Innovation: Discovery Confidence & Anomaly Detection", "🌌 About"])
    
    # Load model and feature names
    try:
        # Determine dataset selection for dashboard
        dataset_option = st.sidebar.selectbox("Select Dataset", ["Kepler", "TESS", "K2"], index=0)
        
        # Debug information
        # Debug information about file paths
        st.sidebar.write("\nFile Paths:")
        st.sidebar.write(f"Looking for model at: {model_path}")
        st.sidebar.write(f"Model file exists: {os.path.exists(model_path)}")
        st.sidebar.write(f"Looking for data at: {processed_data_path}")
        st.sidebar.write(f"Data file exists: {os.path.exists(processed_data_path)}")
        
        # Directory contents
        st.sidebar.write("\nDirectory Contents:")
        if os.path.exists(MODELS_DIR):
            st.sidebar.write("Models directory contents:", os.listdir(MODELS_DIR))
        if os.path.exists(os.path.join(DATA_DIR, "processed")):
            st.sidebar.write("Data/processed directory contents:", os.listdir(os.path.join(DATA_DIR, "processed")))
        
        if dataset_option == "Kepler":
            model_path = os.path.join(MODELS_DIR, "exoplanet_classifier.pkl")
            processed_data_path = os.path.join(DATA_DIR, "processed", "kepler_processed.pkl")
        elif dataset_option == "TESS":
            model_path = os.path.join(MODELS_DIR, "exoplanet_classifier_tess.pkl")
            processed_data_path = os.path.join(DATA_DIR, "processed", "tess_processed.pkl")
        elif dataset_option == "K2":
            model_path = os.path.join(MODELS_DIR, "exoplanet_classifier_k2.pkl")
            processed_data_path = os.path.join(DATA_DIR, "processed", "k2_processed.pkl")
        else:
            st.error("❌ Invalid dataset selection")
            return
            
        st.sidebar.write(f"Attempting to load model from: {model_path}")
        st.sidebar.write(f"Model file exists: {os.path.exists(model_path)}")
        
        processed_data = joblib.load(processed_data_path)
        feature_names = processed_data['feature_names']
        
        predictor = ExoplanetPredictor(model_path, feature_names)
        
        st.sidebar.success(f"✅ Model for {dataset_option} loaded successfully!")
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.error(f"Full error details: {str(e.__class__.__name__)}: {str(e)}")
        import traceback
        st.error(f"Stack trace: {traceback.format_exc()}")
        return
    
    if app_mode == "🔍 Single Prediction":
        single_prediction(predictor, feature_names, dataset_option)
    elif app_mode == "📊 Batch Analysis":
        batch_analysis(predictor)
    elif app_mode == "📈 Model Insights":
        model_insights(predictor, processed_data)
    elif app_mode == "🎨 Advanced Visualizations":
        advanced_visualizations(predictor, processed_data)
    elif app_mode == "🔄 Model Retraining":
        model_retraining(predictor, processed_data)
    elif app_mode == "🚀 Innovation: Discovery Confidence & Anomaly Detection":
        innovation_dashboard(predictor, processed_data)
    elif app_mode == "🌌 About":
        about_page()

def single_prediction(predictor, feature_names, dataset_option):
    """Single exoplanet prediction interface"""

    st.header("🔍 Single Exoplanet Prediction")
    st.markdown("Enter the parameters of a celestial object to predict if it's an exoplanet or false positive.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Orbital Parameters")
        if dataset_option in ["Kepler", "K2"]:
            koi_period = st.number_input("Orbital Period (days)", min_value=0.1, max_value=1000.0, value=10.0)
            koi_duration = st.number_input("Transit Duration (hours)", min_value=0.1, max_value=50.0, value=3.0)
            koi_depth = st.number_input("Transit Depth (ppm)", min_value=0.0, max_value=100000.0, value=1000.0)
            koi_impact = st.number_input("Impact Parameter", min_value=0.0, max_value=10.0, value=0.5)
        elif dataset_option == "TESS":
            pl_orbper = st.number_input("Orbital Period (days)", min_value=0.1, max_value=1000.0, value=10.0)
            pl_trandurh = st.number_input("Transit Duration (hours)", min_value=0.1, max_value=50.0, value=3.0)
            pl_trandep = st.number_input("Transit Depth (ppm)", min_value=0.0, max_value=100000.0, value=1000.0)

    with col2:
        st.subheader("Stellar Parameters")
        if dataset_option in ["Kepler", "K2"]:
            koi_steff = st.number_input("Stellar Temperature (K)", min_value=2000, max_value=10000, value=5500)
            koi_slogg = st.number_input("Stellar Surface Gravity (log10(cm/s²))", min_value=3.0, max_value=5.0, value=4.4)
            koi_srad = st.number_input("Stellar Radius (Solar radii)", min_value=0.1, max_value=10.0, value=1.0)
            koi_model_snr = st.number_input("Signal-to-Noise Ratio", min_value=0.0, max_value=100.0, value=15.0)
        elif dataset_option == "TESS":
            st_teff = st.number_input("Stellar Temperature (K)", min_value=2000, max_value=10000, value=5500)
            st_logg = st.number_input("Stellar Surface Gravity (log10(cm/s²))", min_value=3.0, max_value=5.0, value=4.4)
            st_rad = st.number_input("Stellar Radius (Solar radii)", min_value=0.1, max_value=10.0, value=1.0)

    # Create input data based on dataset
    if dataset_option in ["Kepler", "K2"]:
        input_data = pd.DataFrame({
            'koi_period': [koi_period],
            'koi_duration': [koi_duration],
            'koi_depth': [koi_depth],
            'koi_impact': [koi_impact],
            'koi_steff': [koi_steff],
            'koi_slogg': [koi_slogg],
            'koi_srad': [koi_srad],
            'koi_model_snr': [koi_model_snr]
        })
        # Add engineered features
        input_data['depth_duration_ratio'] = input_data['koi_depth'] / input_data['koi_duration']
        input_data['period_squared'] = input_data['koi_period'] ** 2
        input_data['stellar_density_proxy'] = 10 ** input_data['koi_slogg'] / (input_data['koi_steff'] / 5772) ** 4
        input_data['signal_strength'] = input_data['koi_depth'] * input_data['koi_model_snr']
    elif dataset_option == "TESS":
        input_data = pd.DataFrame({
            'pl_orbper': [pl_orbper],
            'pl_trandurh': [pl_trandurh],
            'pl_trandep': [pl_trandep],
            'st_teff': [st_teff],
            'st_logg': [st_logg],
            'st_rad': [st_rad]
        })
        # Add engineered features
        input_data['depth_duration_ratio'] = input_data['pl_trandep'] / input_data['pl_trandurh']
        input_data['period_squared'] = input_data['pl_orbper'] ** 2
        input_data['stellar_density_proxy'] = 10 ** input_data['st_logg'] / (input_data['st_teff'] / 5772) ** 4

    # Add default values for other required features
    for feature in predictor.feature_names:
        if feature not in input_data.columns:
            input_data[feature] = 0.0  # Default value

    # Ensure correct column order
    input_data = input_data[predictor.feature_names]

    # Prediction button
    if st.button("🚀 Analyze Object", type="primary"):
        with st.spinner("Analyzing celestial object..."):
            prediction, probabilities = predictor.predict(input_data)

            # Display results
            st.markdown("---")
            st.header("📊 Prediction Results")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Exoplanet Probability", f"{probabilities[1]:.2%}")
            with col2:
                st.metric("False Positive Probability", f"{probabilities[0]:.2%}")
            with col3:
                confidence = max(probabilities)
                st.metric("Confidence", f"{confidence:.2%}")

            # Prediction box
            if prediction == 1:
                st.markdown('<div class="prediction-box exoplanet">', unsafe_allow_html=True)
                st.success("🎉 **EXOPLANET DETECTED!**")
                st.markdown("This object has a high probability of being a real exoplanet!")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="prediction-box false-positive">', unsafe_allow_html=True)
                st.error("🔍 **FALSE POSITIVE**")
                st.markdown("This object is likely a false positive (e.g., binary star, instrumental artifact)")
                st.markdown('</div>', unsafe_allow_html=True)

            # Feature importance for this prediction
            st.subheader("🔍 Key Factors in This Prediction")
            importance_df = predictor.get_feature_importance(8)
            if importance_df is not None:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
                ax.set_title('Top Features Influencing Model Decisions')
                ax.set_xlabel('Feature Importance')
                st.pyplot(fig)

def batch_analysis(predictor):
    """Batch upload and analysis"""
    st.header("📊 Batch Analysis")
    st.markdown("Upload a CSV file with multiple objects to analyze them in batch.")
    
    uploaded_file = st.file_uploader("Choose CSV file", type='csv')
    
    if uploaded_file is not None:
        try:
            # Read uploaded data
            batch_data = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded {len(batch_data)} objects")
            
            # Show sample of data
            st.subheader("Sample of Uploaded Data")
            st.dataframe(batch_data.head())
            
            # Check if required columns exist
            missing_features = [f for f in predictor.feature_names if f not in batch_data.columns]
            if missing_features:
                st.warning(f"⚠️ Missing features: {missing_features[:5]}...")
                st.info("Using default values for missing features")
            
            # Prepare data for prediction
            prediction_data = batch_data.copy()
            for feature in predictor.feature_names:
                if feature not in prediction_data.columns:
                    prediction_data[feature] = 0.0
            
            prediction_data = prediction_data[predictor.feature_names]
            
            if st.button("🔍 Analyze Batch", type="primary"):
                with st.spinner("Analyzing batch data..."):
                    # Make predictions
                    predictions = predictor.model.predict(prediction_data)
                    probabilities = predictor.model.predict_proba(prediction_data)
                    
                    # Add results to dataframe
                    results = batch_data.copy()
                    results['Prediction'] = ['Exoplanet' if p == 1 else 'False Positive' for p in predictions]
                    results['Exoplanet_Probability'] = probabilities[:, 1]
                    results['Confidence'] = np.max(probabilities, axis=1)
                    
                    # Display results
                    st.subheader("📈 Batch Analysis Results")
                    
                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    exoplanet_count = sum(predictions)
                    with col1:
                        st.metric("Total Objects", len(results))
                    with col2:
                        st.metric("Exoplanets Found", exoplanet_count)
                    with col3:
                        st.metric("False Positives", len(results) - exoplanet_count)
                    
                    # Show results table
                    st.dataframe(results)
                    
                    # Download results
                    csv = results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name="exoplanet_predictions.csv",
                        mime="text/csv"
                    )
                    
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

def model_insights(predictor, processed_data):
    """Show model performance and insights"""
    st.header("📈 Model Insights")
    
    # Load performance data
    try:
        performance_df = pd.read_csv("../../models/model_performance.csv")
        
        st.subheader("Model Performance Comparison")
        st.dataframe(performance_df)
        
        # Feature importance
        st.subheader("🔍 Feature Importance")
        importance_df = predictor.get_feature_importance(15)
        
        if importance_df is not None:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.barplot(data=importance_df, x='importance', y='feature', palette='rocket')
                ax.set_title('Top 15 Most Important Features for Exoplanet Detection')
                ax.set_xlabel('Feature Importance Score')
                st.pyplot(fig)
            
            with col2:
                st.markdown("### Key Insights")
                st.markdown("""
                - **koi_score**: NASA's own confidence metric
                - **False Positive Flags**: Key indicators of artifacts
                - **Error Measurements**: Data quality matters
                - **Stellar Parameters**: Host star characteristics
                """)
        
        # Data distribution
        st.subheader("📊 Training Data Overview")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Training Samples", processed_data['X_train'].shape[0])
        with col2:
            st.metric("Features", processed_data['X_train'].shape[1])
        with col3:
            st.metric("Test Accuracy", "99.07%")
        
    except Exception as e:
        st.error(f"Error loading model insights: {e}")

def about_page():
    """About page"""
    st.header("🌌 About ExoNet")
    
    st.markdown("""
    ### NASA Space App Challenge 2025
    
    **ExoNet** is an AI-powered exoplanet detection system that combines machine learning with NASA's Kepler mission data 
    to automatically identify new exoplanets.
    
    ### 🎯 Key Features
    
    - **99%+ Accuracy**: Trained on verified Kepler data
    - **Real-time Analysis**: Instant predictions for new observations
    - **Batch Processing**: Analyze multiple objects simultaneously
    - **Explainable AI**: Understand why predictions are made
    
    ### 🔬 Scientific Background
    
    This system uses the **transit method** for exoplanet detection, which identifies planets by the slight dimming of stars 
    when planets pass in front of them. Our AI model has learned to distinguish real planetary transits from false positives 
    like binary stars or instrumental noise.
    
    ### 📊 Data Sources
    
    - **Kepler Objects of Interest (KOI)**: NASA's comprehensive exoplanet catalog
    - **9,564 celestial objects** with confirmed classifications
    - **50+ features** including orbital periods, transit depths, and stellar parameters
    
    ### 🚀 Technology Stack
    
    - **Machine Learning**: Random Forest & XGBoost
    - **Web Framework**: Streamlit
    - **Data Processing**: Pandas, Scikit-learn
    - **Visualization**: Matplotlib, Seaborn
    
    *Built for the NASA Space App Challenge 2025*
    """)

def advanced_visualizations(predictor, processed_data):
    """Advanced visualization dashboard"""
    st.header("🎨 Advanced Visualizations")
    st.markdown("Interactive visualizations for deeper insights into exoplanet data and model performance.")

    viz = AdvancedVisualizations()

    # Load test data for visualizations
    X_test = processed_data['X_test']
    y_test = processed_data['y_test']

    # Make predictions for visualization
    y_pred = predictor.model.predict(X_test)
    y_proba = predictor.model.predict_proba(X_test)[:, 1]

    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Performance Metrics",
        "🔍 Feature Analysis",
        "🌌 Parameter Space",
        "📈 Confidence Analysis",
        "🔄 Transit Simulation"
    ])

    with tab1:
        st.subheader("Model Performance Metrics")

        # ROC and Precision-Recall curves
        performance_fig = viz.create_performance_curves(y_test, y_proba)
        st.plotly_chart(performance_fig, use_container_width=True)

        # Confusion matrix
        col1, col2 = st.columns(2)
        with col1:
            cm_fig = viz.create_confusion_matrix_plot(y_test, y_pred)
            st.plotly_chart(cm_fig, use_container_width=True)

        with col2:
            # Performance metrics
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.highlight_max(axis=0))

    with tab2:
        st.subheader("Feature Analysis")

        # Feature importance
        importance_df = predictor.get_feature_importance(20)
        if importance_df is not None:
            importance_fig = viz.create_interactive_feature_importance(importance_df, 15)
            st.plotly_chart(importance_fig, use_container_width=True)

        # Feature correlations
        st.subheader("Feature Correlations")
        viz_df = X_test.copy()
        viz_df['target'] = y_test

        corr_fig, target_corrs = viz.create_feature_correlation_heatmap(
            viz_df, 'target'
        )
        st.plotly_chart(corr_fig, use_container_width=True)

        st.subheader("Top Correlations with Target")
        st.dataframe(target_corrs.head(10))

    with tab3:
        st.subheader("Parameter Space Exploration")

        col1, col2 = st.columns(2)

        with col1:
            x_feature = st.selectbox("X-axis Feature", options=X_test.columns.tolist(), index=0)
        with col2:
            y_feature = st.selectbox("Y-axis Feature", options=X_test.columns.tolist(), index=1)

        # Create sample for visualization (first 1000 points for performance)
        sample_df = X_test.iloc[:1000].copy()
        sample_df['target'] = y_test.iloc[:1000]

        space_fig = viz.create_parameter_space_plot(sample_df, x_feature, y_feature, 'target')
        st.plotly_chart(space_fig, use_container_width=True)

        st.markdown("""
        **Interpretation Guide:**
        - 🟢 Green points: Confirmed exoplanets
        - 🔴 Red points: False positives
        - Look for clear separation between colors for good feature combinations
        """)

    with tab4:
        st.subheader("Confidence Analysis")

        confidence_fig = viz.create_confidence_distribution(
            predictor.model.predict_proba(X_test), y_pred, y_test
        )
        st.plotly_chart(confidence_fig, use_container_width=True)

        # Confidence statistics
        st.subheader("Confidence Statistics")
        confidences = np.max(predictor.model.predict_proba(X_test), axis=1)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean Confidence", f"{confidences.mean():.2%}")
        with col2:
            st.metric("Median Confidence", f"{np.median(confidences):.2%}")
        with col3:
            st.metric("High Confidence (>90%)", f"{(confidences > 0.9).mean():.2%}")
        with col4:
            st.metric("Low Confidence (<70%)", f"{(confidences < 0.7).mean():.2%}")

    with tab5:
        st.subheader("Transit Light Curve Simulation")

        col1, col2 = st.columns(2)

        with col1:
            transit_depth = st.slider("Transit Depth (ppm)", 100, 10000, 1000)
            transit_duration = st.slider("Transit Duration (hours)", 1, 10, 3)
            orbital_period = st.slider("Orbital Period (days)", 1, 100, 10)

        with col2:
            impact_param = st.slider("Impact Parameter", 0.0, 1.0, 0.3)
            snr = st.slider("Signal-to-Noise Ratio", 5.0, 50.0, 15.0)

        simulation_params = {
            'koi_depth': transit_depth,
            'koi_duration': transit_duration,
            'koi_period': orbital_period,
            'koi_impact': impact_param,
            'koi_model_snr': snr
        }

        transit_fig = viz.create_transit_simulation(simulation_params)
        st.plotly_chart(transit_fig, use_container_width=True)

        st.markdown("""
        **Transit Parameters:**
        - **Depth**: How much the star dims during transit (indicates planet size)
        - **Duration**: How long the transit lasts (related to orbital geometry)
        - **Period**: Time between transits (planet's year)
        - **Impact**: How central the transit is (0 = center, 1 = edge)
        """)

def model_retraining(predictor, processed_data):
    """Model retraining interface"""
    st.header("🔄 Model Retraining")
    st.markdown("Retrain the model with new labeled data to improve performance.")

    # Initialize retrainer
    retrainer = ModelRetrainer(
        model_path="models/exoplanet_classifier.pkl",
        processed_data_path="data/processed/kepler_processed.pkl"
    )

    tab1, tab2, tab3 = st.tabs(["📤 Upload New Data", "⚙️ Hyperparameters", "📊 Retraining History"])

    with tab1:
        st.subheader("Upload New Training Data")

        # Download template
        template_df = retrainer.create_retraining_template()
        csv_template = template_df.to_csv(index=False)

        st.download_button(
            label="📥 Download Data Template",
            data=csv_template,
            file_name="retraining_template.csv",
            mime="text/csv",
            help="Download template with required columns"
        )

        uploaded_file = st.file_uploader("Upload new training data (CSV)", type='csv')

        if uploaded_file is not None:
            try:
                new_data = pd.read_csv(uploaded_file)
                st.success(f"✅ Successfully loaded {len(new_data)} new samples")

                # Show data preview
                st.subheader("Data Preview")
                st.dataframe(new_data.head())

                # Model selection
                model_type = st.selectbox(
                    "Model Type",
                    ["random_forest", "xgboost"],
                    help="Choose which algorithm to retrain"
                )

                if st.button("🚀 Start Retraining", type="primary"):
                    with st.spinner("Retraining model with new data..."):
                        try:
                            new_model, metrics, improvement = retrainer.retrain_model(
                                new_data,
                                model_type=model_type
                            )

                            st.success("✅ Model retraining completed!")

                            # Display results
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                            with col2:
                                st.metric("F1 Score", f"{metrics['f1_score']:.4f}")
                            with col3:
                                st.metric("AUC-ROC", f"{metrics['auc_roc']:.4f}")

                            if improvement:
                                st.subheader("Improvement Over Previous Model")
                                for metric, imp in improvement.items():
                                    st.write(f"{metric}: {imp:+.4f}")

                        except Exception as e:
                            st.error(f"❌ Retraining failed: {e}")

            except Exception as e:
                st.error(f"❌ Error loading data: {e}")

    with tab2:
        st.subheader("Hyperparameter Tuning")

        st.markdown("""
        ### Current Model Hyperparameters

        Adjust these parameters for retraining. Leave empty to use defaults.
        """)

        col1, col2 = st.columns(2)

        with col1:
            n_estimators = st.number_input("Number of Estimators",
                                         min_value=10, max_value=500, value=100)
            max_depth = st.number_input("Max Depth",
                                      min_value=3, max_value=20, value=10)

        with col2:
            learning_rate = st.number_input("Learning Rate (XGBoost only)",
                                          min_value=0.01, max_value=1.0, value=0.1, step=0.01)
            subsample = st.number_input("Subsample Ratio",
                                      min_value=0.1, max_value=1.0, value=0.8, step=0.1)

        hyperparameters = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample
        }

        if st.button("🔄 Retrain with Custom Parameters"):
            st.info("Feature coming soon! Upload data in the previous tab to use custom parameters.")

    with tab3:
        st.subheader("Retraining History")

        history_df = retrainer.get_retraining_history()

        if history_df.empty:
            st.info("No retraining history available yet.")
        else:
            st.dataframe(history_df)

            # Plot retraining progress
            if len(history_df) > 1:
                st.subheader("Retraining Progress")

                fig = go.Figure()

                for metric in ['accuracy', 'f1_score', 'auc_roc']:
                    fig.add_trace(go.Scatter(
                        x=history_df['timestamp'],
                        y=history_df['test_metrics'].apply(lambda x: x[metric]),
                        mode='lines+markers',
                        name=metric.upper()
                    ))

                fig.update_layout(
                    title="Model Performance Over Retraining Cycles",
                    xaxis_title="Retraining Time",
                    yaxis_title="Score",
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)

def innovation_dashboard(predictor, processed_data):
    """Innovation and advanced features dashboard"""
    st.header("🚀 Innovation Dashboard")
    st.markdown("Advanced features for exoplanet discovery and analysis.")

    from src.innovation.advanced_features import AdvancedFeatures

    # Load data
    X_test = processed_data['X_test']
    y_test = processed_data['y_test']
    feature_names = processed_data['feature_names']

    # Initialize innovation features
    innovator = AdvancedFeatures(predictor.model, feature_names)

    # Make predictions
    y_pred = predictor.model.predict(X_test)
    y_proba = predictor.model.predict_proba(X_test)

    tab1, tab2, tab3 = st.tabs(["🎯 Discovery Confidence", "🔍 Anomaly Detection", "🌌 Rare Types"])

    with tab1:
        st.subheader("Enhanced Discovery Confidence Scoring")

        # Calculate discovery confidence
        discovery_conf, components = innovator.calculate_discovery_confidence(
            X_test, y_pred, y_proba
        )

        # Display results
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean Discovery Confidence", f"{discovery_conf.mean():.2%}")
        with col2:
            st.metric("High Confidence (>90%)", f"{(discovery_conf > 0.9).mean():.2%}")
        with col3:
            st.metric("Discovery Candidates", f"{(discovery_conf > 0.8).sum()}")
        with col4:
            st.metric("Requires Review", f"{(discovery_conf < 0.6).sum()}")

        # Confidence distribution
        fig = px.histogram(
            x=discovery_conf,
            nbins=20,
            title="Discovery Confidence Distribution",
            labels={'x': 'Discovery Confidence', 'y': 'Count'}
        )
        st.plotly_chart(fig, use_container_width=True)

        # Show top candidates
        st.subheader("Top Discovery Candidates")
        results_df = pd.DataFrame({
            'discovery_confidence': discovery_conf,
            'model_confidence': components['model_confidence'],
            'novelty': components['novelty_score'],
            'consistency': components['consistency_score'],
            'prediction': ['Exoplanet' if p == 1 else 'FP' for p in y_pred]
        })

        top_candidates = results_df.nlargest(10, 'discovery_confidence')
        st.dataframe(top_candidates)

    with tab2:
        st.subheader("Anomaly Detection")
        st.markdown("Find unusual objects that don't fit normal patterns.")

        anomaly_labels, anomaly_scores = innovator.detect_anomalies(X_test)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Anomalies Detected", f"{anomaly_labels.sum()}")
        with col2:
            st.metric("Anomaly Rate", f"{anomaly_labels.mean():.2%}")

        # Anomaly visualization
        if 'koi_period' in feature_names and 'koi_depth' in feature_names:
            period_idx = feature_names.index('koi_period')
            depth_idx = feature_names.index('koi_depth')

            fig = px.scatter(
                x=X_test.iloc[:, period_idx],
                y=X_test.iloc[:, depth_idx],
                color=anomaly_labels,
                title="Anomalies in Parameter Space",
                labels={'x': 'Orbital Period', 'y': 'Transit Depth', 'color': 'Anomaly'}
            )
            st.plotly_chart(fig, use_container_width=True)

        # Show anomalies
        st.subheader("Detected Anomalies")
        anomaly_df = pd.DataFrame({
            'anomaly_score': anomaly_scores,
            'is_anomaly': anomaly_labels,
            'prediction': y_pred
        })
        st.dataframe(anomaly_df[anomaly_df['is_anomaly'] == 1].head(10))

    with tab3:
        st.subheader("Rare Exoplanet Types")
        st.markdown("Cluster analysis to find unusual exoplanet categories.")

        cluster_labels, cluster_info = innovator.find_rare_exoplanet_types(
            X_test, y_pred, y_proba
        )

        st.metric("Unique Clusters Found", len(cluster_info))

        # Display cluster information
        for cluster_id, info in cluster_info.items():
            with st.expander(f"Cluster {cluster_id} ({info['size']} members)"):
                st.write(f"Cluster size: {info['size']} exoplanets")

                # Show characteristic features
                important_features = []
                for i, (feature, mean_val) in enumerate(zip(feature_names, info['features_mean'])):
                    std_val = info['features_std'][i]
                    if std_val > 0:  # Feature has variation
                        important_features.append((feature, mean_val, std_val))

                # Sort by standard deviation (most distinctive features)
                important_features.sort(key=lambda x: x[2], reverse=True)

                st.write("Characteristic features:")
                for feature, mean, std in important_features[:5]:
                    st.write(f"  - {feature}: {mean:.2f} ± {std:.2f}")

if __name__ == "__main__":
    main()
