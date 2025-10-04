# ExoNet - AI-Powered Exoplanet Detection Platform 🪐

ExoNet is a sophisticated AI platform developed for NASA Space Apps Challenge 2025, designed to detect and analyze potential exoplanets using data from multiple space telescopes including Kepler, TESS, and K2.

## 🌟 Features

- **## 🙏 Acknowledgmentsulti-Mission Support**: Analyze data from different space telescopes:
  - Kepler Space Telescope
  - TESS (Transiting Exoplanet Survey Satellite)
  - K2 Mission
- **Interactive Web Interface**: User-friendly Streamlit application for:
  - Single object analysis
  - Batch prediction capabilities
  - Advanced data visualization
  - Model performance insights
- **Advanced AI Models**: Machine learning models trained on vast datasets from NASA's Exoplanet Archive
- **Real-time Analysis**: Quick and accurate predictions for potential exoplanet candidates

## 🚀 Getting Started

### Prerequisites

- Python 3.13+
- pip (Python package manager)
- Git
- 8GB+ RAM recommended
- CUDA-capable GPU (optional, for faster processing)

### Requirements

```txt
numpy>=1.24.0
pandas>=2.1.0
scikit-learn>=1.3.0
tensorflow>=2.14.0  # Optional, for deep learning models
torch>=2.1.0        # Optional, for deep learning models
streamlit>=1.27.0
matplotlib>=3.8.0
seaborn>=0.13.0
astropy>=5.3.0
astroquery>=0.4.6
lightkurve>=2.4.0
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/WAQASCHANNA/NASA-Space-app-Challenge25-Exoplanet-AI-Platform.git
cd exoplanet-ai-platform
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the web application:
```bash
cd webapp
streamlit run app.py
```

## 📂 Project Structure

```
exoplanet-ai-platform/
│
├── data/                      # Data directory
│   ├── raw/                  # Raw data files
│   └── processed/            # Processed datasets
│
├── models/                   # Trained ML models
│   ├── exoplanet_classifier.pkl
│   ├── exoplanet_classifier_tess.pkl
│   └── exoplanet_classifier_k2.pkl
│
├── notebooks/               # Jupyter notebooks
│   ├── 01_data_exploration.py
│   ├── 02_data_preprocessing.py
│   ├── 02_data_preprocessing_tess.py
│   ├── 02_data_preprocessing_k2.py
│   └── 03_model_training.py
│
├── src/                    # Source code
│   ├── data_loader.py
│   ├── processing/
│   ├── models/
│   ├── visualization/
│   └── utils/
│
├── webapp/                 # Streamlit web application
│   ├── app.py
│   └── requirements.txt
│
├── config.py              # Configuration settings
├── requirements.txt       # Project dependencies
└── README.md             # Project documentation
```

## 🛠️ Technical Details

### Data Sources
- **Kepler**: NASA's Exoplanet Archive (API)
- **TESS**: Local dataset from NASA's TOI catalog
- **K2**: Local dataset from K2 mission

### Machine Learning Pipeline
1. **Data Preprocessing**:
   - Missing value imputation
   - Outlier detection and removal
   - Feature engineering
   - Data normalization

2. **Model Architecture**:
   - Random Forest Classifier
   - Feature importance analysis
   - Cross-validation
   - Hyperparameter optimization

3. **Evaluation Metrics**:
   - Precision
   - Recall
   - F1 Score
   - ROC-AUC

## 📊 Features Used

### Kepler Features
- Orbital period
- Transit duration
- Transit depth
- Impact parameter
- Stellar parameters (temperature, surface gravity, radius)
- Signal-to-noise ratio

### TESS Features
- Orbital period
- Transit duration/depth
- Planetary radius
- Stellar parameters
- Insolation flux
- Equilibrium temperature

### K2 Features
- Orbital period
- Planetary radius
- Insolation flux
- Stellar parameters
- Transit duration

## 🎯 Use Cases

1. **Single Object Analysis**:
   - Input object parameters
   - Get real-time predictions
   - View feature importance
   - Confidence scores

2. **Batch Analysis**:
   - Upload multiple objects
   - Bulk predictions
   - Export results

3. **Model Insights**:
   - Performance metrics
   - Feature importance
   - Decision boundaries

## 📈 Performance

- Kepler Model Accuracy: ~95%
- TESS Model Accuracy: ~92%
- K2 Model Accuracy: ~90%

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## � Future Work

### Model Improvements
1. **Deep Learning Integration**:
   - Implement CNN models for light curve analysis
   - Add RNN/LSTM models for time-series data
   - Explore transformer architectures for better feature extraction

2. **Advanced Feature Engineering**:
   - Incorporate stellar oscillation patterns
   - Add spectral analysis features
   - Include star cluster membership data
   - Develop automated feature selection

3. **Multi-Mission Fusion**:
   - Cross-validate findings between different telescopes
   - Develop ensemble models combining multiple missions
   - Create unified feature representation across missions

### Platform Enhancements
1. **Real-time Processing**:
   - Live data streaming from telescopes
   - Automated alerts for promising candidates
   - Real-time model updates

2. **Advanced Visualization**:
   - 3D orbital visualization
   - Interactive light curve analysis
   - Comparative system visualization
   - AR/VR integration for data exploration

3. **API Development**:
   - RESTful API for external access
   - WebSocket support for real-time updates
   - Integration with other astronomical tools

### Additional Features
1. **Automated Report Generation**:
   - Detailed PDF reports for discoveries
   - Publication-ready figures
   - Automated literature search

2. **Community Features**:
   - User authentication system
   - Collaborative analysis tools
   - Discovery sharing platform
   - Peer review system

3. **Educational Components**:
   - Interactive tutorials
   - Educational resources
   - Citizen science integration
   - Student research support

### Data Integration
1. **New Data Sources**:
   - James Webb Space Telescope data
   - Ground-based observatory integration
   - Amateur astronomer data pipeline
   - Additional space telescope missions

2. **Enhanced Data Processing**:
   - Improved noise reduction
   - Better artifact removal
   - Advanced detrending methods
   - Automated quality assessment

3. **Data Management**:
   - Distributed storage system
   - Version control for datasets
   - Automated backup system
   - Data integrity checks

### Research Directions
1. **Scientific Expansion**:
   - Habitability analysis
   - Atmospheric composition prediction
   - Planet formation models
   - System stability analysis

2. **Collaboration Tools**:
   - Integration with research databases
   - Automated paper drafting
   - Citation management
   - Research workflow automation

### Infrastructure
1. **Cloud Integration**:
   - Multi-cloud deployment
   - Serverless computing options
   - Edge computing for preprocessing
   - GPU acceleration support

2. **Scalability**:
   - Microservices architecture
   - Container orchestration
   - Load balancing
   - Automated scaling

3. **Security**:
   - Enhanced data protection
   - Access control systems
   - Audit logging
   - Compliance management

## �🙏 Acknowledgments

- NASA Space Apps Challenge 2025
- NASA's Exoplanet Archive
- TESS Science Team
- Kepler/K2 Science Team
- Streamlit Community

## ❓ Troubleshooting

### Common Issues

1. **Installation Problems**
   ```bash
   # If you encounter SSL certificate issues:
   pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
   ```

2. **CUDA Issues**
   - Ensure CUDA toolkit matches your TensorFlow/PyTorch version
   - Check GPU compatibility with `nvidia-smi`

3. **Memory Issues**
   - Reduce batch size in config.py
   - Use data streaming for large datasets
   - Clear cache: `import torch; torch.cuda.empty_cache()`

4. **Data Loading**
   - Check internet connection for API access
   - Verify data file permissions
   - Ensure correct file paths in config.py

### Getting Help
- Open an issue on GitHub
- Check existing issues for solutions
- Join our Discord community

## 📧 Contact

For questions and support:
- GitHub Issues: [Project Issues](https://github.com/yourusername/exoplanet-ai-platform/issues)
- Email: exonet.support@example.com
- Discord: [ExoNet Community](https://discord.gg/exonet)

Project Link: [https://github.com/WAQASCHANNA/NASA-Space-app-Challenge25-Exoplanet-AI-Platform]

---
*Built with ❤️ for NASA Space Apps Challenge 2025*
