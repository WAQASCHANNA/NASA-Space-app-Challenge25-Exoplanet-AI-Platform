# notebooks/04_create_batch_template.py

import pandas as pd
import numpy as np
import joblib

def create_batch_template():
    """Create a template CSV file for batch testing"""
    
    # Load the processed data to get feature names
    try:
        processed_data = joblib.load('../data/processed/kepler_processed.pkl')
        feature_names = processed_data['feature_names']
        
        print(f"Using {len(feature_names)} features from trained model")
        
    except:
        # Fallback to our known important features
        feature_names = [
            'koi_period', 'koi_duration', 'koi_depth', 'koi_impact',
            'koi_steff', 'koi_slogg', 'koi_srad', 'koi_model_snr',
            'koi_prad', 'koi_teq', 'koi_insol', 'koi_score'
        ]
        print("Using default important features")
    
    # Create sample data with realistic values
    sample_data = []
    
    # Sample 1: Likely Exoplanet (based on Kepler-186f parameters)
    sample_data.append({
        'koi_period': 129.9,      # Orbital period in days
        'koi_duration': 4.5,      # Transit duration in hours  
        'koi_depth': 432.1,       # Transit depth in ppm
        'koi_impact': 0.3,        # Impact parameter
        'koi_steff': 3755,        # Stellar temperature (K)
        'koi_slogg': 4.7,         # Stellar surface gravity
        'koi_srad': 0.47,         # Stellar radius (Solar radii)
        'koi_model_snr': 18.2,    # Signal-to-noise ratio
        'koi_prad': 1.11,         # Planetary radius (Earth radii)
        'koi_teq': 230,           # Equilibrium temperature (K)
        'koi_insol': 0.32,        # Insolation flux
        'koi_score': 0.98,        # NASA confidence score
        'object_id': 'K186.01',   # Custom identifier
        'star_name': 'Kepler-186' # Star name
    })
    
    # Sample 2: Likely False Positive (binary star system)
    sample_data.append({
        'koi_period': 2.4,
        'koi_duration': 2.1, 
        'koi_depth': 15200.0,     # Very deep - suggests large object
        'koi_impact': 0.1,
        'koi_steff': 5800,
        'koi_slogg': 4.4,
        'koi_srad': 1.02,
        'koi_model_snr': 45.6,    # Very high SNR but deep transit
        'koi_prad': 8.5,          # Very large radius
        'koi_teq': 850,
        'koi_insol': 12.4,
        'koi_score': 0.45,        # Low confidence score
        'object_id': 'K123.01',
        'star_name': 'Unknown'
    })
    
    # Sample 3: Borderline case
    sample_data.append({
        'koi_period': 15.3,
        'koi_duration': 3.2,
        'koi_depth': 890.5,
        'koi_impact': 0.6,
        'koi_steff': 5200,
        'koi_slogg': 4.5,
        'koi_srad': 0.85,
        'koi_model_snr': 12.1,
        'koi_prad': 2.3,
        'koi_teq': 450,
        'koi_insol': 1.8,
        'koi_score': 0.72,
        'object_id': 'K456.01', 
        'star_name': 'Test-Star'
    })
    
    # Create DataFrame
    df = pd.DataFrame(sample_data)
    
    # Add any missing features with default values
    for feature in feature_names:
        if feature not in df.columns:
            df[feature] = 0.0  # Default value for missing features
    
    # Ensure we have all required features in correct order
    final_columns = ['object_id', 'star_name'] + feature_names
    df = df[final_columns]
    
    # Save template
    template_path = '../data/batch_template.csv'
    df.to_csv(template_path, index=False)
    
    print(f"Batch template created: {template_path}")
    print(f"Sample data includes {len(df)} test objects:")
    print(f"   - 1 likely exoplanet")
    print(f"   - 1 likely false positive")
    print(f"   - 1 borderline case")
    print(f"\nYou can add more rows to this file for batch analysis!")
    
    return df

if __name__ == "__main__":
    create_batch_template()
