# config.py - UPDATED WITH CORRECT URLS

DATA_SOURCES = {
    "kepler": {
        # Correct API endpoint for Kepler data
        "url": "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=cumulative&format=csv",
        "target_column": "koi_disposition",
        "feature_columns": [
            'koi_period',           # Orbital period (days)
            'koi_time0bk',          # Transit epoch
            'koi_impact',           # Impact parameter
            'koi_duration',         # Transit duration (hours)
            'koi_depth',            # Transit depth (ppm)
            'koi_prad',             # Planetary radius (Earth radii)
            'koi_teq',              # Equilibrium temperature (K)
            'koi_insol',            # Insolation flux (Earth fluxes)
            'koi_model_snr',        # Signal-to-Noise ratio
            'koi_steff',            # Stellar effective temperature (K)
            'koi_slogg',            # Stellar surface gravity (log10(cm/s**2))
            'koi_srad',             # Stellar radius (Solar radii)
            'koi_kepmag'            # Kepler magnitude
        ]
    },
    "tess": {
        # Correct API endpoint for TESS data
        "url": "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=toi&format=csv",
        "target_column": "tfopwg_disp",
        "feature_columns": [
            'pl_orbper',            # Orbital period (days)
            'pl_trandurh',          # Transit duration (hours)
            'pl_trandep',           # Transit depth (ppm)
            'pl_rade',              # Planetary radius (Earth radii)
            'pl_insol',             # Insolation flux (Earth fluxes)
            'pl_eqt',               # Equilibrium temperature (K)
            'st_tmag',              # TESS magnitude
            'st_dist',              # Stellar distance (pc)
            'st_teff',              # Stellar effective temperature (K)
            'st_logg',              # Stellar surface gravity (log10(cm/s**2))
            'st_rad'                # Stellar radius (Solar radii)
        ]
    },
    "k2": {
        # K2 data loaded from local CSV file
        "url": "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=k2pandc&format=csv",
        "target_column": "disposition",
        "feature_columns": [
            'pl_orbper',        # Orbital period (days)
            'pl_rade',          # Planet radius (Earth radii)
            'pl_insol',         # Insolation flux (Earth fluxes)
            'pl_eqt',           # Equilibrium temperature (K)
            'st_teff',          # Stellar effective temperature (K)
            'st_logg',          # Stellar surface gravity (log10(cm/s**2))
            'st_rad',           # Stellar radius (Solar radii)
            'st_mass',          # Stellar mass (Solar masses)
            'sy_dist',          # Distance (pc)
            'pl_trandur'        # Transit duration (hours)
        ]
    }
}

# Model configuration
MODEL_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "validation_size": 0.1
}

# Path configuration
PATHS = {
    "raw_data": "data/raw",
    "processed_data": "data/processed",
    "models": "models",
    "notebooks": "notebooks"
}

# Data preprocessing
PREPROCESSING = {
    "min_samples_per_class": 100,
    "test_size": 0.2,
    "random_state": 42
}
