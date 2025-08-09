from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np
from joblib import dump, load
import os

# Configuration
FEATURES = [
    'conn_zscore_mean', 'conn_zscore_max',
    'conn_log_mean', 'conn_log_median', 'conn_log_max',
    'hour_mean', 'day_of_week_mean'
]
ANOMALY_THRESHOLD = 0.7  # Threshold for anomaly classification
MODEL_PATH = 'isolation_forest.joblib'
SCALER_PATH = 'scaler.joblib'
STATS_PATH = 'train_data_stats.joblib'

def train_model(data, features=FEATURES):
    if data.empty:
        return None, None, {}
    
    scaler = StandardScaler()
    X = scaler.fit_transform(data[features])
    X = np.nan_to_num(X, nan=0.0)
    
    iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    iso_forest.fit(X)
    
    # Save model and scaler
    dump(iso_forest, MODEL_PATH)
    dump(scaler, SCALER_PATH)
    
    return iso_forest, scaler

def predict_anomalies(data, train_data_stats, features=FEATURES):
    if data.empty:
        return pd.DataFrame()
    
    iso_forest = load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
    scaler = load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
    
    if iso_forest is None or scaler is None:
        return pd.DataFrame()
    
    X = scaler.transform(data[features])
    X = np.nan_to_num(X, nan=0.0)
    
    # Predict anomaly scores (negated: higher = worse)
    data['anomaly_score'] = -iso_forest.score_samples(X)
    
    # Apply threshold
    data['is_anomaly'] = data['anomaly_score'] > ANOMALY_THRESHOLD
    
    # Create service identifier
    data['service_id'] = data['ServerName'] + '_' + data['IP_port']
    
    return data