import pandas as pd
import numpy as np
import networkx as nx
from sklearn.ensemble import IsolationForest
from prometheus_api_client import PrometheusConnect
import json

class IstioIntelligenceEngine:
    def __init__(self, url):
        self.prom = PrometheusConnect(url=url, disable_ssl=True)
    
    def fetch(self, query):
        res = self.prom.custom_query(query=query)
        return pd.DataFrame([dict(item['metric'], value=float(item['value'][1])) for item in res])

    ## --- MODULE: Rate & Error Rate ---
    def get_service_health(self):
        # Query Rate (RPS)
        q_rate = 'sum(rate(istio_requests_total{reporter="source"}[5m])) by (source_canonical_service, destination_canonical_service)'
        # Error Rate %
        q_error = 'sum(rate(istio_requests_total{reporter="source", response_code=~"5.."}[5m])) by (source_canonical_service, destination_canonical_service) / ' \
                  'sum(rate(istio_requests_total{reporter="source"}[5m])) by (source_canonical_service, destination_canonical_service)'
        
        df_r = self.fetch(q_rate).rename(columns={'value': 'rps'})
        df_e = self.fetch(q_error).rename(columns={'value': 'error_rate'})
        
        # Fill NaN error rates with 0 (meaning no errors)
        return pd.merge(df_r, df_e, on=['source_canonical_service', 'destination_canonical_service'], how='left').fillna(0)

    ## --- MODULE: Isolation Forest Anomaly Detection ---
    def detect_ml_anomalies(self, df):
        if df.empty or len(df) < 5: return df # Need enough points for IF
        
        # Prepare data for Isolation Forest
        # We use latency and error_rate as features to find outliers
        model = IsolationForest(contamination=0.1, random_state=42)
        features = df[['latency_ms', 'error_rate']].values
        
        # Fit and Predict (-1 is anomaly, 1 is normal)
        df['ml_score'] = model.fit_predict(features)
        df['is_outlier'] = df['ml_score'] == -1
        return df

    ## --- MODULE: Unified Orchestrator ---
    def run_full_analysis(self):
        # Fetch Latency (P95)
        q_lat = 'histogram_quantile(0.95, sum(rate(istio_request_duration_milliseconds_bucket{reporter="source"}[5m])) by (le, source_canonical_service, destination_canonical_service))'
        df_lat = self.fetch(q_lat).rename(columns={'value': 'latency_ms'})
        
        # Fetch Health (Rate/Errors)
        df_health = self.get_service_health()
        
        # Final Join
        master_df = pd.merge(df_lat, df_health, on=['source_canonical_service', 'destination_canonical_service'])
        
        # Run ML Anomaly Detection
        master_df = self.detect_ml_anomalies(master_df)
        
        return master_df

# --- EXECUTION ---
engine = IstioIntelligenceEngine("http://prometheus.istio-system:9090")
report = engine.run_full_analysis()

# Highlight critical failures
top_anomalies = report[report['is_outlier'] == True].sort_values(by='latency_ms', ascending=False)
print("Top Anomalous Services Detected via Isolation Forest:")
print(top_anomalies[['source_canonical_service', 'destination_canonical_service', 'latency_ms', 'error_rate']])
