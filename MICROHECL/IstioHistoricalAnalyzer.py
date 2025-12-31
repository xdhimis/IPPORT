import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from prometheus_api_client import PrometheusConnect, utl
from datetime import datetime, timedelta

class IstioHistoricalAnalyzer:
    def __init__(self, url):
        self.prom = PrometheusConnect(url=url, disable_ssl=True)

    def fetch_range_df(self, query, start_time, end_time, step="1m"):
        """Fetches historical time series and flattens them for ML processing."""
        res = self.prom.custom_query_range(
            query=query,
            start_time=start_time,
            end_time=end_time,
            step=step
        )
        
        flattened_data = []
        for item in res:
            metric_info = item['metric']
            # item['values'] contains [[timestamp, value], [timestamp, value]...]
            for ts, val in item['values']:
                row = metric_info.copy()
                row['timestamp'] = datetime.fromtimestamp(ts)
                row['value'] = float(val)
                flattened_data.append(row)
        
        return pd.DataFrame(flattened_data)

    def analyze_past_range(self, hours_back=3):
        # 1. Define Time Window
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours_back)

        # 2. Historical Queries
        # We fetch the rate of requests and P95 latency over the whole range
        q_lat = 'histogram_quantile(0.95, sum(rate(istio_request_duration_milliseconds_bucket{reporter="source"}[5m])) by (le, source_canonical_service, destination_canonical_service))'
        q_err = 'sum(rate(istio_requests_total{reporter="source", response_code=~"5.."}[5m])) by (source_canonical_service, destination_canonical_service) / ' \
                'sum(rate(istio_requests_total{reporter="source"}[5m])) by (source_canonical_service, destination_canonical_service)'

        print(f"Fetching data from {start_time} to {end_time}...")
        df_lat = self.fetch_range_df(q_lat, start_time, end_time)
        df_err = self.fetch_range_df(q_err, start_time, end_time)

        if df_lat.empty: return "No historical data found."

        # 3. Merge and prepare for ML
        # We group by service pair and timestamp to align latency and errors
        master_df = pd.merge(
            df_lat.rename(columns={'value': 'latency_ms'}),
            df_err.rename(columns={'value': 'error_rate'}),
            on=['timestamp', 'source_canonical_service', 'destination_canonical_service'],
            how='left'
        ).fillna(0)

        # 4. Isolation Forest on the Time Series
        # This identifies moments in time where a service-pair behaved abnormally
        model = IsolationForest(contamination=0.05)
        features = master_df[['latency_ms', 'error_rate']]
        master_df['anomaly_score'] = model.fit_predict(features)
        
        return master_df

# --- Execution ---
analyzer = IstioHistoricalAnalyzer("http://prometheus.istio-system:9090")
# Analyze the last 6 hours of service mesh behavior
history_df = analyzer.analyze_past_range(hours_back=6)

# Find the specific service pairs that had the most "Anomalous Minutes"
summary = history_df[history_df['anomaly_score'] == -1].groupby(
    ['source_canonical_service', 'destination_canonical_service']
).size().reset_index(name='anomalous_minutes_count')

print(summary.sort_values(by='anomalous_minutes_count', ascending=False))
