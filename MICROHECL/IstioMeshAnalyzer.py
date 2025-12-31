import pandas as pd
import networkx as nx
from prometheus_api_client import PrometheusConnect
from prometheus_api_client.utils import parse_datetime

class IstioMeshAnalyzer:
    def __init__(self, prom_url):
        self.prom = PrometheusConnect(url=prom_url, disable_ssl=True)
        self.graph = nx.DiGraph()

    def fetch_df(self, query):
        """Helper to convert PromQL results to a cleaned Pandas DataFrame."""
        result = self.prom.custom_query(query=query)
        rows = []
        for item in result:
            row = item['metric']
            row['value'] = float(item['value'][1])
            rows.append(row)
        return pd.DataFrame(rows)

    def analyze(self):
        # 1. Fetch Current P95 Latency (Last 5m)
        # We use canonical labels for logical caller-callee mapping
        current_latency_q = """
        histogram_quantile(0.95, 
            sum(rate(istio_request_duration_milliseconds_bucket{reporter="source"}[5m])) 
            by (le, source_canonical_service, destination_canonical_service)
        )
        """
        
        # 2. Fetch Historical Baseline (Avg & StdDev over last 1h)
        # This calculates the 'normal' behavior for every pair
        baseline_avg_q = """
        avg_over_time(
            (sum(rate(istio_request_duration_milliseconds_sum{reporter="source"}[5m])) by (source_canonical_service, destination_canonical_service) 
            / 
            sum(rate(istio_requests_total{reporter="source"}[5m])) by (source_canonical_service, destination_canonical_service))[1h:5m]
        )
        """

        print("Fetching metrics from Prometheus...")
        df_current = self.fetch_df(current_latency_q)
        df_baseline = self.fetch_df(baseline_avg_q)

        if df_current.empty or df_baseline.empty:
            return "No data found. Ensure Istio sidecars are emitting metrics."

        # Rename columns for merging
        df_current = df_current.rename(columns={'value': 'latency_now'})
        df_baseline = df_baseline.rename(columns={'value': 'latency_avg'})

        # Merge data on the caller-callee relationship
        merged = pd.merge(
            df_current, 
            df_baseline, 
            on=['source_canonical_service', 'destination_canonical_service']
        )

        # 3. Anomaly Detection (Simple Z-Score logic)
        # We flag anything 2x higher than the average as a basic anomaly
        merged['is_anomaly'] = merged['latency_now'] > (merged['latency_avg'] * 2)

        # 4. Build NetworkX Graph
        for _, row in merged.iterrows():
            src = row['source_canonical_service']
            dst = row['destination_canonical_service']
            
            self.graph.add_edge(
                src, dst, 
                latency=row['latency_now'],
                anomaly=row['is_anomaly']
            )

        return merged

# --- Execution Block ---
if __name__ == "__main__":
    # Update with your Prometheus service URL
    analyzer = IstioMeshAnalyzer("http://prometheus.istio-system:9090")
    results = analyzer.analyze()

    print("\n--- Service Dependency Report ---")
    print(results[['source_canonical_service', 'destination_canonical_service', 'latency_now', 'is_anomaly']])

    # Example Graph Analysis: Find services causing the most downstream pain
    if not analyzer.graph.edges:
        print("Graph is empty.")
    else:
        pagerank = nx.pagerank(analyzer.graph)
        critical_node = max(pagerank, key=pagerank.get)
        print(f"\nMost Central Service (Risk Factor): {critical_node}")

        anomalies = [e for e, attr in analyzer.graph.edges.items() if attr['anomaly']]
        print(f"Number of Anomalous Links Detected: {len(anomalies)}")
