import json
import pandas as pd
import networkx as nx
from networkx.readwrite import json_graph
from prometheus_api_client import PrometheusConnect

class IstioMeshAnalyzer:
    def __init__(self, prom_url):
        self.prom = PrometheusConnect(url=prom_url, disable_ssl=True)
        self.graph = nx.DiGraph()

    def fetch_df(self, query):
        """Helper to fetch PromQL and return a cleaned DataFrame."""
        result = self.prom.custom_query(query=query)
        rows = []
        for item in result:
            row = item['metric']
            row['value'] = float(item['value'][1])
            rows.append(row)
        return pd.DataFrame(rows)

    def analyze(self):
        # 1. PromQL Queries using Canonical Labels
        current_latency_q = """
        histogram_quantile(0.95, 
            sum(rate(istio_request_duration_milliseconds_bucket{reporter="source"}[5m])) 
            by (le, source_canonical_service, destination_canonical_service)
        )
        """
        baseline_avg_q = """
        avg_over_time(
            (sum(rate(istio_request_duration_milliseconds_sum{reporter="source"}[5m])) by (source_canonical_service, destination_canonical_service) 
            / 
            sum(rate(istio_requests_total{reporter="source"}[5m])) by (source_canonical_service, destination_canonical_service))[1h:5m]
        )
        """

        print("Analyzing Mesh Telemetry...")
        df_now = self.fetch_df(current_latency_q)
        df_base = self.fetch_df(baseline_avg_q)

        if df_now.empty or df_base.empty:
            return pd.DataFrame()

        # Merge and detect anomalies (> 2x historical average)
        df = pd.merge(df_now.rename(columns={'value': 'latency_now'}), 
                      df_base.rename(columns={'value': 'latency_avg'}), 
                      on=['source_canonical_service', 'destination_canonical_service'])
        
        df['is_anomaly'] = df['latency_now'] > (df['latency_avg'] * 2)

        # 2. Update NetworkX Graph with metadata
        for _, row in df.iterrows():
            self.graph.add_edge(
                row['source_canonical_service'], 
                row['destination_canonical_service'], 
                latency=row['latency_now'],
                anomaly=row['is_anomaly']
            )
        return df

    def export_viz_json(self, filename="mesh_graph.json", format="d3"):
        """Exports the graph to a JSON format for web frontends."""
        if format == "d3":
            # Node-link format for D3 Force Layouts
            data = json_graph.node_link_data(self.graph)
        else:
            # Format specifically for Cytoscape.js
            data = nx.cytoscape_data(self.graph)
            
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Graph exported to {filename} in {format} format.")

# --- Run ---
if __name__ == "__main__":
    analyzer = IstioMeshAnalyzer("http://prometheus.istio-system:9090")
    results = analyzer.analyze()

    if not results.empty:
        # Export for frontend visualization
        analyzer.export_viz_json(format="d3")
        
        # Quick CLI Summary
        critical_path = nx.dag_longest_path(analyzer.graph) if nx.is_directed_acyclic_graph(analyzer.graph) else "Cycle detected"
        print(f"Logical Critical Path: {' -> '.join(critical_path)}")
