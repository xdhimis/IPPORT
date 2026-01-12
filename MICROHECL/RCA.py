import requests
import json
import networkx as nx
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
import time
import argparse
from datetime import datetime, timedelta
from prometheus_api_client import PrometheusConnect, MetricRangeDataFrame

# Note: Using PrometheusConnect from prometheus_api_client as requested.
# Ensure prometheus_api_client is installed in your environment (pip install prometheus_api_client).

# Modular Functions

def parse_arguments():
    """
    Parse command-line arguments for timings, Prometheus URL, etc.
    Defaults: baseline last 1 hour from now, current last 10 minutes.
    Timestamps in Unix seconds.
    Entry service is now optional; if not provided, auto-detected.
    """
    parser = argparse.ArgumentParser(description="RCA Script for Istio Prometheus Metrics")
    parser.add_argument('--prom_url', type=str, default='http://prometheus:9090', help='Prometheus API URL')
    parser.add_argument('--entry_service', type=str, default=None, help='Initial anomalous service (workload name, optional)')
    parser.add_argument('--baseline_start', type=int, default=int(time.time() - 3600), help='Baseline start Unix timestamp')
    parser.add_argument('--baseline_end', type=int, default=int(time.time()), help='Baseline end Unix timestamp')
    parser.add_argument('--current_start', type=int, default=int(time.time() - 600), help='Current start Unix timestamp')
    parser.add_argument('--current_end', type=int, default=int(time.time()), help='Current end Unix timestamp')
    return parser.parse_args()

def query_prom(pc, query, start, end, step='60s'):
    """
    Query Prometheus range API using PrometheusConnect for time series.
    Returns list of results or empty if none.
    """
    start_time = datetime.fromtimestamp(start)
    end_time = datetime.fromtimestamp(end)
    data = pc.custom_query_range(
        query=query,
        start_time=start_time,
        end_time=end_time,
        step=step
    )
    return data

def discover_graph(pc, prom_url, baseline_start, baseline_end, current_start, current_end):
    """
    Step 1: Discover workloads and build directed graph.
    """
    # Union time range for discovery
    union_start = min(baseline_start, current_start)
    union_end = max(baseline_end, current_end)
    time_range = f'[{union_end - union_start}s]'

    # Discover unique workloads using HTTP since PrometheusConnect may not support label values directly
    workloads = set()
    for label in ['source_workload', 'destination_workload']:
        resp = requests.get(f'{prom_url}/api/v1/label/{label}/values')
        if resp.status_code == 200:
            workloads.update(resp.json().get('data', []))

    # Build graph
    G = nx.DiGraph()
    for src in list(workloads):
        for dst in list(workloads):
            if src == dst:
                continue
            q = f'sum(istio_requests_total{{source_workload="{src}", destination_workload="{dst}"}}{time_range}) > 0'
            data = query_prom(pc, q, union_start, union_end)
            if data:
                G.add_edge(src, dst)
    return G, workloads

def collect_metrics(pc, G, baseline_start, baseline_end, current_start, current_end):
    """
    Step 2: Collect metrics and compute features for each edge.
    Returns dicts for baseline and current features per type per edge.
    """
    step = '60s'  # 1 min step
    baseline_features = {'performance': {}, 'reliability': {}, 'traffic': {}}
    current_features = {'performance': {}, 'reliability': {}, 'traffic': {}}
    vectors = {'baseline': {}, 'current': {}}  # Store raw vectors for correlations

    for caller, callee in G.edges():
        edge_key = f"{caller}->{callee}"

        # Queries
        rt_query = f'histogram_quantile(0.9, sum(rate(istio_request_duration_seconds_bucket{{source_workload="{caller}", destination_workload="{callee}"}}[1m])) by (le))'
        error_query = f'sum(rate(istio_requests_total{{response_code=~"5..", source_workload="{caller}", destination_workload="{callee}"}}[1m])) / ignoring(response_code) sum(rate(istio_requests_total{{source_workload="{caller}", destination_workload="{callee}"}}[1m]))'
        qps_query = f'sum(rate(istio_requests_total{{source_workload="{caller}", destination_workload="{callee}"}}[1m]))'

        # Baseline vectors
        rt_baseline_data = query_prom(pc, rt_query, baseline_start, baseline_end, step)
        error_baseline_data = query_prom(pc, error_query, baseline_start, baseline_end, step)
        qps_baseline_data = query_prom(pc, qps_query, baseline_start, baseline_end, step)

        rt_baseline = MetricRangeDataFrame(rt_baseline_data)['value'].values if rt_baseline_data else np.array([])
        error_baseline = MetricRangeDataFrame(error_baseline_data)['value'].values if error_baseline_data else np.array([])
        qps_baseline = MetricRangeDataFrame(qps_baseline_data)['value'].values if qps_baseline_data else np.array([])

        # Current vectors
        rt_current_data = query_prom(pc, rt_query, current_start, current_end, step)
        error_current_data = query_prom(pc, error_query, current_start, current_end, step)
        qps_current_data = query_prom(pc, qps_query, current_start, current_end, step)

        rt_current = MetricRangeDataFrame(rt_current_data)['value'].values if rt_current_data else np.array([])
        error_current = MetricRangeDataFrame(error_current_data)['value'].values if error_current_data else np.array([])
        qps_current = MetricRangeDataFrame(qps_current_data)['value'].values if qps_current_data else np.array([])

        vectors['baseline'][edge_key] = {'rt': rt_baseline, 'error': error_baseline, 'qps': qps_baseline}
        vectors['current'][edge_key] = {'rt': rt_current, 'error': error_current, 'qps': qps_current}

        # Handle empty arrays
        if len(rt_baseline) == 0 or len(rt_current) == 0:
            continue

        # Performance (RT)
        baseline_mean, baseline_std, baseline_max = np.mean(rt_baseline), np.std(rt_baseline), np.max(rt_baseline)
        over_max_count = np.sum(rt_current > baseline_max)
        max_delta = np.max(rt_current) - baseline_max
        over_avg_count = np.sum(rt_current > baseline_mean)
        avg_ratio = np.mean(rt_current) / baseline_mean if baseline_mean != 0 else 0
        baseline_features['performance'][edge_key] = [over_max_count, max_delta, over_avg_count, avg_ratio]
        current_features['performance'][edge_key] = [over_max_count, max_delta, over_avg_count, avg_ratio]

        # Reliability (Error)
        if len(error_baseline) > 0 and len(error_current) > 0:
            baseline_deltas = np.diff(error_baseline)
            baseline_outliers = baseline_deltas[np.abs(baseline_deltas - np.mean(baseline_deltas)) > 3 * np.std(baseline_deltas)] if len(baseline_deltas) > 0 else np.array([])
            baseline_outlier_val = np.mean(baseline_outliers) if len(baseline_outliers) > 0 else 0

            current_deltas = np.diff(error_current)
            current_outliers = current_deltas[np.abs(current_deltas - np.mean(current_deltas)) > 3 * np.std(current_deltas)] if len(current_deltas) > 0 else np.array([])
            current_outlier_val = np.mean(current_outliers) if len(current_outliers) > 0 else 0

            rt_over_thresh = 1 if np.mean(rt_current) > 0.05 else 0  # 50ms = 0.05s
            max_error = np.max(error_current)
            if len(error_current) == len(rt_current) and len(error_current) > 1:
                corr_rt_error, _ = pearsonr(error_current, rt_current)
            else:
                corr_rt_error = 0

            baseline_features['reliability'][edge_key] = [baseline_outlier_val, 0, 0, 0, 0]  # Placeholder
            current_features['reliability'][edge_key] = [0, current_outlier_val, rt_over_thresh, max_error, corr_rt_error]

        # Traffic (QPS)
        if len(qps_baseline) > 0 and len(qps_current) > 0:
            baseline_mean, baseline_std = np.mean(qps_baseline), np.std(qps_baseline)
            mean_delta = np.mean(qps_current) - baseline_mean
            std_delta = np.std(qps_current) - baseline_std
            outlier_count = np.sum(np.abs(qps_current - baseline_mean) > 3 * baseline_std)

            # Cluster-wide QPS for proxy
            cluster_qps_query = 'sum(rate(istio_requests_total[1m]))'
            cluster_current_data = query_prom(pc, cluster_qps_query, current_start, current_end, step)
            cluster_qps_current = MetricRangeDataFrame(cluster_current_data)['value'].values if cluster_current_data else np.array([])

            if len(cluster_qps_current) == len(qps_current) and len(qps_current) > 1:
                corr_cluster, _ = pearsonr(qps_current, cluster_qps_current)
            else:
                corr_cluster = 0

            baseline_features['traffic'][edge_key] = [0, 0, 0]  # Placeholder
            current_features['traffic'][edge_key] = [mean_delta, std_delta, outlier_count]

            vectors['current'][edge_key]['corr_cluster'] = corr_cluster  # Store for later use

    return baseline_features, current_features, vectors

def detect_anomalies(baseline_features, current_features, vectors):
    """
    Step 3: Detect anomalies per edge per type.
    Returns dict of anomalous edges with types.
    """
    anomalies = {}
    for typ in ['performance', 'reliability', 'traffic']:
        if typ in baseline_features and baseline_features[typ]:
            for edge_key in baseline_features[typ]:
                base_feats = np.array([baseline_features[typ].get(edge_key, [])])
                curr_feats = np.array([current_features[typ].get(edge_key, [])])
                if base_feats.size == 0 or curr_feats.size == 0:
                    continue

                if typ == 'performance':
                    model = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
                    model.fit(base_feats)
                    if model.predict(curr_feats) == [-1]:
                        anomalies.setdefault(edge_key, []).append(typ)
                elif typ == 'reliability':
                    model = IsolationForest(contamination=0.1)
                    model.fit(base_feats)
                    if model.predict(curr_feats) == [-1]:
                        anomalies.setdefault(edge_key, []).append(typ)
                elif typ == 'traffic':
                    # 3-sigma + corr
                    outlier_count = current_features[typ][edge_key][2]
                    corr_cluster = vectors['current'][edge_key].get('corr_cluster', 0)
                    if outlier_count > 0 and corr_cluster > 0.9:
                        anomalies.setdefault(edge_key, []).append(typ)
                    elif outlier_count > 0:  # Fallback stricter
                        if np.sum(np.abs(vectors['current'][edge_key]['qps'] - np.mean(vectors['baseline'][edge_key]['qps'])) > 4 * np.std(vectors['baseline'][edge_key]['qps']) ) > 0:
                            anomalies.setdefault(edge_key, []).append(typ)
    return anomalies

def auto_select_entry(G, anomalies, pc, current_start, current_end):
    """
    Auto-select entry service if not provided.
    Pick the node with highest incoming QPS or most anomalies.
    """
    if not anomalies:
        # Fallback to highest QPS destination
        q = 'topk(1, sum(rate(istio_requests_total[10m])) by (destination_workload))'
        data = query_prom(pc, q, current_start, current_end)
        if data:
            df = MetricRangeDataFrame(data)
            return df['destination_workload'].iloc[-1] if 'destination_workload' in df.columns else None
        return None

    # Count anomalies per node (as callee)
    node_anoms = {}
    for edge_key, typs in anomalies.items():
        callee = edge_key.split('->')[1]
        node_anoms[callee] = node_anoms.get(callee, 0) + len(typs)

    if node_anoms:
        return max(node_anoms, key=node_anoms.get)

    # If no anomalies, pick root node (no predecessors)
    roots = [n for n in G.nodes if G.in_degree(n) == 0]
    return roots[0] if roots else None

def perform_rca(G, entry_service, anomalies, vectors):
    """
    Step 4: RCA - Entry analysis and chain extension with pruning.
    Returns list of candidate root causes.
    """
    chains = []
    directions = {'performance': 'downstream', 'reliability': 'downstream', 'traffic': 'upstream'}  # backtrack dir: downstream means extend to successors (callees)

    # Entry Node Analysis
    if entry_service not in G.nodes:
        raise ValueError(f"Entry service {entry_service} not in graph")
    for typ in ['performance', 'reliability', 'traffic']:
        backtrack_dir = directions[typ]
        # For downstream propagation types, check callees (successors); for upstream, check callers (predecessors)
        neighbors = G.successors(entry_service) if backtrack_dir == 'downstream' else G.predecessors(entry_service)
        for neigh in neighbors:
            edge_key = f"{entry_service}->{neigh}" if backtrack_dir == 'downstream' else f"{neigh}->{entry_service}"
            if edge_key in anomalies and typ in anomalies[edge_key]:
                chains.append({'type': typ, 'path': [entry_service, neigh]})

    # Chain Extension with Pruning
    for chain in chains[:]:  # Copy to avoid modification during iteration
        typ = chain['type']
        current_end = chain['path'][-1]
        extend_func = G.successors if directions[typ] == 'downstream' else G.predecessors
        while True:
            added = False
            candidates = list(extend_func(current_end))
            for cand in candidates:
                edge_key = f"{current_end}->{cand}" if directions[typ] == 'downstream' else f"{cand}->{current_end}"
                if edge_key in anomalies and typ in anomalies[edge_key]:
                    # Prune by correlation
                    prev_edge_key = f"{chain['path'][-2]}->{current_end}" if directions[typ] == 'downstream' else f"{current_end}->{chain['path'][-2]}"
                    metric = 'rt' if typ == 'performance' else 'error' if typ == 'reliability' else 'qps'
                    vec1 = vectors['current'].get(prev_edge_key, {}).get(metric, np.array([]))
                    vec2 = vectors['current'].get(edge_key, {}).get(metric, np.array([]))
                    if len(vec1) > 0 and len(vec2) > 0 and len(vec1) == len(vec2):
                        corr, _ = pearsonr(vec1, vec2)
                    else:
                        corr = 0  # Default no prune if cannot compute
                    if corr >= 0.7:
                        chain['path'].append(cand)
                        current_end = cand
                        added = True
            if not added:
                break

    # Step 5: Candidates
    candidates = []
    for chain in chains:
        root = chain['path'][-1]
        candidates.append({'root': root, 'type': chain['type'], 'chain': ' -> '.join(chain['path'])})

    # Rank by chain length (longer = deeper root)
    candidates.sort(key=lambda x: len(x['chain'].split(' -> ')), reverse=True)
    return candidates

def main():
    args = parse_arguments()
    pc = PrometheusConnect(url=args.prom_url, disable_ssl=True)  # Assume no SSL; adjust as needed
    G, workloads = discover_graph(pc, args.prom_url, args.baseline_start, args.baseline_end, args.current_start, args.current_end)
    baseline_features, current_features, vectors = collect_metrics(pc, G, args.baseline_start, args.baseline_end, args.current_start, args.current_end)
    anomalies = detect_anomalies(baseline_features, current_features, vectors)

    if args.entry_service is None:
        args.entry_service = auto_select_entry(G, anomalies, pc, args.current_start, args.current_end)
        if args.entry_service is None:
            raise ValueError("Could not auto-detect entry service. Please provide --entry_service.")
        print(f"Auto-detected entry service: {args.entry_service}")

    candidates = perform_rca(G, args.entry_service, anomalies, vectors)

    print("Candidate Root Causes:")
    for cand in candidates:
        print(f"Root: {cand['root']} ({cand['type']}), Chain: {cand['chain']}")

if __name__ == "__main__":
    main()
