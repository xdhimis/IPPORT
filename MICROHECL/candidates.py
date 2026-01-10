import requests
import json
import networkx as nx
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest  # Fallback for RF if no labels
import time
import argparse
from datetime import datetime, timedelta

# Modular Functions

def parse_arguments():
    """
    Parse command-line arguments for timings, Prometheus URL, etc.
    Defaults: baseline last 1 hour, current last 10 minutes.
    Timestamps in Unix seconds.
    """
    parser = argparse.ArgumentParser(description="RCA Script for Istio Prometheus Metrics")
    parser.add_argument('--prom_url', type=str, default='http://prometheus:9090', help='Prometheus API URL')
    parser.add_argument('--entry_service', type=str, required=True, help='Initial anomalous service (workload name)')
    parser.add_argument('--baseline_start', type=int, default=int(time.time() - 3600), help='Baseline start Unix timestamp')
    parser.add_argument('--baseline_end', type=int, default=int(time.time()), help='Baseline end Unix timestamp')
    parser.add_argument('--current_start', type=int, default=int(time.time() - 600), help='Current start Unix timestamp')
    parser.add_argument('--current_end', type=int, default=int(time.time()), help='Current end Unix timestamp')
    return parser.parse_args()

def query_prom(url, query, start, end, step='60'):
    """
    Query Prometheus range API for time series.
    Returns list of results.
    """
    params = {'query': query, 'start': start, 'end': end, 'step': step}
    resp = requests.get(f'{url}/api/v1/query_range', params=params)
    if resp.status_code != 200:
        raise ValueError(f"Prometheus query failed: {resp.text}")
    data = resp.json()['data']['result']
    return data

def discover_graph(prom_url, baseline_start, baseline_end, current_start, current_end):
    """
    Step 1: Discover workloads and build directed graph.
    """
    # Union time range for discovery
    union_start = min(baseline_start, current_start)
    union_end = max(baseline_end, current_end)
    time_range = f'[{union_end - union_start}s]'

    # Discover unique workloads
    workloads = set()
    for label in ['source_workload', 'destination_workload']:
        resp = requests.get(f'{prom_url}/api/v1/label/{label}/values')
        if resp.status_code == 200:
            workloads.update(resp.json()['data'])

    # Build graph
    G = nx.DiGraph()
    for src in workloads:
        for dst in workloads:
            if src == dst:
                continue
            q = f'sum(istio_requests_total{{source_workload="{src}", destination_workload="{dst}"}}{time_range}) > 0'
            data = query_prom(prom_url, q, union_start, union_end)
            if data:
                G.add_edge(src, dst)
    return G, workloads

def collect_metrics(prom_url, G, baseline_start, baseline_end, current_start, current_end):
    """
    Step 2: Collect metrics and compute features for each edge.
    Returns dicts for baseline and current features per type per edge.
    """
    step = '60'  # 1 min step
    baseline_features = {'performance': {}, 'reliability': {}, 'traffic': {}}
    current_features = {'performance': {}, 'reliability': {}, 'traffic': {}}
    vectors = {'baseline': {}, 'current': {}}  # Store raw vectors for correlations

    for caller, callee in G.edges():
        edge_key = f"{caller}->{callee}"

        # Queries
        rt_query = f'histogram_quantile(0.9, sum(rate(istio_request_duration_seconds_bucket{{source_workload="{caller}", destination_workload="{callee}"}}[1m])) by (le))'
        error_query = f'sum(rate(istio_requests_total{{response_code=~"5..", source_workload="{caller}", destination_workload="{callee}"}}[1m])) / sum(rate(istio_requests_total{{source_workload="{caller}", destination_workload="{callee}"}}[1m]))'
        qps_query = f'sum(rate(istio_requests_total{{source_workload="{caller}", destination_workload="{callee}"}}[1m]))'

        # Baseline vectors
        rt_baseline = [float(v[1]) for v in query_prom(prom_url, rt_query, baseline_start, baseline_end, step)[0]['values']] if query_prom(prom_url, rt_query, baseline_start, baseline_end, step) else []
        error_baseline = [float(v[1]) for v in query_prom(prom_url, error_query, baseline_start, baseline_end, step)[0]['values']] if query_prom(prom_url, error_query, baseline_start, baseline_end, step) else []
        qps_baseline = [float(v[1]) for v in query_prom(prom_url, qps_query, baseline_start, baseline_end, step)[0]['values']] if query_prom(prom_url, qps_query, baseline_start, baseline_end, step) else []

        # Current vectors
        rt_current = [float(v[1]) for v in query_prom(prom_url, rt_query, current_start, current_end, step)[0]['values']] if query_prom(prom_url, rt_query, current_start, current_end, step) else []
        error_current = [float(v[1]) for v in query_prom(prom_url, error_query, current_start, current_end, step)[0]['values']] if query_prom(prom_url, error_query, current_start, current_end, step) else []
        qps_current = [float(v[1]) for v in query_prom(prom_url, qps_query, current_start, current_end, step)[0]['values']] if query_prom(prom_url, qps_query, current_start, current_end, step) else []

        vectors['baseline'][edge_key] = {'rt': np.array(rt_baseline), 'error': np.array(error_baseline), 'qps': np.array(qps_baseline)}
        vectors['current'][edge_key] = {'rt': np.array(rt_current), 'error': np.array(error_current), 'qps': np.array(qps_current)}

        # Compute features
        # Performance (RT)
        if len(rt_baseline) > 0 and len(rt_current) > 0:
            baseline_mean, baseline_std, baseline_max = np.mean(rt_baseline), np.std(rt_baseline), np.max(rt_baseline)
            over_max_count = np.sum(rt_current > baseline_max)
            max_delta = np.max(rt_current) - baseline_max
            over_avg_count = np.sum(rt_current > baseline_mean)
            avg_ratio = np.mean(rt_current) / baseline_mean
            baseline_features['performance'][edge_key] = [over_max_count, max_delta, over_avg_count, avg_ratio]
            current_features['performance'][edge_key] = [over_max_count, max_delta, over_avg_count, avg_ratio]

        # Reliability (Error)
        if len(error_baseline) > 0 and len(error_current) > 0:
            baseline_deltas = np.diff(error_baseline)
            baseline_outliers = baseline_deltas[np.abs(baseline_deltas - np.mean(baseline_deltas)) > 3 * np.std(baseline_deltas)]
            baseline_outlier_val = np.mean(baseline_outliers) if len(baseline_outliers) > 0 else 0

            current_deltas = np.diff(error_current)
            current_outliers = current_deltas[np.abs(current_deltas - np.mean(current_deltas)) > 3 * np.std(current_deltas)]
            current_outlier_val = np.mean(current_outliers) if len(current_outliers) > 0 else 0

            rt_over_thresh = 1 if np.mean(rt_current) > 0.05 else 0  # 50ms = 0.05s
            max_error = np.max(error_current)
            corr_rt_error, _ = pearsonr(error_current, rt_current) if len(error_current) == len(rt_current) > 1 else (0, 0)

            baseline_features['reliability'][edge_key] = [baseline_outlier_val, 0, 0, 0, 0]  # Placeholder for training
            current_features['reliability'][edge_key] = [0, current_outlier_val, rt_over_thresh, max_error, corr_rt_error]

        # Traffic (QPS)
        if len(qps_baseline) > 0 and len(qps_current) > 0:
            baseline_mean, baseline_std = np.mean(qps_baseline), np.std(qps_baseline)
            mean_delta = np.mean(qps_current) - baseline_mean
            std_delta = np.std(qps_current) - baseline_std
            outlier_count = np.sum(np.abs(qps_current - baseline_mean) > 3 * baseline_std)

            # Cluster-wide QPS for proxy
            cluster_qps_query = 'sum(rate(istio_requests_total[1m]))'
            cluster_qps_current = [float(v[1]) for v in query_prom(prom_url, cluster_qps_query, current_start, current_end, step)[0]['values']] if query_prom(prom_url, cluster_qps_query, current_start, current_end, step) else []
            corr_cluster, _ = pearsonr(qps_current, cluster_qps_current) if len(cluster_qps_current) == len(qps_current) > 1 else (0, 0)

            baseline_features['traffic'][edge_key] = [0, 0, 0]  # Placeholder
            current_features['traffic'][edge_key] = [mean_delta, std_delta, outlier_count]

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
                base_feats = np.array([baseline_features[typ][edge_key]])  # 2D for sklearn
                curr_feats = np.array([current_features[typ][edge_key]])

                if typ == 'performance':
                    model = OneClassSVM().fit(base_feats)
                    if model.predict(curr_feats) == -1:
                        anomalies.setdefault(edge_key, []).append(typ)
                elif typ == 'reliability':
                    model = IsolationForest().fit(base_feats)
                    if model.predict(curr_feats) == -1:
                        anomalies.setdefault(edge_key, []).append(typ)
                elif typ == 'traffic':
                    # 3-sigma + corr
                    outlier_count = current_features[typ][edge_key][2]
                    qps_current = vectors['current'][edge_key]['qps']
                    cluster_qps_query = 'sum(rate(istio_requests_total[1m]))'
                    # Assuming we have cluster vector from collect, but for simplicity, assume corr=0 if not
                    corr_cluster = 0  # Placeholder; compute as in collect
                    if outlier_count > 0 and corr_cluster > 0.9:
                        anomalies.setdefault(edge_key, []).append(typ)

    return anomalies

def perform_rca(G, entry_service, anomalies, vectors):
    """
    Step 4: RCA - Entry analysis and chain extension with pruning.
    Returns list of candidate root causes.
    """
    chains = []
    directions = {'performance': 'downstream', 'reliability': 'downstream', 'traffic': 'upstream'}  # backtrack dir

    # Entry Node Analysis
    if entry_service not in G.nodes:
        raise ValueError("Entry service not in graph")
    for typ in ['performance', 'reliability', 'traffic']:
        dir_check = directions[typ]
        neighbors = list(G.successors(entry_service)) if dir_check == 'downstream' else list(G.predecessors(entry_service))
        for neigh in neighbors:
            edge_key = f"{entry_service}->{neigh}" if dir_check == 'downstream' else f"{neigh}->{entry_service}"
            if edge_key in anomalies and typ in anomalies[edge_key]:
                chains.append({'type': typ, 'path': [entry_service, neigh]})

    # Chain Extension with Pruning
    for chain in chains:
        typ = chain['type']
        current_end = chain['path'][-1]
        extend_dir = 'successors' if directions[typ] == 'downstream' else 'predecessors'
        while True:
            added = False
            candidates = list(getattr(G, extend_dir)(current_end))
            for cand in candidates:
                edge_key = f"{current_end}->{cand}" if extend_dir == 'successors' else f"{cand}->{current_end}"
                if edge_key in anomalies and typ in anomalies[edge_key]:
                    # Prune by correlation
                    prev_edge_key = f"{chain['path'][-2]}->{current_end}" if extend_dir == 'successors' else f"{current_end}->{chain['path'][-2]}"
                    metric = 'rt' if typ == 'performance' else 'error' if typ == 'reliability' else 'qps'
                    vec1 = vectors['current'].get(prev_edge_key, {}).get(metric, np.array([]))
                    vec2 = vectors['current'].get(edge_key, {}).get(metric, np.array([]))
                    if len(vec1) == len(vec2) > 1:
                        corr, _ = pearsonr(vec1, vec2)
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

    # Rank by arbitrary severity (e.g., chain length)
    candidates.sort(key=lambda x: len(x['chain'].split('->')), reverse=True)
    return candidates

def main():
    args = parse_arguments()
    G, workloads = discover_graph(args.prom_url, args.baseline_start, args.baseline_end, args.current_start, args.current_end)
    baseline_features, current_features, vectors = collect_metrics(args.prom_url, G, args.baseline_start, args.baseline_end, args.current_start, args.current_end)
    anomalies = detect_anomalies(baseline_features, current_features, vectors)
    candidates = perform_rca(G, args.entry_service, anomalies, vectors)

    print("Candidate Root Causes:")
    for cand in candidates:
        print(f"Root: {cand['root']} ({cand['type']}), Chain: {cand['chain']}")

if __name__ == "__main__":
    main()
