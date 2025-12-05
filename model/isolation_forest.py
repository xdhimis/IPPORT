import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import grangercausalitytests
import joblib

ANOMALY_THRESHOLD = 0.7
FEATURES = ['connection_count', 'rolling_mean', 'diff', 'rate_change', 'over_max_count', 'over_avg_count', 'delta_max', 'delta_avg', 'ratio_avg', 'conn_log', 'hour', 'day_of_week']
Z_FEATS = [f'z_{f}' for f in FEATURES if f != 'hour' and f != 'day_of_week'] + ['hour', 'day_of_week']  # hour/day not z-normalized

def train_model(df_normal, model_path='model.joblib'):
    model = IsolationForest(contamination=0.01, random_state=42)
    model.fit(df_normal[Z_FEATS])
    joblib.dump(model, model_path)
    return model

def predict_anomalies(df_problem, model):
    df_problem['anomaly_score'] = -model.decision_function(df_problem[Z_FEATS])  # Higher >0.5 anomalous (sklearn convention)
    df_problem['is_anomaly'] = df_problem['anomaly_score'] > ANOMALY_THRESHOLD
    return df_problem

def flag_3sigma(df_problem, means, stds, group_key=['IP_port', 'ServerName']):
    def flag(row):
        key = (row['IP_port'], row['ServerName'])
        m = means[key]
        s = stds[key]
        delta_outlier = abs(row['diff']) > 3 * df_problem.groupby(group_key)['diff'].std()[key].clip(lower=MIN_STD)
        return (abs(row['connection_count'] - m) > 3 * s) or delta_outlier
    df_problem['3sigma'] = df_problem.apply(flag, axis=1)
    return df_problem

def infer_dependency_graph(df_normal, corr_threshold=0.7):
    group_key = ['IP_port', 'ServerName']
    df_normal['service_id'] = df_normal['ServerName'] + '_' + df_normal['IP_port']
    pivot = df_normal.pivot(index='timestamp', columns='service_id', values='connection_count').fillna(0)
    corr = pivot.corr(method='pearson')
    graph = {}
    for col1 in corr.columns:
        graph[col1] = [col2 for col2 in corr.columns if col1 != col2 and abs(corr.loc[col1, col2]) > corr_threshold]
    # Directions via Granger (causality)
    directions = {}
    for col1 in pivot.columns:
        for col2 in graph.get(col1, []):
            try:
                res = grangercausalitytests(pivot[[col1, col2]], maxlag=5, verbose=False)
                p_vals = [res[lag+1][0]['ssr_ftest'][1] for lag in range(5)]
                directions[(col1, col2)] = any(p < 0.05 for p in p_vals)
            except:
                directions[(col1, col2)] = False
    return graph, directions, corr

def compute_corr_diff(df_problem, corr_normal):
    df_problem['service_id'] = df_problem['ServerName'] + '_' + df_problem['IP_port']
    pivot_problem = df_problem.pivot(index='timestamp', columns='service_id', values='connection_count').fillna(0)
    corr_problem = pivot_problem.corr(method='pearson')
    return corr_problem - corr_normal

def localize_root_causes(df_problem, graph, directions, overall_metric='connection_count'):
    group_key = ['IP_port', 'ServerName']
    df_problem['service_id'] = df_problem['ServerName'] + '_' + df_problem['IP_port']
    # Anomaly/3sigma rates
    rates = df_problem.groupby('service_id').agg({'is_anomaly': 'mean', '3sigma': 'mean'}).rename(columns={'is_anomaly': 'anomaly_rate', '3sigma': 'sigma_rate'})
    # Propagation chains
    high_anom = rates['anomaly_rate'].sort_values(ascending=False).index[:5]
    chains = {}
    for start in high_anom:
        chain = [start]
        current = start
        visited = set()
        while current not in visited:
            visited.add(current)
            neighbors = [n for n in graph.get(current, []) if directions.get((current, n), False)]
            if not neighbors:
                break
            next_n = max(neighbors, key=lambda n: rates['anomaly_rate'][n])
            chain.append(next_n)
            current = next_n
        chains[start] = chain
    # Ranking (corr to overall system + chain length weight)
    overall = df_problem.groupby('timestamp')[overall_metric].sum()
    ranks = {}
    for sid in df_problem['service_id'].unique():
        group_ts = df_problem[df_problem['service_id'] == sid]['connection_count']
        if len(group_ts) == len(overall):
            corr, _ = pearsonr(group_ts, overall)
            ranks[sid] = abs(corr) * (len(chains.get(sid, [])) or 1)
    ranks = pd.Series(ranks).sort_values(ascending=False)
    # Merge to df_problem or output CSV
    worst = pd.concat([rates, ranks.rename('rank_score')], axis=1).sort_values('rank_score', ascending=False)
    worst['chain'] = worst.index.map(chains.get)
    worst.to_csv('outputs/results/worst_performing_services.csv')
    return worst