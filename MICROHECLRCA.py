import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM  # Updated from IsolationForest
from scipy.stats import skew, kurtosis, pearsonr
from statsmodels.tsa.stattools import grangercausalitytests

# Sample data generation (replace with your real data loading)
# Assume df_normal and df_problem are DataFrames with columns: 'timestamp', 'service_name', 'response_time', 'qps', 'eps'
np.random.seed(42)
timestamps = pd.date_range('2023-01-01', periods=100, freq='30s')
services = ['serviceA', 'serviceB', 'serviceC']
df_normal = pd.DataFrame({
    'timestamp': np.tile(timestamps, len(services)),
    'service_name': np.repeat(services, len(timestamps)),
    'response_time': np.random.normal(100, 20, len(timestamps) * len(services)),  # ms
    'qps': np.random.normal(50, 10, len(timestamps) * len(services)),
    'eps': np.random.normal(1, 0.5, len(timestamps) * len(services))
})
df_normal.loc[df_normal['eps'] < 0, 'eps'] = 0  # Errors can't be negative

df_problem = df_normal.copy()
mask = df_problem['service_name'] == 'serviceA'
df_problem.loc[mask, 'response_time'] += np.random.normal(200, 50, sum(mask))
df_problem.loc[mask, 'eps'] += np.random.normal(5, 2, sum(mask))

# Composite group key (service_name)
group_key = 'service_name'

# Metrics to process (paper focuses on response_time, but include others)
metrics = ['response_time', 'qps', 'eps']

# Compute mean/std from normal data for each metric
means_dict = {}
stds_dict = {}
for metric in metrics:
    groups = df_normal.groupby(group_key)
    means_dict[metric] = groups[metric].mean()
    stds_dict[metric] = groups[metric].std()

# Z-score function for each metric
def calc_z(df, metric, means, stds):
    def z_func(x):
        return (x - means[x.name]) / stds[x.name]
    df[f'z_{metric}'] = df.groupby(group_key)[metric].transform(z_func)
    return df

for metric in metrics:
    df_normal = calc_z(df_normal.copy(), metric, means_dict[metric], stds_dict[metric])
    df_problem = calc_z(df_problem.copy(), metric, means_dict[metric], stds_dict[metric])

# Add temporal and paper-inspired features for each metric (aligned with paper's 12 features; approximated here)
def add_features(df, window=10):  # Paper uses 10-min window
    df = df.sort_values([group_key, 'timestamp'])
    for metric in metrics:
        # Temporal features
        df[f'{metric}_rolling_mean'] = df.groupby(group_key)[metric].rolling(5).mean().reset_index(0, drop=True)
        df[f'{metric}_diff'] = df.groupby(group_key)[metric].diff()
        df[f'{metric}_rate_change'] = df[f'{metric}_diff'] / df[metric].shift(1)
        
        # MicroHECL-inspired features (e.g., over-max/avg counts, deltas, ratios; paper has 12 similar)
        rolling_max = df.groupby(group_key)[metric].rolling(window).max().reset_index(0, drop=True)
        rolling_mean = df.groupby(group_key)[metric].rolling(window).mean().reset_index(0, drop=True)
        df[f'{metric}_over_max_count'] = (df[metric] > rolling_max.shift(1)).groupby(df.groupby(group_key).cumcount() // window).transform('sum')
        df[f'{metric}_over_avg_count'] = (df[metric] > rolling_mean.shift(1)).groupby(df.groupby(group_key).cumcount() // window).transform('sum')
        df[f'{metric}_delta_max'] = df.groupby(group_key)[metric].transform(lambda x: x.rolling(window).max().diff(window))
        df[f'{metric}_delta_avg'] = df.groupby(group_key)[metric].transform(lambda x: x.rolling(window).mean().diff(window))
        df[f'{metric}_ratio_avg'] = df[metric] / rolling_mean.shift(1)
    return df.fillna(0)

df_normal = add_features(df_normal)
df_problem = add_features(df_problem)

# All features per metric (paper extracts 12 from RT; here ~8 per metric)
features_per_metric = ['rolling_mean', 'diff', 'rate_change', 'over_max_count', 'over_avg_count', 'delta_max', 'delta_avg', 'ratio_avg']
all_features = [f'{metric}_{feat}' for metric in metrics for feat in features_per_metric] + [f'z_{metric}' for metric in metrics]

# Z-normalize additional features
feat_means_dict = {}
feat_stds_dict = {}
for feat in all_features:
    groups = df_normal.groupby(group_key)
    feat_means_dict[feat] = groups[feat].mean()
    feat_stds_dict[feat] = groups[feat].std()

for feat in all_features:
    def z_func(x, feat):
        return (x - feat_means_dict[feat][x.name]) / feat_stds_dict[feat][x.name]
    df_normal[f'z_{feat}'] = df_normal.groupby(group_key)[feat].transform(lambda x: z_func(x, feat))
    df_problem[f'z_{feat}'] = df_problem.groupby(group_key)[feat].transform(lambda x: z_func(x, feat))

# One-Class SVM (updated to match paper; trained on normal data, nu=contamination rate)
z_feats = [f'z_{feat}' for feat in all_features]
model_ocsvm = OneClassSVM(nu=0.01, kernel='rbf', gamma='auto')  # Paper uses OC-SVM; adjust params as needed
model_ocsvm.fit(df_normal[z_feats])
df_problem['anomaly_multi'] = model_ocsvm.predict(df_problem[z_feats])
df_problem['anomaly_multi'] = df_problem['anomaly_multi'].map({1: 0, -1: 1})  # 1=anomaly (outlier)
df_problem['score_multi'] = model_ocsvm.decision_function(df_problem[z_feats])  # Lower = more anomalous

# 3-sigma flags per metric (supplemental, not in paper)
def flag_3sigma(row, metric):
    m = means_dict[metric][row[group_key]]
    s = stds_dict[metric][row[group_key]]
    delta_outlier = abs(row[f'{metric}_diff']) > 3 * stds_dict[metric][row[group_key]]
    return int((abs(row[metric] - m) > 3 * s) or delta_outlier)

for metric in metrics:
    df_problem[f'3sigma_{metric}'] = df_problem.apply(lambda row: flag_3sigma(row, metric), axis=1)
df_problem['3sigma'] = df_problem[[f'3sigma_{m}' for m in metrics]].max(axis=1)

# Infer dependency graph from normal correlations (paper uses Pearson for pruning; use QPS or RT)
pivot_normal = df_normal.pivot(index='timestamp', columns=group_key, values='response_time').fillna(0)  # Use RT as in paper
corr_normal = pivot_normal.corr(method='pearson')
graph = {}
corr_threshold = 0.7  # Paper uses similar for pruning weak edges
for col1 in corr_normal.columns:
    graph[col1] = [col2 for col2 in corr_normal.columns if col1 != col2 and abs(corr_normal.loc[col1, col2]) > corr_threshold]

# Directionality via Granger causality (paper-inspired for propagation)
directions = {}
for col1 in pivot_normal.columns:
    for col2 in graph.get(col1, []):
        try:
            res = grangercausalitytests(pivot_normal[[col1, col2]], maxlag=5, verbose=False)
            p_vals = [res[lag+1][0]['ssr_ftest'][1] for lag in range(5)]
            directions[(col1, col2)] = any(p < 0.05 for p in p_vals)
        except:
            directions[(col1, col2)] = False

# Anomaly propagation chains (as in paper)
high_anom_groups = df_problem.groupby(group_key)['anomaly_multi'].mean().sort_values(ascending=False).index[:5]
chains = {}
for start in high_anom_groups:
    chain = [start]
    current = start
    visited = set()
    while current not in visited:
        visited.add(current)
        neighbors = [n for n in graph.get(current, []) if directions.get((current, n), False)]
        if not neighbors:
            break
        next_n = max(neighbors, key=lambda n: df_problem.groupby(group_key)['anomaly_multi'].mean()[n])
        chain.append(next_n)
        current = next_n
    chains[start] = chain

# Ranking root causes (corr to overall system metric, e.g., total EPS; paper ranks by propagation impact)
overall_problem = df_problem.groupby('timestamp')['eps'].sum()  # System-wide errors
ranks = {}
for g in df_problem[group_key].unique():
    group_ts = df_problem[df_problem[group_key] == g]['eps']  # Correlate errors
    if len(group_ts) == len(overall_problem):
        corr, _ = pearsonr(group_ts, overall_problem)
        ranks[g] = abs(corr) * (len(chains.get(g, [])) or 1)
problematic_services = pd.Series(ranks).sort_values(ascending=False)

# Correlation differences (for detecting broken dependencies)
pivot_problem = df_problem.pivot(index='timestamp', columns=group_key, values='response_time').fillna(0)
corr_problem = pivot_problem.corr(method='pearson')
diff_corr = corr_problem - corr_normal

# Other stats per service in problematic data
stats_problem = df_problem.groupby(group_key)[metrics].agg(['mean', 'std', skew, kurtosis])

# Output results (in sample, 'serviceA' is anomalous)
print("Root Cause Ranking:")
print(problematic_services)
print("\nPropagation Chains:")
print(chains)
print("\nCorrelation Differences:")
print(diff_corr)
print("\nProblematic Stats:")
print(stats_problem)
