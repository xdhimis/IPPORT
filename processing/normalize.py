import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

MIN_STD = 1e-6

def normalize_data(df, means=None, stds=None, is_training=True):
    group_key = ['IP_port', 'ServerName']
    if is_training:
        groups = df.groupby(group_key)
        means = groups['connection_count'].mean()
        stds = groups['connection_count'].std().clip(lower=MIN_STD)
    def z_func(x):
        key = tuple(x.name) if isinstance(x.name, tuple) else x.name
        return (x - means[key]) / stds[key]
    df['conn_zscore'] = df.groupby(group_key)['connection_count'].transform(z_func)
    df['conn_zscore'] = df['conn_zscore'].fillna(0)  # Impute NaNs
    df['conn_log'] = np.log1p(df['connection_count'])  # Existing log transform
    return df, means, stds

def add_features(df, window=10):  # window ≈5 min for 30s intervals
    group_key = ['IP_port', 'ServerName']
    df = df.sort_values(group_key + ['timestamp'])
    # Temporal features (from discussion)
    df['rolling_mean'] = df.groupby(group_key)['connection_count'].rolling(5).mean().reset_index(0, drop=True)
    df['diff'] = df.groupby(group_key)['connection_count'].diff()
    df['rate_change'] = df['diff'] / df['connection_count'].shift(1)
    # Paper-inspired (MicroHECL adaptations)
    rolling_max = df.groupby(group_key)['connection_count'].rolling(window).max().reset_index(0, drop=True)
    rolling_mean = df.groupby(group_key)['connection_count'].rolling(window).mean().reset_index(0, drop=True)
    df['over_max_count'] = (df['connection_count'] > rolling_max.shift(1)).groupby(df.groupby(group_key).cumcount() // window).transform('sum')
    df['over_avg_count'] = (df['connection_count'] > rolling_mean.shift(1)).groupby(df.groupby(group_key).cumcount() // window).transform('sum')
    df['delta_max'] = df.groupby(group_key)['connection_count'].transform(lambda x: x.rolling(window).max().diff(window))
    df['delta_avg'] = df.groupby(group_key)['connection_count'].transform(lambda x: x.rolling(window).mean().diff(window))
    df['ratio_avg'] = df['connection_count'] / rolling_mean.shift(1)
    # Existing temporal
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
    df = df.fillna(0)
    return df

def z_normalize_features(df, features, means_dict=None, stds_dict=None, is_training=True):
    group_key = ['IP_port', 'ServerName']
    if is_training:
        means_dict = {}
        stds_dict = {}
        for feat in features:
            groups = df.groupby(group_key)
            means_dict[feat] = groups[feat].mean()
            stds_dict[feat] = groups[feat].std().clip(lower=MIN_STD)
    for feat in features:
        def z_func(x, feat):
            key = tuple(x.name) if isinstance(x.name, tuple) else x.name
            return (x - means_dict[feat][key]) / stds_dict[feat][key]
        df[f'z_{feat}'] = df.groupby(group_key)[feat].transform(lambda x: z_func(x, feat))
    return df, means_dict, stds_dict

def compute_stats(df):
    group_key = ['IP_port', 'ServerName']
    stats = df.groupby(group_key)['connection_count'].agg(['mean', 'std', skew, kurtosis])
    return stats