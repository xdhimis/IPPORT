import pandas as pd
import numpy as np

def normalize_per_service(group, service_stats=None, min_std=1e-6):
    group_id = f"{group['ServerName'].iloc[0]}_{group['IP_port'].iloc[0]}"
    if service_stats is None:
        conn_std = max(group['connection_count'].std(), min_std)
        service_stats = {
            'conn_mean': group['connection_count'].mean(),
            'conn_std': conn_std
        }
        group['conn_zscore'] = (group['connection_count'] - service_stats['conn_mean']) / service_stats['conn_std']
    else:
        conn_std = max(service_stats['conn_std'], min_std)
        group['conn_zscore'] = (group['connection_count'] - service_stats['conn_mean']) / conn_std
    group['conn_zscore'] = group['conn_zscore'].fillna(0)
    group['conn_log'] = np.log1p(group['connection_count'])
    return group, service_stats

def preprocess_data(data, train_data_stats=None, is_training=False):
    if data.empty:
        return pd.DataFrame(), train_data_stats or {}
    
    data_groups = []
    if is_training:
        train_data_stats = {}
        for (server, ip_port), group in data.groupby(['ServerName', 'IP_port']):
            normalized_group, stats = normalize_per_service(group)
            data_groups.append(normalized_group)
            train_data_stats[f"{server}_{ip_port}"] = stats
    else:
        default_stats = {
            'conn_mean': data['connection_count'].mean(),
            'conn_std': max(data['connection_count'].std(), 1e-6)
        }
        for (server, ip_port), group in data.groupby(['ServerName', 'IP_port']):
            stats = train_data_stats.get(f"{server}_{ip_port}", default_stats)
            normalized_group, _ = normalize_per_service(group, stats)
            data_groups.append(normalized_group)
    
    data = pd.concat(data_groups).reset_index(drop=True)
    
    # Extract temporal features
    data['hour'] = data['timestamp'].dt.hour
    data['day_of_week'] = data['timestamp'].dt.dayofweek
    
    # Aggregate per service
    agg_data = data.groupby(['ServerName', 'IP_port']).agg({
        'conn_zscore': ['mean', 'max'],
        'conn_log': ['mean', 'median', 'max'],
        'hour': 'mean',
        'day_of_week': 'mean',
        'connection_count': ['mean', 'count']
    }).reset_index()
    
    # Flatten column names
    agg_data.columns = ['_'.join(col).strip() if col[1] else col[0] for col in agg_data.columns]
    
    # Filter low data volume (relaxed for real-time)
    min_count = 10 if is_training else 1
    agg_data = agg_data[agg_data['connection_count_count'] >= min_count]
    
    return agg_data, train_data_stats