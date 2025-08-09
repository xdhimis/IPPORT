import pandas as pd
from influxdb import InfluxDBClient
from datetime import datetime

# InfluxDB configuration
INFLUX_CONFIG = {
    "host": "localhost",
    "port": 8086,
    "database": "mydb",  # Replace with your database name
    "user": "<your-username>",  # Optional
    "password": "<your-password>"  # Optional
}

def get_influx_client():
    return InfluxDBClient(
        host=INFLUX_CONFIG["host"],
        port=INFLUX_CONFIG["port"],
        username=INFLUX_CONFIG["user"],
        password=INFLUX_CONFIG["password"],
        database=INFLUX_CONFIG["database"]
    )

def query_historical_data(client, start_time=None, end_time=None):
    if start_time and end_time:
        query = (
            "SELECT time, ServerName, IP_port, connection_count "
            "FROM network_connections "
            "WHERE time >= '" + start_time + "' AND time <= '" + end_time + "'"
        )
    else:
        query = (
            "SELECT time, ServerName, IP_port, connection_count "
            "FROM network_connections "
            "WHERE time >= now() - 3h"
        )
    result = client.query(query)
    points = list(result.get_points())
    if not points:
        return pd.DataFrame()
    df = pd.DataFrame(points)
    df['timestamp'] = pd.to_datetime(df['time'])
    df = df.drop(columns=['time'])
    return df

def query_realtime_data(client, start_time=None, end_time=None):
    if start_time and end_time:
        query = (
            "SELECT time, ServerName, IP_port, connection_count "
            "FROM network_connections "
            "WHERE time >= '" + start_time + "' AND time <= '" + end_time + "'"
        )
    else:
        query = (
            "SELECT time, ServerName, IP_port, connection_count "
            "FROM network_connections "
            "WHERE time >= now() - 30s"
        )
    result = client.query(query)
    points = list(result.get_points())
    if not points:
        return pd.DataFrame()
    df = pd.DataFrame(points)
    df['timestamp'] = pd.to_datetime(df['time'])
    df = df.drop(columns=['time'])
    return df

# Simulated data for testing
def simulate_data(n_records, services, start_time):
    import numpy as np
    np.random.seed(42)
    data = pd.DataFrame({
        'timestamp': pd.date_range(start=start_time, periods=n_records, freq='30S'),
        'ServerName': np.random.choice(services, size=n_records),
        'IP_port': np.random.choice([f"192.168.1.{i}:8080" for i in range(1, 11)], size=n_records),
        'connection_count': np.random.lognormal(mean=4, sigma=1, size=n_records).clip(1, 1000)
    })
    anomaly_services = ['svc_100', 'svc_new_1']
    anomaly_indices = data['ServerName'].isin(anomaly_services)
    data.loc[anomaly_indices, 'connection_count'] *= 2
    data.loc[data['ServerName'] == 'svc_400', 'connection_count'] = 100
    return data