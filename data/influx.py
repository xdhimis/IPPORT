from influxdb import InfluxDBClient

def query_influx(host, port, username, password, dbname, time_window='30s'):
    client = InfluxDBClient(host=host, port=port, username=username, password=password, database=dbname)
    query = f"""
    SELECT MEAN("connection_count") AS "connection_count",
           "ServerName",
           "IP_port"
    FROM "network_connections"
    WHERE time >= now() - {time_window}
    GROUP BY "ServerName", "IP_port", time(30s)
    """
    result = client.query(query)
    df = pd.DataFrame(result.get_points())
    df['timestamp'] = pd.to_datetime(df['time'])
    return df