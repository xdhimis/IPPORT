from fastapi import FastAPI
import uvicorn
from preprocessing.normalize import normalize_data, add_features, z_normalize_features, compute_stats
from model.isolation_forest import train_model, predict_anomalies, flag_3sigma, infer_dependency_graph, compute_corr_diff, localize_root_causes
from data.influx import query_influx
from visualization.plots import generate_scatter, generate_bar

app = FastAPI()

# Config (from env or file)
INFLUX_CONFIG = {'host': 'localhost', 'port': 8086, 'username': 'user', 'password': 'pass', 'dbname': 'net_db'}

@app.post("/train")
def train():
    df_normal = query_influx(**INFLUX_CONFIG, time_window='7d')  # Historical normal
    df_normal = add_features(df_normal)
    df_normal, means, stds = normalize_data(df_normal)
    features = ['connection_count', 'rolling_mean', 'diff', 'rate_change', 'over_max_count', 'over_avg_count', 'delta_max', 'delta_avg', 'ratio_avg', 'conn_log']
    df_normal, means_dict, stds_dict = z_normalize_features(df_normal, features)
    model = train_model(df_normal)
    graph, directions, corr_normal = infer_dependency_graph(df_normal)
    # Save means/stds/graph etc. via joblib
    return {"status": "trained"}

@app.post("/predict")
def predict():
    df_problem = query_influx(**INFLUX_CONFIG, time_window='30s')
    df_problem = add_features(df_problem)
    df_problem, _, _ = normalize_data(df_problem, means, stds, is_training=False)  # Load saved means/stds
    df_problem, _, _ = z_normalize_features(df_problem, features, means_dict, stds_dict, is_training=False)
    df_problem = predict_anomalies(df_problem, model)  # Load saved model
    df_problem = flag_3sigma(df_problem, means, stds)
    diff_corr = compute_corr_diff(df_problem, corr_normal)  # Load saved
    worst = localize_root_causes(df_problem, graph, directions)  # Enhanced localization
    generate_scatter(df_problem)
    generate_bar(worst)
    return {"status": "predicted", "anomalies": worst.to_dict()}

@app.get("/results")
def get_results():
    worst = pd.read_csv('outputs/results/worst_performing_services.csv')
    return worst.to_dict()

@app.get("/stats")
def get_stats():
    df_problem = query_influx(**INFLUX_CONFIG, time_window='30s')  # Recent
    stats = compute_stats(df_problem)
    return stats.to_dict()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)