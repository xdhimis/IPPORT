from fastapi import FastAPI, BackgroundTasks, Query
from pydantic import BaseModel
import pandas as pd
import os
from datetime import datetime, timedelta
import asyncio
from data.influx import get_influx_client, query_historical_data, query_realtime_data, simulate_data
from preprocessing.normalize import preprocess_data
from model.isolation_forest import train_model, predict_anomalies, MODEL_PATH, STATS_PATH
from visualization.plots import create_scatter_plot, create_bar_plot

app = FastAPI()

# Configuration
OUTPUT_CSV = 'outputs/results/worst_performing_services.csv'
n_services = 600
services = [f"svc_{i}" for i in range(n_services)] + [f"svc_new_{i}" for i in range(50)]

# Pydantic models
class PredictionResult(BaseModel):
    timestamp: str
    service_id: str
    anomaly_score: float
    connection_count_mean: float
    is_anomaly: bool

class PredictionResponse(BaseModel):
    results: list[PredictionResult]
    scatter_plot: str | None
    bar_plot: str | None

# Initialize CSV
os.makedirs('outputs/results', exist_ok=True)
if not os.path.exists(OUTPUT_CSV):
    pd.DataFrame(columns=['timestamp', 'service_id', 'anomaly_score', 'connection_count_mean', 'is_anomaly']).to_csv(OUTPUT_CSV, index=False)

# Prediction function
async def predict_once(start_time=None, end_time=None):
    client = get_influx_client()
    # Simulate real-time data for testing
    global simulate_time
    predict_data = simulate_data(n_records=60, services=services, start_time=simulate_time if not start_time else start_time)
    if not start_time:
        simulate_time += timedelta(seconds=30)
    
    # Uncomment for InfluxDB
    # predict_data = query_realtime_data(client, start_time, end_time)
    
    if predict_data.empty:
        print(f"No new data at {datetime.now().strftime('%Y-%m-%d_%H%M%S')}.")
        client.close()
        return {"results": [], "scatter_plot": None, "bar_plot": None}
    
    train_data_stats = load(STATS_PATH) if os.path.exists(STATS_PATH) else {}
    predict_agg_data, _ = preprocess_data(predict_data, train_data_stats, is_training=False)
    if predict_agg_data.empty:
        print(f"No valid data for prediction at {datetime.now().strftime('%Y-%m-%d_%H%M%S')}.")
        client.close()
        return {"results": [], "scatter_plot": None, "bar_plot": None}
    
    result_data = predict_anomalies(predict_agg_data, train_data_stats)
    if result_data.empty:
        print("Prediction failed.")
        client.close()
        return {"results": [], "scatter_plot": None, "bar_plot": None}
    
    top_5_worst = result_data[['service_id', 'anomaly_score', 'connection_count_mean', 'is_anomaly', 'hour_mean']].sort_values(by='anomaly_score', ascending=False).head(5)
    
    current_time = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    timestamp_str = current_time.replace(':', '')
    top_5_worst['timestamp'] = current_time
    
    print(f"\nTop 5 worst-performing services at {current_time}:")
    print(top_5_worst[['timestamp', 'service_id', 'anomaly_score', 'connection_count_mean', 'is_anomaly']].round(2))
    
    top_5_worst[['timestamp', 'service_id', 'anomaly_score', 'connection_count_mean', 'is_anomaly']].to_csv(OUTPUT_CSV, mode='a', header=False, index=False)
    
    scatter_plot = create_scatter_plot(result_data, top_5_worst, timestamp_str)
    bar_plot = create_bar_plot(top_5_worst, timestamp_str)
    
    client.close()
    
    return {
        "results": top_5_worst[['timestamp', 'service_id', 'anomaly_score', 'connection_count_mean', 'is_anomaly']].to_dict('records'),
        "scatter_plot": scatter_plot,
        "bar_plot": bar_plot
    }

# Training endpoint
@app.get("/train")
async def train_endpoint(start_time: str = Query(None, description="Start time in RFC3339 format"), end_time: str = Query(None, description="End time in RFC3339 format")):
    client = get_influx_client()
    # Simulate historical data for testing
    train_data = simulate_data(n_records=360, services=services, start_time='2025-08-08 14:00:00' if not start_time else start_time)
    
    # Uncomment for InfluxDB
    # train_data = query_historical_data(client, start_time, end_time)
    
    if train_data.empty:
        return {"message": "No historical data for training."}
    
    train_agg_data, train_data_stats = preprocess_data(train_data, is_training=True)
    if train_agg_data.empty:
        return {"message": "No valid data after preprocessing."}
    
    iso_forest, scaler = train_model(train_agg_data)
    if iso_forest and scaler:
        dump(train_data_stats, STATS_PATH)
        return {"message": "Model trained and saved."}
    
    client.close()
    return {"message": "Training failed."}

# Prediction endpoint
@app.get("/predict", response_model=PredictionResponse)
async def trigger_prediction(start_time: str = Query(None, description="Start time in RFC3339 format"), end_time: str = Query(None, description="End time in RFC3339 format")):
    return await predict_once(start_time, end_time)

# Results endpoint
@app.get("/results", response_model=list[PredictionResult])
async def get_results(limit: int = Query(100, description="Number of results to return")):
    if not os.path.exists(OUTPUT_CSV):
        return []
    df = pd.read_csv(OUTPUT_CSV)
    return df.tail(limit).to_dict('records')

# Visualizations endpoint
@app.get("/visualizations")
async def get_visualizations():
    scatter_dir = 'outputs/visualizations/scatter'
    bar_dir = 'outputs/visualizations/bar'
    latest_scatter = max([os.path.join(scatter_dir, f) for f in os.listdir(scatter_dir)], key=os.path.getctime, default=None)
    latest_bar = max([os.path.join(bar_dir, f) for f in os.listdir(bar_dir)], key=os.path.getctime, default=None)
    return {"latest_scatter": latest_scatter, "latest_bar": latest_bar}

# Background task for real-time predictions
async def run_realtime_predictions():
    global simulate_time
    simulate_time = pd.to_datetime('2025-08-08 17:00:00')
    while True:
        await predict_once()  # Uses default range
        await asyncio.sleep(30)

@app.on_event("startup")
async def startup_event():
    await train_endpoint()  # Train with default range on startup
    asyncio.create_task(run_realtime_predictions())  # Start real-time loop