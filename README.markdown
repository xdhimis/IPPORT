# Network Connection Anomaly Detection

This project implements a real-time anomaly detection system for network connections using InfluxDB 1.x (InfluxQL), Isolation Forest, and FastAPI. It monitors services identified by `ServerName` and `IP_port`, detecting anomalies based on `connection_count` metrics. The system trains on 3 hours of historical data, predicts anomalies every 30 seconds, and generates visualizations (scatter and bar plots). It includes a threshold-based anomaly classification (default: 0.7) and exposes a REST API for training, prediction, and result retrieval.

## Features
- **Data Source**: Queries InfluxDB 1.x for `network_connections` measurement (`ServerName`, `IP_port`, `connection_count`).
- **Preprocessing**: Normalizes data per service with z-scores, handling low-variance cases (`min_std=1e-6`).
- **Model**: Uses Isolation Forest with features like `conn_zscore`, `conn_log`, and temporal metrics (`hour`, `day_of_week`).
- **Threshold**: Labels services with anomaly scores > 0.7 as anomalous.
- **Real-Time**: Predicts every 30 seconds, with optional custom time ranges (RFC3339 format).
- **API**: FastAPI endpoints for training, prediction, results, and visualizations.
- **Visualizations**: Generates scatter (`connection_count_mean` vs. `hour_mean`) and bar plots (top 5 services).
- **Output**: Saves results to `worst_performing_services.csv` and plots to `outputs/visualizations/`.

## Project Structure
```
anomaly_detection/
├── data/
│   └── influx.py             # InfluxDB 1.x querying
├── preprocessing/
│   └── normalize.py          # Data normalization and feature engineering
├── model/
│   └── isolation_forest.py   # Isolation Forest training and prediction
├── visualization/
│   └── plots.py              # Scatter and bar plot generation
├── api/
│   └── main.py               # FastAPI app and endpoints
├── outputs/
│   ├── results/
│   │   └── worst_performing_services.csv
│   └── visualizations/
│       ├── scatter/
│       └── bar/
├── README.md
```

## Prerequisites
- **Python**: 3.6+
- **InfluxDB**: 1.x (e.g., 1.8) with a database containing `network_connections` measurement.
- **Dependencies**: Install via `requirements.txt` (see below).

## Setup
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd anomaly_detection
   ```

2. **Install Dependencies**:
   Create and activate a virtual environment, then install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install influxdb pandas numpy scikit-learn matplotlib seaborn joblib fastapi uvicorn
   ```
   Or use a `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure InfluxDB**:
   - Edit `data/influx.py` to update `INFLUX_CONFIG` with your InfluxDB credentials:
     ```python
     INFLUX_CONFIG = {
         "host": "localhost",
         "port": 8086,
         "database": "mydb",  # Your database name
         "user": "<your-username>",  # Optional
         "password": "<your-password>"  # Optional
     }
     ```
   - Ensure the `network_connections` measurement exists with:
     - **Field**: `connection_count` (float/int)
     - **Tags**: `ServerName`, `IP_port` (e.g., `192.168.1.1:8080`)
     - Example InfluxDB line protocol:
       ```
       network_connections,ServerName=svc_100,IP_port=192.168.1.1:8080 connection_count=150 1699488000000000000
       ```

4. **Enable InfluxDB Queries**:
   - In `api/main.py`, uncomment InfluxDB query lines:
     ```python
     # train_data = query_historical_data(client, start_time, end_time)
     # predict_data = query_realtime_data(client, start_time, end_time)
     ```
   - Comment out simulated data:
     ```python
     # train_data = simulate_data(n_records=360, services=services, start_time='2025-08-08 14:00:00' if not start_time else start_time)
     # predict_data = simulate_data(n_records=60, services=services, start_time=simulate_time if not start_time else start_time)
     ```

## Usage
1. **Run the FastAPI Server**:
   ```bash
   uvicorn api.main:app --reload
   ```
   - The server runs at `http://localhost:8000`.
   - Access interactive API docs at `http://localhost:8000/docs`.

2. **API Endpoints**:
   - **Train Model** (`GET /train`):
     - Trains the Isolation Forest model on historical data.
     - Parameters:
       - `start_time`: RFC3339 format (e.g., `2025-08-08T00:00:00Z`). Default: `now() - 3h`.
       - `end_time`: RFC3339 format (e.g., `2025-08-08T03:00:00Z`). Default: `now()`.
     - Example:
       ```
       http://localhost:8000/train?start_time=2025-08-08T00:00:00Z&end_time=2025-08-08T03:00:00Z
       ```
   - **Predict Anomalies** (`GET /predict`):
     - Runs a prediction cycle, returns top 5 anomalous services.
     - Parameters:
       - `start_time`: RFC3339 format (e.g., `2025-08-08T17:00:00Z`). Default: `now() - 30s`.
       - `end_time`: RFC3339 format (e.g., `2025-08-08T17:00:30Z`). Default: `now()`.
     - Example:
       ```
       http://localhost:8000/predict?start_time=2025-08-08T17:00:00Z&end_time=2025-08-08T17:00:30Z
       ```
   - **Get Results** (`GET /results`):
     - Retrieves recent prediction results.
     - Parameter: `limit` (e.g., `50`). Default: `100`.
     - Example:
       ```
       http://localhost:8000/results?limit=50
       ```
   - **Get Visualizations** (`GET /visualizations`):
     - Returns paths to the latest scatter and bar plots.
     - Example:
       ```
       http://localhost:8000/visualizations
       ```

3. **Real-Time Predictions**:
   - On startup, the server runs a background task to predict anomalies every 30 seconds using the default range (`now() - 30s`).
   - Results are appended to `outputs/results/worst_performing_services.csv`.
   - Visualizations are saved in `outputs/visualizations/scatter/` and `outputs/visualizations/bar/`.

## Sample Output
- **API Response** (`/predict`):
  ```json
  {
    "results": [
      {
        "timestamp": "2025-08-09_121500",
        "service_id": "svc_100_192.168.1.1:8080",
        "anomaly_score": 0.73,
        "connection_count_mean": 298.45,
        "is_anomaly": true
      },
      ...
    ],
    "scatter_plot": "outputs/visualizations/scatter/conn_vs_hour_scatter_20250809_121500.png",
    "bar_plot": "outputs/visualizations/bar/top_5_worst_bar_20250809_121500.png"
  }
  ```
- **Console Output**:
  ```
  Top 5 worst-performing services at 2025-08-09_121500:
              timestamp                service_id  anomaly_score  connection_count_mean  is_anomaly
  0  2025-08-09_121500    svc_100_192.168.1.1:8080       0.73                 298.45        True
  1  2025-08-09_121500  svc_new_1_192.168.1.2:8080       0.71                 280.78        True
  ...
  ```

## Notes
- **NaN Handling**:
  - Uses `min_std=1e-6` to prevent division by zero in z-score calculations.
  - Imputes `NaN` z-scores with 0 and handles `NaN` in model inputs.
  - To debug `NaN` issues, log problematic services:
    ```python
    nan_services = predict_data[predict_data['conn_zscore'].isna()]['service_id'].unique()
    print(f"Services with NaN z-scores: {nan_services}")
    ```
- **InfluxDB Optimization**:
  - For large datasets, aggregate in InfluxQL:
    ```sql
    SELECT MEAN(connection_count) AS connection_count, ServerName, IP_port
    FROM network_connections
    WHERE time >= now() - 30s
    GROUP BY ServerName, IP_port
    ```
- **Threshold Tuning**:
  - Adjust `ANOMALY_THRESHOLD` (default: 0.7) in `model/isolation_forest.py` to change anomaly sensitivity.
- **Deployment**:
  - For production, use `gunicorn`:
    ```bash
    pip install gunicorn
    gunicorn -w 4 -k uvicorn.workers.UvicornWorker api.main:app
    ```
- **Testing**:
  - Simulated data is used by default for testing. Uncomment InfluxDB queries in `api/main.py` for production.
- **Visualization**:
  - For web-based plots, integrate ChartJS (request a sample configuration if needed).

## Troubleshooting
- **Empty Data**: Ensure InfluxDB has data in the `network_connections` measurement.
- **Connection Errors**: Verify InfluxDB credentials and server status.
- **NaN Issues**: Check for missing `connection_count` values and add imputation if needed:
  ```python
  train_data.fillna({'connection_count': train_data['connection_count'].median()}, inplace=True)
  ```
- **API Issues**: Check logs in the terminal or use `http://localhost:8000/docs` for debugging.

## Future Improvements
- Add alerting for anomalies (e.g., email, Slack).
- Implement alternative algorithms (e.g., ADTK’s `SeasonalAD`).
- Optimize InfluxDB queries with aggregations for large datasets.
- Add ChartJS for web-based visualizations.

## License
MIT License. See `LICENSE` file (create one if needed).