import io
import json
import joblib
import pandas as pd
from google.cloud import storage

class GCSStorageManager:
    def __init__(self, bucket_name):
        """
        Initialize the GCS client and bucket.
        Assume Google Cloud authentication is set up (e.g., via service account key or gcloud CLI).
        """
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        print(f"Connected to GCS bucket: {bucket_name}")

    # Method to save model to GCS (using joblib for scikit-learn or similar models)
    def save_model(self, model, gcs_path):
        buffer = io.BytesIO()
        joblib.dump(model, buffer)
        buffer.seek(0)
        blob = self.bucket.blob(gcs_path)
        blob.upload_from_file(buffer, content_type='application/octet-stream')
        print(f"Model saved to {gcs_path}")

    # Method to load model from GCS
    def load_model(self, gcs_path):
        blob = self.bucket.blob(gcs_path)
        buffer = io.BytesIO()
        blob.download_to_file(buffer)
        buffer.seek(0)
        model = joblib.load(buffer)
        print(f"Model loaded from {gcs_path}")
        return model

    # Method to save scaler to GCS (assuming scikit-learn StandardScaler or similar)
    def save_scaler(self, scaler, gcs_path):
        buffer = io.BytesIO()
        joblib.dump(scaler, buffer)
        buffer.seek(0)
        blob = self.bucket.blob(gcs_path)
        blob.upload_from_file(buffer, content_type='application/octet-stream')
        print(f"Scaler saved to {gcs_path}")

    # Method to load scaler from GCS
    def load_scaler(self, gcs_path):
        blob = self.bucket.blob(gcs_path)
        buffer = io.BytesIO()
        blob.download_to_file(buffer)
        buffer.seek(0)
        scaler = joblib.load(buffer)
        print(f"Scaler loaded from {gcs_path}")
        return scaler

    # Method to save stats to GCS (assuming stats is a dict; adjust if it's another format)
    def save_stats(self, stats, gcs_path):
        stats_str = json.dumps(stats)
        blob = self.bucket.blob(gcs_path)
        blob.upload_from_string(stats_str, content_type='application/json')
        print(f"Stats saved to {gcs_path}")

    # Method to load stats from GCS
    def load_stats(self, gcs_path):
        blob = self.bucket.blob(gcs_path)
        stats_str = blob.download_as_string().decode('utf-8')
        stats = json.loads(stats_str)
        print(f"Stats loaded from {gcs_path}")
        return stats

    # Method to save Pandas DataFrame as CSV to GCS
    def save_dataframe_as_csv(self, df, gcs_path):
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        blob = self.bucket.blob(gcs_path)
        blob.upload_from_string(buffer.getvalue(), content_type='text/csv')
        print(f"DataFrame saved as CSV to {gcs_path}")

    # Method to load CSV from GCS into a Pandas DataFrame
    def load_csv_as_dataframe(self, gcs_path):
        blob = self.bucket.blob(gcs_path)
        csv_str = blob.download_as_string().decode('utf-8')
        df = pd.read_csv(io.StringIO(csv_str))
        print(f"CSV loaded as DataFrame from {gcs_path}")
        return df

# Example usage:
# manager = GCSStorageManager('your-bucket-name')

# During training:
# manager.save_model(your_trained_model, 'models/model.pkl')
# manager.save_scaler(your_fitted_scaler, 'scalers/scaler.pkl')
# manager.save_stats({'accuracy': 0.95, 'loss': 0.05}, 'stats/training_stats.json')
# manager.save_dataframe_as_csv(pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]}), 'data/training_data.csv')

# During prediction:
# model = manager.load_model('models/model.pkl')
# scaler = manager.load_scaler('scalers/scaler.pkl')
# stats = manager.load_stats('stats/training_stats.json')
# df = manager.load_csv_as_dataframe('data/training_data.csv')