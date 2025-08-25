import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt  # For optional visualization

# Step 1: Generate Synthetic Data (Replace with real vmstat data loading)
def generate_synthetic_data(num_servers=100, timesteps=1000, metrics=['cpu_us', 'cpu_sy', 'mem_free', 'io_bi']):
    np.random.seed(42)
    server_types = ['DB'] * 40 + ['Web'] * 30 + ['App'] * 30
    data = []
    for i in range(num_servers):
        server_type = server_types[i]
        ts = np.arange(timesteps)
        if server_type == 'DB':  # High I/O variation
            cpu_us = 20 + 10 * np.sin(ts / 10) + np.random.normal(0, 5, timesteps)
            cpu_sy = 10 + 5 * np.sin(ts / 15) + np.random.normal(0, 3, timesteps)
            mem_free = 500 + 100 * np.cos(ts / 20) + np.random.normal(0, 50, timesteps)
            io_bi = 1000 + 500 * np.sin(ts / 5) + np.random.normal(0, 200, timesteps)  # High variation
        elif server_type == 'Web':  # High CPU
            cpu_us = 50 + 20 * np.sin(ts / 10) + np.random.normal(0, 10, timesteps)
            cpu_sy = 20 + 10 * np.sin(ts / 15) + np.random.normal(0, 5, timesteps)
            mem_free = 300 + 50 * np.cos(ts / 20) + np.random.normal(0, 30, timesteps)
            io_bi = 200 + 100 * np.sin(ts / 5) + np.random.normal(0, 50, timesteps)
        else:  # App: Balanced
            cpu_us = 30 + 15 * np.sin(ts / 10) + np.random.normal(0, 7, timesteps)
            cpu_sy = 15 + 7 * np.sin(ts / 15) + np.random.normal(0, 4, timesteps)
            mem_free = 400 + 80 * np.cos(ts / 20) + np.random.normal(0, 40, timesteps)
            io_bi = 500 + 200 * np.sin(ts / 5) + np.random.normal(0, 100, timesteps)
        
        # Clip to realistic ranges
        cpu_us = np.clip(cpu_us, 0, 100)
        cpu_sy = np.clip(cpu_sy, 0, 100)
        mem_free = np.clip(mem_free, 0, 1000)
        io_bi = np.clip(io_bi, 0, 2000)
        
        df = pd.DataFrame({
            'server_id': i,
            'timestamp': pd.date_range(start='2025-01-01', periods=timesteps, freq='30S'),
            'type': server_type,
            'cpu_us': cpu_us,
            'cpu_sy': cpu_sy,
            'mem_free': mem_free,
            'io_bi': io_bi
        })
        data.append(df)
    
    full_data = pd.concat(data)
    
    # Introduce anomalies in prediction timeframe (last 30% timesteps)
    prediction_start = int(timesteps * 0.7)
    for i in range(num_servers):
        anomaly_indices = np.random.choice(range(prediction_start, timesteps), size=20, replace=False)
        for idx in anomaly_indices:
            # Spike random metric
            metric = np.random.choice(metrics)
            full_data.loc[(full_data['server_id'] == i) & (full_data.index == idx + i*timesteps), metric] *= np.random.uniform(1.5, 3.0)
    
    return full_data

# Load or generate data
data = generate_synthetic_data()

# Step 2: Preprocess Data
def preprocess_data(group_data, seq_length=10):
    scalers = {}
    scaled_data = group_data.copy()
    metrics = ['cpu_us', 'cpu_sy', 'mem_free', 'io_bi']
    for metric in metrics:
        scalers[metric] = MinMaxScaler()
        scaled_data[metric] = scalers[metric].fit_transform(group_data[[metric]])
    
    # Create sliding windows for sequences
    sequences = []
    for i in range(len(scaled_data) - seq_length):
        seq = scaled_data[metrics].iloc[i:i+seq_length].values
        sequences.append(seq)
    return np.array(sequences), scalers

# Step 3: Define LSTM Autoencoder
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, input_dim, num_layers, batch_first=True)
    
    def forward(self, x):
        _, (h, _) = self.encoder(x)
        # Repeat hidden state for decoding
        h = h.repeat(x.size(1), 1, 1).permute(1, 0, 2)
        output, _ = self.decoder(h)
        return output

# Dataset for PyTorch
class TimeSeriesDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]

# Step 4: Train and Detect Anomalies per Group
def train_and_detect(group_name, group_data, seq_length=10, epochs=50, batch_size=32, hidden_dim=32):
    # Split timeframes
    baseline_end = int(len(group_data) * 0.7)
    baseline_data = group_data.iloc[:baseline_end]
    prediction_data = group_data.iloc[baseline_end:]
    
    # Preprocess
    baseline_seqs, scalers = preprocess_data(baseline_data, seq_length)
    prediction_seqs, _ = preprocess_data(prediction_data, seq_length)  # Use same scalers
    
    # Dataloaders
    train_dataset = TimeSeriesDataset(baseline_seqs)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Model
    input_dim = baseline_seqs.shape[2]  # Number of metrics
    model = LSTMAutoencoder(input_dim, hidden_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train on baseline
    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print(f'Group {group_name} - Epoch {epoch}, Loss: {loss.item():.4f}')
    
    # Compute reconstruction errors on baseline
    model.eval()
    with torch.no_grad():
        baseline_recon = model(torch.tensor(baseline_seqs, dtype=torch.float32))
        baseline_errors = np.mean((baseline_seqs - baseline_recon.numpy()) ** 2, axis=(1,2))
    threshold = np.mean(baseline_errors) + 3 * np.std(baseline_errors)
    
    # Detect on prediction
    with torch.no_grad():
        pred_recon = model(torch.tensor(prediction_seqs, dtype=torch.float32))
        pred_errors = np.mean((prediction_seqs - pred_recon.numpy()) ** 2, axis=(1,2))
    anomalies = pred_errors > threshold
    anomaly_count = np.sum(anomalies)
    
    print(f'Group {group_name}: Detected {anomaly_count} anomalies in prediction timeframe.')
    
    # Optional: Visualize errors
    # plt.plot(pred_errors)
    # plt.axhline(threshold, color='r')
    # plt.title(f'{group_name} Prediction Errors')
    # plt.show()
    
    return anomalies, pred_errors, threshold

# Step 5: Run for Each Group
groups = data.groupby('type')
results = {}
for group_name, group_data in groups:
    print(f'Processing group: {group_name}')
    results[group_name] = train_and_detect(group_name, group_data)

# Output: results dict has anomalies per group