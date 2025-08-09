import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_scatter_plot(data, top_5_worst, timestamp_str):
    os.makedirs('outputs/visualizations/scatter', exist_ok=True)
    plt.figure(figsize=(10, 6))
    valid_data = data.dropna(subset=['connection_count_mean', 'hour_mean', 'anomaly_score'])
    if not valid_data.empty:
        scatter = plt.scatter(
            valid_data['connection_count_mean'], valid_data['hour_mean'],
            c=valid_data['anomaly_score'], cmap='RdBu_r',
            alpha=0.6, s=100
        )
        plt.colorbar(scatter, label='Anomaly Score (higher = worse)')
        plt.xlabel('Mean Connection Count')
        plt.ylabel('Mean Hour of Day')
        plt.title(f'Service Performance: Connection Count vs Hour ({timestamp_str})')
        
        # Annotate top 5 with anomaly status
        for idx, row in top_5_worst.iterrows():
            if not pd.isna(row['connection_count_mean']) and not pd.isna(row['hour_mean']):
                label = f"{row['service_id']} ({'Anomaly' if row['is_anomaly'] else 'Normal'})"
                plt.annotate(
                    label, (row['connection_count_mean'], row['hour_mean']),
                    textcoords="offset points", xytext=(5,5), ha='center', fontsize=8
                )
        plt.grid(True)
        plt.savefig(f'outputs/visualizations/scatter/conn_vs_hour_scatter_{timestamp_str}.png')
        plt.close()
        return f'outputs/visualizations/scatter/conn_vs_hour_scatter_{timestamp_str}.png'
    return None

def create_bar_plot(top_5_worst, timestamp_str):
    os.makedirs('outputs/visualizations/bar', exist_ok=True)
    plt.figure(figsize=(8, 5))
    sns.barplot(x='service_id', y='anomaly_score', hue='is_anomaly', dodge=False, data=top_5_worst)
    plt.xlabel('Service ID (ServerName_IP:port)')
    plt.ylabel('Anomaly Score (higher = worse)')
    plt.title(f'Top 5 Worst-Performing Services ({timestamp_str})')
    plt.xticks(rotation=45)
    plt.legend(title='Anomaly Status')
    plt.tight_layout()
    plt.savefig(f'outputs/visualizations/bar/top_5_worst_bar_{timestamp_str}.png')
    plt.close()
    return f'outputs/visualizations/bar/top_5_worst_bar_{timestamp_str}.png'