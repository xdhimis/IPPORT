import matplotlib.pyplot as plt
import seaborn as sns

def generate_scatter(df, output_dir='outputs/visualizations/scatter/'):
    # Existing scatter (e.g., conn vs time per service)
    for sid in df['service_id'].unique():
        subset = df[df['service_id'] == sid]
        plt.figure()
        plt.scatter(subset['timestamp'], subset['connection_count'], c=subset['is_anomaly'])
        plt.savefig(f'{output_dir}{sid}_scatter.png')
        plt.close()

def generate_bar(df_worst, output_dir='outputs/visualizations/bar/'):
    # Existing bar (anomaly scores)
    plt.figure()
    sns.barplot(x=df_worst.index, y='anomaly_rate', data=df_worst)
    plt.savefig(f'{output_dir}anomaly_rates.png')
    plt.close()
    # New: Rank scores
    plt.figure()
    sns.barplot(x=df_worst.index, y='rank_score', data=df_worst)
    plt.savefig(f'{output_dir}rank_scores.png')
    plt.close()
    # Corr diff heatmap
    if 'diff_corr' in globals():  # Assume from model
        plt.figure()
        sns.heatmap(diff_corr, annot=True)
        plt.savefig(f'{output_dir}corr_diff_heatmap.png')
        plt.close()