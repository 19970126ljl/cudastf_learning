
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

def plot_reduction_performance(csv_file):
    """
    Reads reduction performance data from a CSV file and generates plots.

    Args:
        csv_file (str): The path to the CSV file containing the benchmark results.
    """
    try:
        # Read the CSV data into a pandas DataFrame
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: The file '{csv_file}' was not found.")
        print("Please run the benchmark program first to generate the results file.")
        sys.exit(1)

    # --- Plot 1: Execution Time vs. Problem Size ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    sns.barplot(data=df, x='N', y='Time_ms', hue='Method', ax=ax)

    ax.set_title('Reduction Execution Time vs. Problem Size', fontsize=16)
    ax.set_xlabel('Problem Size (N)', fontsize=12)
    ax.set_ylabel('Average Execution Time (ms)', fontsize=12)
    ax.set_yscale('log')  # Use a logarithmic scale for better visualization
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title='Method')
    
    # Add labels to the bars
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}", 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 9), 
                    textcoords='offset points')

    plt.tight_layout()
    plt.savefig('reduction_times.png')
    print("Generated plot: reduction_times.png")

    # --- Plot 2: Bandwidth vs. Problem Size ---
    fig, ax = plt.subplots(figsize=(12, 8))

    sns.barplot(data=df, x='N', y='GB_s', hue='Method', ax=ax)

    ax.set_title('Reduction Bandwidth vs. Problem Size', fontsize=16)
    ax.set_xlabel('Problem Size (N)', fontsize=12)
    ax.set_ylabel('Bandwidth (GB/s)', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title='Method')

    # Add labels to the bars
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}", 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 9), 
                    textcoords='offset points')

    plt.tight_layout()
    plt.savefig('reduction_bandwidth.png')
    print("Generated plot: reduction_bandwidth.png")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        csv_file_path = sys.argv[1]
    else:
        csv_file_path = 'reduction_benchmark_results.csv'
    
    plot_reduction_performance(csv_file_path)
