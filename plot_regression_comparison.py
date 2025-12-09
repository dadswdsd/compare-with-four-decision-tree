import csv
import matplotlib.pyplot as plt
import os
import sys
import argparse

def find_latest_result(dataset_name):
    # Base directory for runs
    runs_dir = os.path.join("outputs", "runs")
    if not os.path.exists(runs_dir):
        return None
        
    # Get all subdirectories in runs/
    subdirs = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
    
    # Sort descending: newest timestamp first
    subdirs.sort(reverse=True)
    
    # Iterate to find the first one containing dataset/combined_results.csv
    for d in subdirs:
        candidate = os.path.join(runs_dir, d, dataset_name, "combined_results.csv")
        if os.path.exists(candidate):
            return candidate
            
    return None

def plot_comparison(dataset_name, csv_path):
    print(f"[{dataset_name}] Reading data from: {csv_path}")
    models = []
    rmses = []
    r2s = []
    maes = []
    
    try:
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                models.append(row['model'])
                rmses.append(float(row['rmse']))
                r2s.append(float(row['r2']))
                maes.append(float(row['mae']))
    except Exception as e:
        print(f"[{dataset_name}] Error reading CSV: {e}")
        return

    if not models:
        print(f"[{dataset_name}] No data found.")
        return

    # Setup plot: One grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(models))
    width = 0.25
    
    rects1 = ax.bar([i - width for i in x], rmses, width, label='RMSE', color='#d62728')
    rects2 = ax.bar(x, r2s, width, label='R²', color='#2ca02c')
    rects3 = ax.bar([i + width for i in x], maes, width, label='MAE', color='#1f77b4')
    
    ax.set_ylabel('Score')
    ax.set_title(f'Model Comparison: RMSE, R², MAE ({dataset_name.capitalize()} Dataset)')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    
    # Calculate limits to accommodate all bars properly
    all_values = rmses + r2s + maes
    min_val = min(all_values)
    max_val = max(all_values)
    
    # Add a bit of padding
    if min_val < 0:
        ax.set_ylim(min_val * 1.1, max_val * 1.1)
    else:
        ax.set_ylim(0, max_val * 1.1)

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Label bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, rotation=0)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    
    # Save output
    output_dir = os.path.dirname(csv_path)
    output_path = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(output_path, dpi=300)
    print(f"[{dataset_name}] Chart saved to {output_path}")
    plt.close(fig)

def process_dataset(dataset_name):
    csv_path = find_latest_result(dataset_name)
    if csv_path:
        plot_comparison(dataset_name, csv_path)
    else:
        print(f"[{dataset_name}] No results found in outputs/runs/")

if __name__ == "__main__":
    print("Starting regression comparison plotting...")
    process_dataset("boston")
    process_dataset("melb")
    print("Done.")
