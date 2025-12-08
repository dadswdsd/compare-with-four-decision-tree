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
    
    try:
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                models.append(row['model'])
                rmses.append(float(row['rmse']))
                r2s.append(float(row['r2']))
    except Exception as e:
        print(f"[{dataset_name}] Error reading CSV: {e}")
        return

    if not models:
        print(f"[{dataset_name}] No data found.")
        return

    # Setup plot with 2 subplots (R2 and RMSE)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    x = range(len(models))
    
    # Plot R2 (Higher is better)
    bars1 = ax1.bar(x, r2s, color='#2ca02c')
    ax1.set_title(f'{dataset_name.capitalize()} - R² Score (Higher is Better)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.set_ylim(min(r2s) * 0.9 if min(r2s) > 0 else 0, max(r2s) * 1.05)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Label R2
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom')

    # Plot RMSE (Lower is better)
    bars2 = ax2.bar(x, rmses, color='#d62728')
    ax2.set_title(f'{dataset_name.capitalize()} - RMSE (Lower is Better)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.set_ylim(0, max(rmses) * 1.1)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Label RMSE
    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom')

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
