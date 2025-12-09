import csv
import matplotlib.pyplot as plt
import os
import sys
import argparse
import glob

def find_latest_bank_result():
    # Base directory for runs
    runs_dir = os.path.join("outputs", "runs")
    if not os.path.exists(runs_dir):
        return None
        
    # Get all subdirectories in runs/
    # Filter for directories that match timestamp pattern (roughly) to be safe, 
    # but strictly just sorting by name (YYYYMMDD_HHMMSS) works for finding newest.
    subdirs = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
    
    # Sort descending: newest timestamp first
    subdirs.sort(reverse=True)
    
    # Iterate to find the first one containing bank/combined_results.csv
    for d in subdirs:
        candidate = os.path.join(runs_dir, d, "bank", "combined_results.csv")
        if os.path.exists(candidate):
            return candidate
            
    return None

def plot_comparison(csv_path, output_path):
    print(f"Reading data from: {csv_path}")
    # Read data using csv module
    models = []
    f1_scores = []
    roc_aucs = []
    
    try:
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                models.append(row['model'])
                f1_scores.append(float(row['f1']))
                roc_aucs.append(float(row['roc_auc']))
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    if not models:
        print("No data found in CSV.")
        return

    # Setup plot
    x = range(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar([i - width/2 for i in x], f1_scores, width, label='F1-Score', color='#1f77b4')
    rects2 = ax.bar([i + width/2 for i in x], roc_aucs, width, label='ROC AUC', color='#ff7f0e')
    
    # Add labels and title
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison: F1-Score vs ROC AUC (Bank Dataset)')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0.8, 0.95)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add value labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Chart saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate bar chart for Bank dataset model comparison.")
    parser.add_argument("--csv_path", help="Path to the combined_results.csv file. If not provided, finds the latest run.")
    parser.add_argument("--output_path", help="Path to save the output image. Defaults to model_comparison.png in the same directory as the CSV.")
    
    args = parser.parse_args()
    
    csv_file = args.csv_path
    
    if not csv_file:
        print("No CSV path provided, searching for latest run...")
        csv_file = find_latest_bank_result()
        if not csv_file:
            # Fallback to the specific example path if no new runs found
            default_example = r"outputs/runs/20251125_175602/bank/combined_results.csv"
            if os.path.exists(default_example):
                print(f"No recent runs found, using example file: {default_example}")
                csv_file = default_example
            else:
                print("Error: Could not find any 'combined_results.csv' in outputs/runs/*/bank/. Please provide a path using --csv_path.")
                sys.exit(1)
    
    if args.output_path:
        output_file = args.output_path
    else:
        output_file = os.path.join(os.path.dirname(csv_file), "model_comparison.png")
        
    if not os.path.exists(csv_file):
        print(f"Error: Could not find {csv_file}")
        sys.exit(1)
    else:
        plot_comparison(csv_file, output_file)
