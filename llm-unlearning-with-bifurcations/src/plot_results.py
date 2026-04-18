import os
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def load_metrics(run_dir):
    data = {}
    base_path = Path(run_dir)
    
    # Load baselines
    baseline_dir = base_path / "baseline"
    data["baseline"] = {}
    if baseline_dir.exists():
        for bucket in baseline_dir.iterdir():
            if bucket.is_dir() and (bucket / "retain_reference").exists():
                eval_file = bucket / "retain_reference" / "TOFU_EVAL.json"
                if eval_file.exists():
                    with open(eval_file, "r") as f:
                        try:
                            # Use bucket name itself
                            data["baseline"][bucket.name] = json.load(f)
                        except json.JSONDecodeError:
                            pass

    # Load methods
    for method_dir in base_path.iterdir():
        if method_dir.is_dir() and method_dir.name != "baseline":
            method_name = method_dir.name
            data[method_name] = {}
            for bucket in method_dir.iterdir():
                if bucket.is_dir():
                    eval_file = bucket / "logs" / "TOFU_EVAL.json"
                    if eval_file.exists():
                        with open(eval_file, "r") as f:
                            try:
                                data[method_name][bucket.name] = json.load(f)
                            except json.JSONDecodeError:
                                pass
    return data

def extract_metric(eval_data, metric_name, agg_key="agg_value"):
    if not isinstance(eval_data, dict):
        return None
    metric_entry = eval_data.get(metric_name)
    if metric_entry and isinstance(metric_entry, dict):
        return metric_entry.get(agg_key)
    return None

def plot_bar_chart(run_dir, data, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    methods = [m for m in data.keys() if m != "baseline"]
    
    # find all buckets
    buckets = set()
    for m in data.values():
        buckets.update(m.keys())
    buckets = sorted(list(buckets))
    if not buckets:
        print("No buckets found.")
        return

    # find all metric names
    metrics = set()
    for m in data.values():
        for b_data in m.values():
            metrics.update([k for k in b_data.keys() if isinstance(b_data[k], dict) and "agg_value" in b_data[k]])
            
    metrics = sorted(list(metrics))

    width = 0.8 / (len(methods) + 1)
    x = np.arange(len(buckets))

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot baseline if available
        baseline_vals = [extract_metric(data.get("baseline", {}).get(b, {}), metric) or 0.0 for b in buckets]
        if sum(baseline_vals) > 0:
            ax.bar(x - width * len(methods)/2, baseline_vals, width, label="Baseline (Before)", color="gray", alpha=0.7)
            
        # Plot methods
        for i, method in enumerate(methods):
            method_vals = [extract_metric(data.get(method, {}).get(b, {}), metric) or 0.0 for b in buckets]
            ax.bar(x - width * len(methods)/2 + (i+1)*width, method_vals, width, label=method, alpha=0.9)
            
        ax.set_title(f"Comparison of {metric} across buckets")
        ax.set_ylabel(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(buckets, rotation=45, ha="right")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(Path(output_dir) / f"{metric}.png", dpi=150)
        plt.close()
        
    print(f"Saved {len(metrics)} metric plots to {output_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="Path to the output directory of the run")
    parser.add_argument("--output-dir", default=None, help="Output directory for plots")
    args = parser.parse_args()
    
    out_dir = args.output_dir if args.output_dir else os.path.join(args.run_dir, "plots")
    
    data = load_metrics(args.run_dir)
    print(f"Loaded {len(data)} groups (baseline + specific methods)")
    
    plot_bar_chart(args.run_dir, data, out_dir)

if __name__ == "__main__":
    main()
