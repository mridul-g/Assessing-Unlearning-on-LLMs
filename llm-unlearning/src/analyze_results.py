import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.switch_backend('Agg')

def load_json(file_path):
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return None
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_metric_data(eval_data, metric_name):
    if not eval_data or metric_name not in eval_data:
        return None
    
    metric_info = eval_data[metric_name]
    agg_value = metric_info.get('agg_value', None)
    value_by_index = metric_info.get('value_by_index', {})
    
    # Flatten value_by_index if it's nested (e.g. {'0': {'score': 0.5}})
    flat_values = {}
    for idx, val in value_by_index.items():
        if isinstance(val, dict):
            # Try common keys like 'score', 'prob', 'avg_loss', 'norm'
            for key in ['score', 'prob', 'avg_loss', 'norm']:
                if key in val:
                    flat_values[int(idx)] = val[key]
                    break
        else:
            flat_values[int(idx)] = val
            
    return {'agg': agg_value, 'per_index': flat_values}

def plot_comparison(results, metric_name, output_path, title):
    plt.figure(figsize=(10, 6))
    
    methods = list(results.keys())
    before_vals = [results[m]['before']['agg'] for m in methods if results[m]['before']]
    after_vals = [results[m]['after']['agg'] for m in methods if results[m]['after']]
    
    if not before_vals or not after_vals:
        print(f"No data to plot for {metric_name}")
        plt.close()
        return

    x = np.arange(len(methods))
    width = 0.35

    plt.bar(x - width/2, before_vals, width, label='Before Unlearning', color='skyblue')
    plt.bar(x + width/2, after_vals, width, label='After Unlearning', color='salmon')

    plt.ylabel(metric_name)
    plt.title(title)
    plt.xticks(x, methods, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_per_index_scatter(baseline_per_index, method_per_index, method_name, metric_name, output_path):
    plt.figure(figsize=(8, 8))
    
    common_indices = set(baseline_per_index.keys()) & set(method_per_index.keys())
    if not common_indices:
        plt.close()
        return
        
    x = [baseline_per_index[i] for i in sorted(common_indices)]
    y = [method_per_index[i] for i in sorted(common_indices)]
    
    plt.scatter(x, y, alpha=0.5, s=10)
    plt.plot([min(x), max(x)], [min(x), max(x)], 'r--', alpha=0.7) # Identity line
    
    plt.xlabel('Before Unlearning')
    plt.ylabel('After Unlearning')
    plt.title(f'{metric_name} Comparison: {method_name}')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Analyze unlearning results')
    parser.add_argument('--results_dir', type=str, required=True, help='Directory containing results')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save analysis')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Metrics to analyze
    target_metrics = ['exact_memorization', 'mia_gradnorm', 'model_utility', 'forget_quality']
    
    for benchmark in ['muse', 'tofu']:
        benchmark_dir = Path(args.results_dir) / benchmark
        if not benchmark_dir.exists():
            continue
            
        baseline_root = benchmark_dir / "baseline"
        if not baseline_root.exists():
            continue
            
        data_splits = [d.name for d in baseline_root.iterdir() if d.is_dir()]
        methods = [d.name for d in benchmark_dir.iterdir() if d.is_dir() and d.name != "baseline" and d.name != "analysis"]
        
        all_metrics_summary = []
        
        for split in data_splits:
            baseline_file = baseline_root / split / f"{benchmark.upper()}_EVAL.json"
            baseline_data = load_json(baseline_file)
            if not baseline_data:
                continue

            summary_results = {}
            for method in methods:
                eval_file = benchmark_dir / method / split / "evals" / f"{benchmark.upper()}_EVAL.json"
                eval_data = load_json(eval_file)
                
                summary_results[method] = {}
                for metric in target_metrics:
                    before = extract_metric_data(baseline_data, metric)
                    after = extract_metric_data(eval_data, metric)
                    summary_results[method][metric] = {'before': before, 'after': after}
                    
                    # Plot per-index comparison for each method and split
                    if before and after and before['per_index'] and after['per_index']:
                        scatter_dir = Path(args.output_dir) / benchmark / split / "scatter"
                        os.makedirs(scatter_dir, exist_ok=True)
                        plot_per_index_scatter(
                            before['per_index'], 
                            after['per_index'], 
                            method, 
                            metric, 
                            scatter_dir / f"{method}_{metric}_scatter.png"
                        )

            # Plot aggregate comparisons for this split
            for metric in target_metrics:
                metric_results = {m: summary_results.get(m, {}).get(metric) for m in methods if m in summary_results}
                metric_results = {k: v for k, v in metric_results.items() if v and v['after']}
                
                if not metric_results:
                    continue
                    
                agg_dir = Path(args.output_dir) / benchmark / split / "aggregate"
                os.makedirs(agg_dir, exist_ok=True)
                
                # Prepare summary for CSV
                for m in metric_results:
                    all_metrics_summary.append({
                        'benchmark': benchmark,
                        'split': split,
                        'method': m,
                        'metric': metric,
                        'before': metric_results[m]['before']['agg'],
                        'after': metric_results[m]['after']['agg']
                    })

                plot_comparison(
                    metric_results, 
                    metric, 
                    agg_dir / f"{metric}_comparison.png",
                    f"{benchmark.upper()} ({split}) {metric} Comparison"
                )
            
        # Save CSV summary
        if all_metrics_summary:
            summary_df = pd.DataFrame(all_metrics_summary)
            summary_df.to_csv(Path(args.output_dir) / f"{benchmark}_summary.csv", index=False)

    print(f"Analysis complete. Results saved in {args.output_dir}")

if __name__ == "__main__":
    main()
