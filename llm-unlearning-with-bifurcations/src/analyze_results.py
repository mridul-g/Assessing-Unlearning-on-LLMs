import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from matplotlib.colors import TwoSlopeNorm
from typing import Any, Dict, List, Optional, Tuple

plt.switch_backend('Agg')

SET_COLOR_SCHEMES = {
    'forget': {'color': '#e74c3c', 'label': 'Forget Set', 'before': '#f5a9a3', 'after': '#c0392b'},
    'holdout': {'color': '#f39c12', 'label': 'Holdout Set', 'before': '#f9d5a1', 'after': '#d68910'},
    'retain': {'color': '#27ae60', 'label': 'Retain Set', 'before': '#abebc6', 'after': '#1e8449'},
    'other': {'color': '#7f8c8d', 'label': 'Other', 'before': '#d5dbdb', 'after': '#566573'},
}

BUCKET_ORDER = [
    'low_grad_low_memo',
    'low_grad_high_memo',
    'high_grad_low_memo',
    'high_grad_high_memo',
]

BUCKET_COLOR_SCHEMES = {
    'low_grad_low_memo': {'color': '#7f8c8d', 'label': 'Low Grad / Low Memo'},
    'low_grad_high_memo': {'color': '#f39c12', 'label': 'Low Grad / High Memo'},
    'high_grad_low_memo': {'color': '#3498db', 'label': 'High Grad / Low Memo'},
    'high_grad_high_memo': {'color': '#e74c3c', 'label': 'High Grad / High Memo'},
}

REQUESTED_METRIC_SPECS = [
    {'label': 'Verbatim Probability', 'keys': ['forget_Q_A_Prob']},
    {'label': 'Verbatim ROUGE', 'keys': ['forget_Q_A_ROUGE']},
    {'label': 'Knowledge QA-ROUGE', 'keys': ['forget_Q_A_PARA_ROUGE']},
    {'label': 'Model Utility', 'keys': ['model_utility']},
    {'label': 'Forget Quality', 'keys': ['forget_quality']},
    {'label': 'TruthRatio', 'keys': ['forget_Truth_Ratio', 'forget_truth_ratio']},
    {'label': 'Extraction Strength', 'keys': ['extraction_strength']},
    {'label': 'Exact Memorization', 'keys': ['exact_memorization']},
    {'label': 'MIA Loss', 'keys': ['mia_loss']},
    {'label': 'MIA ZLib', 'keys': ['mia_zlib']},
    {'label': 'MIA Reference', 'keys': ['mia_reference']},
    {'label': 'MIA GradNorm', 'keys': ['mia_gradnorm']},
    {'label': 'MIA MinK', 'keys': ['mia_min_k']},
    {'label': 'MIA MinK++', 'keys': ['mia_min_k_plus_plus']},
]


def slugify_metric_name(value: str) -> str:
    safe = ''.join(char if char.isalnum() else '_' for char in str(value))
    safe = safe.strip('_').lower()
    while '__' in safe:
        safe = safe.replace('__', '_')
    return safe


def resolve_requested_metrics(metric_names: List[str]) -> Tuple[List[str], Dict[str, str]]:
    resolved = []
    labels = {}
    metric_name_set = set(metric_names)
    for spec in REQUESTED_METRIC_SPECS:
        chosen = None
        for key in spec['keys']:
            if key in metric_name_set:
                chosen = key
                break
        if chosen is None:
            continue
        if chosen in resolved:
            continue
        resolved.append(chosen)
        labels[chosen] = spec['label']
    return resolved, labels


def plot_single_metric_bars(
    labels: List[str],
    values: List[Any],
    title: str,
    ylabel: str,
    output_path: Path,
    colors: Optional[List[str]] = None,
) -> None:
    plotted = []
    for idx, label in enumerate(labels):
        numeric = _to_float_or_none(values[idx])
        if numeric is None:
            continue
        plotted.append((label, numeric, colors[idx] if colors and idx < len(colors) else '#2c3e50'))

    if not plotted:
        return

    fig_width = max(10.0, 0.5 * len(plotted) + 4.0)
    plt.figure(figsize=(fig_width, 5.5))
    x = np.arange(len(plotted))
    bar_labels = [item[0] for item in plotted]
    bar_vals = [item[1] for item in plotted]
    bar_colors = [item[2] for item in plotted]
    bars = plt.bar(x, bar_vals, color=bar_colors, alpha=0.9)

    for bar, bar_val in zip(bars, bar_vals):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f'{bar_val:.4f}',
            ha='center',
            va='bottom',
            fontsize=8,
        )

    plt.xticks(x, bar_labels, rotation=45, ha='right', fontsize=9)
    plt.ylabel(ylabel)
    plt.title(title, fontsize=12, fontweight='bold')
    plt.grid(True, axis='y', linestyle='--', alpha=0.35)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def infer_set_type(name):
    """Infer dataset type from name: forget, holdout, retain, or other."""
    name_lower = str(name).lower()
    if 'forget' in name_lower:
        return 'forget'
    if 'holdout' in name_lower:
        return 'holdout'
    if 'retain' in name_lower:
        return 'retain'
    return 'other'


def _to_float_or_none(value):
    """Return a finite float value, otherwise None."""
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def ensure_directory(path: Path) -> None:
    os.makedirs(path, exist_ok=True)


def indexed_label(name: str, index: int, width: int = 2) -> str:
    return f"{index + 1:0{width}d} | {name}"


def metric_feature_name(metric_name: str, field_name: str) -> str:
    return f'{metric_name}__{field_name}'


def collect_metric_names(*eval_payloads: Dict[str, Any]) -> List[str]:
    metric_names = set()
    for payload in eval_payloads:
        if not isinstance(payload, dict):
            continue
        for metric_name, metric_value in payload.items():
            if isinstance(metric_value, dict):
                if 'agg_value' in metric_value or 'value_by_index' in metric_value:
                    metric_names.add(metric_name)
                elif any(_to_float_or_none(value) is not None for value in metric_value.values()):
                    metric_names.add(metric_name)
            elif _to_float_or_none(metric_value) is not None:
                metric_names.add(metric_name)
    return sorted(metric_names)


def summarize_numeric_values(values: List[Any]) -> Dict[str, Optional[float]]:
    cleaned = [_to_float_or_none(value) for value in values]
    cleaned = [value for value in cleaned if value is not None]
    if not cleaned:
        return {
            'count': 0,
            'mean': None,
            'median': None,
            'std': None,
            'min': None,
            'max': None,
            'p25': None,
            'p75': None,
        }

    array = np.asarray(cleaned, dtype=float)
    return {
        'count': int(array.size),
        'mean': float(np.mean(array)),
        'median': float(np.median(array)),
        'std': float(np.std(array, ddof=0)) if array.size > 1 else 0.0,
        'min': float(np.min(array)),
        'max': float(np.max(array)),
        'p25': float(np.percentile(array, 25)),
        'p75': float(np.percentile(array, 75)),
    }


def summarize_value_map(value_map: Dict[Any, Any]) -> Dict[str, Optional[float]]:
    if not isinstance(value_map, dict):
        return summarize_numeric_values([value_map])
    return summarize_numeric_values(list(value_map.values()))


def extract_metric_payload(eval_data: Optional[Dict[str, Any]], metric_name: str) -> Optional[Dict[str, Any]]:
    if not eval_data or metric_name not in eval_data:
        return None

    metric_value = eval_data[metric_name]
    if _to_float_or_none(metric_value) is not None and not isinstance(metric_value, dict):
        return {'agg': _to_float_or_none(metric_value), 'subsets': {}}

    if not isinstance(metric_value, dict):
        return None

    payload = {
        'agg': _to_float_or_none(metric_value.get('agg_value', None)),
        'subsets': {},
    }

    if 'value_by_index' in metric_value and isinstance(metric_value['value_by_index'], dict):
        payload['subsets']['default'] = transform_to_flat_values(metric_value['value_by_index'])

    for subset_name, subset_value in metric_value.items():
        if subset_name in {'agg_value', 'value_by_index'}:
            continue
        if isinstance(subset_value, dict):
            if 'value_by_index' in subset_value and isinstance(subset_value['value_by_index'], dict):
                payload['subsets'][subset_name] = transform_to_flat_values(subset_value['value_by_index'])
            else:
                flattened = transform_to_flat_values(subset_value)
                if flattened:
                    payload['subsets'][subset_name] = flattened
        elif _to_float_or_none(subset_value) is not None:
            payload['subsets'][subset_name] = {0: _to_float_or_none(subset_value)}

    return payload


def safe_percentage_change(before_value: Optional[float], after_value: Optional[float]) -> Optional[float]:
    before_numeric = _to_float_or_none(before_value)
    after_numeric = _to_float_or_none(after_value)
    if before_numeric is None or after_numeric is None or before_numeric == 0:
        return None
    return float(((after_numeric - before_numeric) / abs(before_numeric)) * 100.0)


def build_metric_rows(
    run_id: str,
    bucket: str,
    method: str,
    metric_name: str,
    before_payload: Optional[Dict[str, Any]],
    after_payload: Optional[Dict[str, Any]],
    before_eval_path: Optional[str],
    after_eval_path: Optional[str],
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    before_agg = _to_float_or_none(before_payload.get('agg')) if before_payload else None
    after_agg = _to_float_or_none(after_payload.get('agg')) if after_payload else None
    delta = None if before_agg is None or after_agg is None else float(after_agg - before_agg)
    pct_change = safe_percentage_change(before_agg, after_agg)

    long_row = {
        'run_id': run_id,
        'bucket': bucket,
        'method': method,
        'metric': metric_name,
        'before_agg': before_agg,
        'after_agg': after_agg,
        'delta': delta,
        'pct_change': pct_change,
        'before_eval_path': before_eval_path,
        'after_eval_path': after_eval_path,
    }

    wide_row = {
        'run_id': run_id,
        'bucket': bucket,
        'method': method,
        metric_feature_name(metric_name, 'before_agg'): before_agg,
        metric_feature_name(metric_name, 'after_agg'): after_agg,
        metric_feature_name(metric_name, 'delta'): delta,
        metric_feature_name(metric_name, 'pct_change'): pct_change,
    }

    subset_rows = []
    subset_names = set()
    if before_payload and before_payload.get('subsets'):
        subset_names.update(before_payload['subsets'].keys())
    if after_payload and after_payload.get('subsets'):
        subset_names.update(after_payload['subsets'].keys())

    for subset_name in sorted(subset_names):
        before_values = before_payload.get('subsets', {}).get(subset_name, {}) if before_payload else {}
        after_values = after_payload.get('subsets', {}).get(subset_name, {}) if after_payload else {}
        before_stats = summarize_value_map(before_values)
        after_stats = summarize_value_map(after_values)

        subset_row = {
            'run_id': run_id,
            'bucket': bucket,
            'method': method,
            'metric': metric_name,
            'subset': subset_name,
        }

        for stat_name, stat_value in before_stats.items():
            subset_row[f'before_{stat_name}'] = stat_value
            wide_row[metric_feature_name(metric_name, f'before_{subset_name}_{stat_name}')] = stat_value
        for stat_name, stat_value in after_stats.items():
            subset_row[f'after_{stat_name}'] = stat_value
            wide_row[metric_feature_name(metric_name, f'after_{subset_name}_{stat_name}')] = stat_value

        subset_row['before_mean'] = before_stats.get('mean')
        subset_row['after_mean'] = after_stats.get('mean')
        subset_row['delta_mean'] = None if subset_row['before_mean'] is None or subset_row['after_mean'] is None else float(subset_row['after_mean'] - subset_row['before_mean'])
        subset_row['pct_change_mean'] = safe_percentage_change(subset_row['before_mean'], subset_row['after_mean'])
        subset_rows.append(subset_row)

    if not subset_rows:
        subset_rows.append({
            'run_id': run_id,
            'bucket': bucket,
            'method': method,
            'metric': metric_name,
            'subset': 'overall',
            'before_count': 1 if before_agg is not None else 0,
            'before_mean': before_agg,
            'before_median': before_agg,
            'before_std': 0.0 if before_agg is not None else None,
            'before_min': before_agg,
            'before_max': before_agg,
            'before_p25': before_agg,
            'before_p75': before_agg,
            'after_count': 1 if after_agg is not None else 0,
            'after_mean': after_agg,
            'after_median': after_agg,
            'after_std': 0.0 if after_agg is not None else None,
            'after_min': after_agg,
            'after_max': after_agg,
            'after_p25': after_agg,
            'after_p75': after_agg,
            'delta_mean': delta,
            'pct_change_mean': pct_change,
        })

    return long_row, wide_row, subset_rows


def plot_heatmap_matrix(
    matrix_df: pd.DataFrame,
    title: str,
    output_path: Path,
    cmap: str = 'viridis',
    center: Optional[float] = None,
    value_format: str = '.3f',
    annotate: bool = True,
) -> None:
    if matrix_df.empty:
        return

    data = matrix_df.to_numpy(dtype=float)
    if not np.isfinite(data).any():
        return

    fig_width = max(8.0, 0.8 * len(matrix_df.columns) + 2.5)
    fig_height = max(6.0, 0.5 * len(matrix_df.index) + 2.0)
    plt.figure(figsize=(fig_width, fig_height))
    ax = plt.gca()

    masked = np.ma.masked_invalid(data)
    if center is not None:
        vmin = float(np.nanmin(data))
        vmax = float(np.nanmax(data))
        norm = TwoSlopeNorm(vcenter=center, vmin=vmin, vmax=vmax)
        image = ax.imshow(masked, aspect='auto', cmap=cmap, norm=norm)
    else:
        image = ax.imshow(masked, aspect='auto', cmap=cmap)

    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(matrix_df.columns)))
    ax.set_yticks(np.arange(len(matrix_df.index)))
    ax.set_xticklabels(matrix_df.columns, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(matrix_df.index, fontsize=9)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Methods / Forget Sets')

    if annotate and matrix_df.size <= 400:
        for row_index in range(matrix_df.shape[0]):
            for col_index in range(matrix_df.shape[1]):
                value = data[row_index, col_index]
                if np.isfinite(value):
                    ax.text(
                        col_index,
                        row_index,
                        format(value, value_format),
                        ha='center',
                        va='center',
                        fontsize=8,
                        color='black' if abs(value) < 0.5 else 'white',
                    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def build_correlation_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    numeric_frame = frame.select_dtypes(include=[np.number]).copy()
    if numeric_frame.empty:
        return pd.DataFrame()

    usable_columns = []
    for column in numeric_frame.columns:
        values = numeric_frame[column].dropna()
        if len(values) < 2:
            continue
        if float(values.nunique()) <= 1:
            continue
        usable_columns.append(column)

    if not usable_columns:
        return pd.DataFrame()

    return numeric_frame[usable_columns].corr(method='pearson')


def export_json_records(file_path: Path, records: List[Dict[str, Any]]) -> None:
    ensure_directory(file_path.parent)
    with open(file_path, 'w') as handle:
        json.dump(records, handle, indent=2)


def export_dataframe_artifacts(frame: pd.DataFrame, output_dir: Path, stem: str) -> None:
    ensure_directory(output_dir)
    frame.to_csv(output_dir / f'{stem}.csv', index=False)
    export_json_records(output_dir / f'{stem}.json', frame.to_dict(orient='records'))


def build_wide_feature_frame(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    identifier_columns = ['run_id', 'bucket', 'method']
    ordered_columns = identifier_columns + sorted([column for column in frame.columns if column not in identifier_columns])
    return frame[ordered_columns]

def load_json(file_path):
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return None
    with open(file_path, 'r') as f:
        return json.load(f)


def find_latest_checkpoint_eval_json(method_bucket_dir: Path) -> Optional[Path]:
    candidates = sorted(method_bucket_dir.glob('checkpoint-*/evals/TOFU_EVAL.json'))
    if not candidates:
        return None

    def _step_key(path_obj: Path) -> int:
        checkpoint_name = path_obj.parents[1].name
        try:
            return int(checkpoint_name.split('-')[-1])
        except (TypeError, ValueError):
            return -1

    candidates = sorted(candidates, key=_step_key)
    return candidates[-1]


def list_methods_for_run(run_dir: Path, run_manifest: Dict) -> List[str]:
    ordered = []
    for method in run_manifest.get('methods', []):
        method_dir = run_dir / method
        if method_dir.is_dir():
            ordered.append(method)

    if ordered:
        return ordered

    ignored = {'baseline', 'data'}
    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.name in ignored:
            continue
        if child.name.startswith('.'):
            continue
        ordered.append(child.name)
    return ordered


def resolve_after_eval_path(run_dir: Path, method: str, bucket: str) -> Optional[Path]:
    bucket_dir = run_dir / method / bucket
    if not bucket_dir.is_dir():
        return None

    direct_eval = bucket_dir / 'evals' / 'TOFU_EVAL.json'
    if direct_eval.exists():
        return direct_eval

    return find_latest_checkpoint_eval_json(bucket_dir)


def find_run_dirs(results_dir: Path) -> List[Path]:
    if (results_dir / 'run_manifest.json').exists() and (results_dir / 'split_manifest.json').exists():
        return [results_dir]

    run_dirs = []
    for child in sorted(results_dir.iterdir()):
        if not child.is_dir():
            continue
        if (child / 'run_manifest.json').exists() and (child / 'split_manifest.json').exists():
            run_dirs.append(child)
    return run_dirs

def extract_metric_data(eval_data, metric_name):
    if not eval_data or metric_name not in eval_data:
        return None
    
    metric_info = eval_data[metric_name]
    agg_value = metric_info.get('agg_value', None)
    
    # Check for nested subset structure (MIA/memorization metrics with all three subsets)
    subsets_data = {}
    subset_keys = ['forget', 'holdout', 'retain']
    
    # Collect all available subsets
    found_subsets = []
    for subset in subset_keys:
        if subset in metric_info:
            found_subsets.append(subset)
    
    if found_subsets:
        for subset in found_subsets:
            value_by_index = metric_info.get(subset, {})
            if isinstance(value_by_index, dict) and 'value_by_index' in value_by_index:
                subsets_data[subset] = transform_to_flat_values(value_by_index['value_by_index'])
            else:
                subsets_data[subset] = transform_to_flat_values(value_by_index)
    else:
        # Standard flat structure (no subsets)
        value_by_index = metric_info.get('value_by_index', {})
        subsets_data['default'] = transform_to_flat_values(value_by_index)
            
    return {'agg': agg_value, 'subsets': subsets_data}

def transform_to_flat_values(value_by_index):
    """Auxiliary function to flatten index-level statistics."""
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
    return flat_values

def plot_comparison(results, metric_name, output_path, title, split_name):
    plt.figure(figsize=(10, 6))

    valid = []
    for method, method_result in results.items():
        if not method_result or not method_result.get('before') or not method_result.get('after'):
            continue
        before_agg = _to_float_or_none(method_result['before'].get('agg'))
        after_agg = _to_float_or_none(method_result['after'].get('agg'))
        if before_agg is None or after_agg is None:
            continue
        valid.append((method, before_agg, after_agg))

    if not valid:
        print(f"No data to plot for {metric_name}")
        plt.close()
        return

    methods = [v[0] for v in valid]
    before_vals = [v[1] for v in valid]
    after_vals = [v[2] for v in valid]
    set_type = infer_set_type(split_name)
    set_colors = SET_COLOR_SCHEMES[set_type]

    x = np.arange(len(methods))
    width = 0.35

    plt.bar(
        x - width/2,
        before_vals,
        width,
        label=f'Before Unlearning ({set_type.title()} Set)',
        color=set_colors['before'],
    )
    plt.bar(
        x + width/2,
        after_vals,
        width,
        label=f'After Unlearning ({set_type.title()} Set)',
        color=set_colors['after'],
    )

    plt.ylabel(metric_name)
    plt.title(title)
    plt.xticks(x, methods, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_per_index_scatter(baseline_subsets, method_subsets, method_name, metric_name, output_path):
    plt.figure(figsize=(8, 8))

    colors = {'forget': '#e74c3c', 'retain': '#27ae60', 'default': '#7f8c8d'}
    markers = {'forget': 'o', 'retain': 'x', 'default': 's'}
    
    all_x = []
    has_data = False
    
    for subset_name, baseline_per_index in baseline_subsets.items():
        subset_type = infer_set_type(subset_name)
        method_per_index = method_subsets.get(subset_name, {})
        common_indices = set(baseline_per_index.keys()) & set(method_per_index.keys())
        
        if not common_indices:
            continue
            
        x = []
        y = []
        for i in sorted(common_indices):
            x_val = _to_float_or_none(baseline_per_index[i])
            y_val = _to_float_or_none(method_per_index[i])
            if x_val is None or y_val is None:
                continue
            x.append(x_val)
            y.append(y_val)
            
        if x:
            plt.scatter(x, y, alpha=0.5, s=15, 
                        label=subset_type.capitalize(),
                        color=colors.get(subset_type, 'black'),
                        marker=markers.get(subset_type, 'o'))
            all_x.extend(x)
            has_data = True
            
    if not has_data:
        plt.close()
        return
    
    # Identity line
    lims = [min(all_x), max(all_x)]
    plt.plot(lims, lims, 'r--', alpha=0.7)
    
    plt.xlabel('Before Unlearning')
    plt.ylabel('After Unlearning')
    plt.title(f'{metric_name} Comparison: {method_name}')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_subset_distribution(baseline_subsets, metric_name, output_path, title):
    """Plot value distributions for all three dataset types (forget, holdout, retain) with color coding."""
    subset_points = []
    for subset_name, subset_values in baseline_subsets.items():
        set_type = infer_set_type(subset_name)
        cleaned = []
        for value in subset_values.values():
            value_float = _to_float_or_none(value)
            if value_float is not None:
                cleaned.append(value_float)
        if cleaned:
            subset_points.append((set_type, cleaned))

    if not subset_points:
        return

    plt.figure(figsize=(12, 6))
    palette = {
        'forget': '#e74c3c',      # Red
        'holdout': '#f39c12',     # Orange
        'retain': '#27ae60',      # Green
        'other': '#7f8c8d',       # Gray
    }

    for set_type, values in subset_points:
        color = palette.get(set_type, '#7f8c8d')
        arr = np.array(values, dtype=float)
        label = f"{set_type.title()} (n={len(arr)})"
        plt.hist(arr, bins=30, alpha=0.4, density=True, label=label, color=color, edgecolor='black', linewidth=0.5)

        if torch.cuda.is_available() and len(values) > 0:
            arr_t = torch.tensor(values, dtype=torch.float32, device='cuda')
            p50 = float(torch.quantile(arr_t, 0.50).cpu().item())
            p90 = float(torch.quantile(arr_t, 0.90).cpu().item())
        else:
            p50 = float(np.percentile(arr, 50))
            p90 = float(np.percentile(arr, 90))

        plt.axvline(p50, color=color, linestyle='--', linewidth=2, alpha=0.8, label=f'{set_type.title()} p50')
        plt.axvline(p90, color=color, linestyle=':', linewidth=2, alpha=0.8, label=f'{set_type.title()} p90')

    plt.xlabel(metric_name, fontsize=11)
    plt.ylabel('Density', fontsize=11)
    plt.title(title, fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(fontsize=9, loc='best')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def plot_three_datasets_aggregate(results, metric_name, output_path, title):
    """Plot aggregate metric values for all three dataset types (forget, holdout, retain)."""
    palette = {
        'forget': '#e74c3c',      # Red
        'holdout': '#f39c12',     # Orange
        'retain': '#27ae60',      # Green
    }
    
    # Extract scores for each dataset type
    data_by_type = {'forget': [], 'holdout': [], 'retain': []}
    
    for method_result in results.values():
        if not method_result or not method_result.get('subsets'):
            continue
        for subset_name, subset_vals in method_result['subsets'].items():
            set_type = infer_set_type(subset_name)
            if set_type in data_by_type and subset_vals:
                # Calculate mean for this subset
                values = [_to_float_or_none(v) for v in subset_vals.values()]
                values = [v for v in values if v is not None]
                if values:
                    if torch.cuda.is_available():
                        v_t = torch.tensor(values, dtype=torch.float32, device='cuda')
                        mean_val = float(torch.mean(v_t).cpu().item())
                    else:
                        mean_val = float(np.mean(values))
                    data_by_type[set_type].append(mean_val)
    
    # Check if we have data
    has_data = any(data_by_type.values())
    if not has_data:
        return
    
    plt.figure(figsize=(10, 6))
    
    # Create boxplot for each dataset type
    box_data = []
    box_labels = []
    box_colors = []
    
    for set_type in ['forget', 'holdout', 'retain']:
        if data_by_type[set_type]:
            box_data.append(data_by_type[set_type])
            box_labels.append(set_type.capitalize())
            box_colors.append(palette[set_type])
    
    if box_data:
        bp = plt.boxplot(box_data, labels=box_labels, patch_artist=True, widths=0.6)
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element], color='black', linewidth=1.5)
    
    plt.ylabel(metric_name, fontsize=11)
    plt.title(title, fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.4, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def run_analysis():
    parser = argparse.ArgumentParser(description='Analyze unlearning results')
    parser.add_argument('--results_dir', type=str, required=True, help='Run directory or parent results directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save analysis')
    args = parser.parse_args()

    ensure_directory(Path(args.output_dir))

    target_metrics = [
        'forget_Truth_Ratio',
        'forget_quality',
        'forget_Q_A_Prob',
        'forget_Q_A_ROUGE',
        'forget_Q_A_PARA_ROUGE',
        'model_utility',
        'extraction_strength',
        'exact_memorization',
        'mia_min_k_plus_plus',
        'mia_min_k',
        'mia_loss',
        'mia_zlib',
        'mia_gradnorm',
        'mia_reference',
    ]

    results_root = Path(args.results_dir)
    output_root = Path(args.output_dir)
    run_dirs = find_run_dirs(results_root)
    if not run_dirs:
        raise RuntimeError(
            f'No run directories with run_manifest.json and split_manifest.json found under {results_root}'
        )

    for run_dir in run_dirs:
        run_manifest = load_json(run_dir / 'run_manifest.json')
        split_manifest = load_json(run_dir / 'split_manifest.json')
        if not run_manifest or not split_manifest:
            print(f'Skipping invalid run directory: {run_dir}')
            continue

        run_id = run_manifest.get('run_id', run_dir.name)
        baseline_file = run_dir / 'baseline' / 'TOFU_EVAL.json'
        baseline_data = load_json(baseline_file)
        if not baseline_data:
            print(f'Skipping run with missing baseline eval JSON: {run_dir}')
            continue

        bucket_names = list(split_manifest.get('selected_example_ids', {}).keys())
        if not bucket_names:
            print(f'Skipping run with no selected buckets: {run_dir}')
            continue

        methods = list_methods_for_run(run_dir, run_manifest)
        if not methods:
            print(f'No method directories found for run: {run_dir}')
            continue

        run_output_dir = output_root / run_id
        ensure_directory(run_output_dir)

        bucket_names = [bucket for bucket in BUCKET_ORDER if bucket in bucket_names] + [bucket for bucket in bucket_names if bucket not in BUCKET_ORDER]
        method_order = {method_name: index for index, method_name in enumerate(methods)}
        bucket_order = {bucket_name: index for index, bucket_name in enumerate(bucket_names)}

        eval_cache: Dict[str, Dict[str, Any]] = {}
        metric_names = set(collect_metric_names(baseline_data))

        for bucket in bucket_names:
            eval_cache[bucket] = {}
            for method in methods:
                eval_file = resolve_after_eval_path(run_dir, method, bucket)
                eval_data = load_json(eval_file) if eval_file else None
                eval_cache[bucket][method] = {
                    'eval_data': eval_data,
                    'eval_path': str(eval_file) if eval_file else None,
                }
                metric_names.update(collect_metric_names(eval_data))

        metric_names = sorted(metric_names)
        requested_metric_keys, requested_metric_labels = resolve_requested_metrics(metric_names)
        long_rows: List[Dict[str, Any]] = []
        subset_rows: List[Dict[str, Any]] = []
        wide_records: Dict[Tuple[str, str], Dict[str, Any]] = {}

        for bucket in bucket_names:
            summary_results: Dict[str, Dict[str, Any]] = {}

            for method in methods:
                cache_entry = eval_cache[bucket].get(method, {})
                eval_data = cache_entry.get('eval_data')
                after_eval_path = cache_entry.get('eval_path')
                summary_results[method] = {}

                for metric in metric_names:
                    before = extract_metric_data(baseline_data, metric)
                    after = extract_metric_data(eval_data, metric)
                    if before is None and after is None:
                        continue

                    summary_results[method][metric] = {'before': before, 'after': after}
                    long_row, wide_row, subset_metric_rows = build_metric_rows(
                        run_id,
                        bucket,
                        method,
                        metric,
                        before,
                        after,
                        str(baseline_file),
                        after_eval_path,
                    )
                    long_rows.append(long_row)
                    subset_rows.extend(subset_metric_rows)

                    record_key = (bucket, method)
                    if record_key not in wide_records:
                        wide_records[record_key] = {
                            'run_id': run_id,
                            'bucket': bucket,
                            'method': method,
                            'bucket_index': bucket_order.get(bucket),
                            'method_index': method_order.get(method),
                        }
                    wide_records[record_key].update(wide_row)

                for metric in target_metrics:
                    before = summary_results.get(method, {}).get(metric, {}).get('before')
                    after = summary_results.get(method, {}).get(metric, {}).get('after')
                    if before and after and before.get('subsets') and after.get('subsets'):
                        scatter_dir = run_output_dir / 'legacy' / bucket / 'scatter'
                        ensure_directory(scatter_dir)
                        plot_per_index_scatter(
                            before['subsets'],
                            after['subsets'],
                            method,
                            metric,
                            scatter_dir / f'{method}_{metric}_scatter.png',
                        )

            for metric in target_metrics:
                baseline_metric = extract_metric_data(baseline_data, metric)
                if not baseline_metric or not baseline_metric.get('subsets'):
                    continue

                dist_dir = run_output_dir / 'legacy' / bucket / 'distributions'
                ensure_directory(dist_dir)
                plot_subset_distribution(
                    baseline_metric['subsets'],
                    metric,
                    dist_dir / f'baseline_{metric}_subset_distribution.png',
                    f'{run_id} ({bucket}) {metric}: Forget (Red) vs Holdout (Orange) vs Retain (Green)',
                )

                plot_three_datasets_aggregate(
                    {method: summary_results.get(method, {}).get(metric) for method in methods},
                    metric,
                    dist_dir / f'baseline_{metric}_three_datasets_boxplot.png',
                    f'{run_id} ({bucket}) {metric} by Dataset Type',
                )

            for metric in target_metrics:
                metric_results = {method: summary_results.get(method, {}).get(metric) for method in methods if method in summary_results}
                metric_results = {key: value for key, value in metric_results.items() if value and value.get('before') and value.get('after')}
                if not metric_results:
                    continue

                agg_dir = run_output_dir / 'legacy' / bucket / 'aggregate'
                ensure_directory(agg_dir)
                plot_comparison(
                    metric_results,
                    metric,
                    agg_dir / f'{metric}_comparison.png',
                    f'{run_id} ({bucket}) {metric} Comparison',
                    bucket,
                )

        summary_df = pd.DataFrame(long_rows)
        subset_df = pd.DataFrame(subset_rows)
        wide_df = build_wide_feature_frame(list(wide_records.values()))

        if not summary_df.empty and requested_metric_keys:
            requested_summary_df = summary_df[summary_df['metric'].isin(requested_metric_keys)].copy()
            requested_summary_df['metric_label'] = requested_summary_df['metric'].map(
                lambda key: requested_metric_labels.get(key, key)
            )
        else:
            requested_summary_df = pd.DataFrame()

        reports_dir = run_output_dir / 'reports'
        by_forget_set_dir = reports_dir / 'by_forget_set'
        by_model_dir = reports_dir / 'by_model'
        correlation_dir = reports_dir / 'correlation'
        ensure_directory(by_forget_set_dir)
        ensure_directory(by_model_dir)
        ensure_directory(correlation_dir)

        if not summary_df.empty:
            summary_df.to_csv(reports_dir / 'tofu_bucket_summary.csv', index=False)
            export_json_records(reports_dir / 'tofu_bucket_summary.json', summary_df.to_dict(orient='records'))

        if not requested_summary_df.empty:
            requested_summary_df.to_csv(reports_dir / 'requested_metrics_table.csv', index=False)
            export_json_records(
                reports_dir / 'requested_metrics_table.json',
                requested_summary_df.to_dict(orient='records'),
            )

        if not subset_df.empty:
            subset_df.to_csv(reports_dir / 'tofu_bucket_subset_summary.csv', index=False)
            export_json_records(reports_dir / 'tofu_bucket_subset_summary.json', subset_df.to_dict(orient='records'))

        if not wide_df.empty:
            wide_df.to_csv(reports_dir / 'tofu_bucket_summary_wide.csv', index=False)
            export_json_records(reports_dir / 'tofu_bucket_summary_wide.json', wide_df.to_dict(orient='records'))

        for bucket in bucket_names:
            bucket_wide = wide_df[wide_df['bucket'] == bucket].copy()
            if bucket_wide.empty:
                continue

            bucket_dir = by_forget_set_dir / bucket
            ensure_directory(bucket_dir)
            export_dataframe_artifacts(bucket_wide, bucket_dir, 'summary')

            metric_after_cols = [metric_feature_name(metric, 'after_agg') for metric in metric_names if metric_feature_name(metric, 'after_agg') in bucket_wide.columns]
            metric_delta_cols = [metric_feature_name(metric, 'delta') for metric in metric_names if metric_feature_name(metric, 'delta') in bucket_wide.columns]

            if metric_after_cols:
                after_matrix = bucket_wide.set_index('method')[metric_after_cols].reindex(methods)
                after_matrix.index = [indexed_label(method, method_order[method]) for method in after_matrix.index]
                after_matrix.columns = [column.replace('__after_agg', '') for column in after_matrix.columns]
                plot_heatmap_matrix(
                    after_matrix,
                    f'{run_id} | {bucket} | Methods vs Metrics (after)',
                    bucket_dir / 'methods_vs_metrics_after.png',
                    cmap='viridis',
                )

            if metric_delta_cols:
                delta_matrix = bucket_wide.set_index('method')[metric_delta_cols].reindex(methods)
                delta_matrix.index = [indexed_label(method, method_order[method]) for method in delta_matrix.index]
                delta_matrix.columns = [column.replace('__delta', '') for column in delta_matrix.columns]
                plot_heatmap_matrix(
                    delta_matrix,
                    f'{run_id} | {bucket} | Methods vs Metrics (delta)',
                    bucket_dir / 'methods_vs_metrics_delta.png',
                    cmap='coolwarm',
                    center=0.0,
                    value_format='.4f',
                )

            if requested_metric_keys:
                metric_bar_dir = bucket_dir / 'per_metric'
                ensure_directory(metric_bar_dir)
                method_labels = [indexed_label(method, method_order[method]) for method in methods]
                method_palette = [plt.cm.tab20(index % 20) for index in range(len(methods))]
                for metric_key in requested_metric_keys:
                    metric_col_before = metric_feature_name(metric_key, 'before_agg')
                    metric_col_after = metric_feature_name(metric_key, 'after_agg')
                    metric_col_delta = metric_feature_name(metric_key, 'delta')
                    if metric_col_before in bucket_wide.columns:
                        values = [
                            bucket_wide[bucket_wide['method'] == method][metric_col_before].iloc[0]
                            if not bucket_wide[bucket_wide['method'] == method].empty else None
                            for method in methods
                        ]
                        plot_single_metric_bars(
                            method_labels,
                            values,
                            f'{run_id} | {bucket} | {requested_metric_labels.get(metric_key, metric_key)} (before)',
                            'Before',
                            metric_bar_dir / f'{slugify_metric_name(metric_key)}_before_bar.png',
                            colors=method_palette,
                        )
                    if metric_col_after in bucket_wide.columns:
                        values = [
                            bucket_wide[bucket_wide['method'] == method][metric_col_after].iloc[0]
                            if not bucket_wide[bucket_wide['method'] == method].empty else None
                            for method in methods
                        ]
                        plot_single_metric_bars(
                            method_labels,
                            values,
                            f'{run_id} | {bucket} | {requested_metric_labels.get(metric_key, metric_key)} (after)',
                            'After',
                            metric_bar_dir / f'{slugify_metric_name(metric_key)}_after_bar.png',
                            colors=method_palette,
                        )
                    if metric_col_delta in bucket_wide.columns:
                        values = [
                            bucket_wide[bucket_wide['method'] == method][metric_col_delta].iloc[0]
                            if not bucket_wide[bucket_wide['method'] == method].empty else None
                            for method in methods
                        ]
                        plot_single_metric_bars(
                            method_labels,
                            values,
                            f'{run_id} | {bucket} | {requested_metric_labels.get(metric_key, metric_key)} (delta)',
                            'Delta (after - before)',
                            metric_bar_dir / f'{slugify_metric_name(metric_key)}_delta_bar.png',
                            colors=method_palette,
                        )

            if not requested_summary_df.empty:
                requested_bucket = requested_summary_df[requested_summary_df['bucket'] == bucket].copy()
                if not requested_bucket.empty:
                    requested_bucket.to_csv(bucket_dir / 'requested_metrics_table.csv', index=False)
                    export_json_records(
                        bucket_dir / 'requested_metrics_table.json',
                        requested_bucket.to_dict(orient='records'),
                    )

        for method in methods:
            method_wide = wide_df[wide_df['method'] == method].copy()
            if method_wide.empty:
                continue

            method_dir = by_model_dir / method
            ensure_directory(method_dir)
            export_dataframe_artifacts(method_wide, method_dir, 'summary')

            bucket_after_cols = [metric_feature_name(metric, 'after_agg') for metric in metric_names if metric_feature_name(metric, 'after_agg') in method_wide.columns]
            bucket_delta_cols = [metric_feature_name(metric, 'delta') for metric in metric_names if metric_feature_name(metric, 'delta') in method_wide.columns]

            if bucket_after_cols:
                after_matrix = method_wide.set_index('bucket')[bucket_after_cols].reindex(bucket_names)
                after_matrix.index = [indexed_label(bucket, bucket_order[bucket]) for bucket in after_matrix.index]
                after_matrix.columns = [column.replace('__after_agg', '') for column in after_matrix.columns]
                plot_heatmap_matrix(
                    after_matrix,
                    f'{run_id} | {method} | Forget Sets vs Metrics (after)',
                    method_dir / 'forget_sets_vs_metrics_after.png',
                    cmap='viridis',
                )

            if bucket_delta_cols:
                delta_matrix = method_wide.set_index('bucket')[bucket_delta_cols].reindex(bucket_names)
                delta_matrix.index = [indexed_label(bucket, bucket_order[bucket]) for bucket in delta_matrix.index]
                delta_matrix.columns = [column.replace('__delta', '') for column in delta_matrix.columns]
                plot_heatmap_matrix(
                    delta_matrix,
                    f'{run_id} | {method} | Forget Sets vs Metrics (delta)',
                    method_dir / 'forget_sets_vs_metrics_delta.png',
                    cmap='coolwarm',
                    center=0.0,
                    value_format='.4f',
                )

            if requested_metric_keys:
                metric_bar_dir = method_dir / 'per_metric'
                ensure_directory(metric_bar_dir)
                bucket_labels = [indexed_label(bucket, bucket_order[bucket]) for bucket in bucket_names]
                bucket_palette = [BUCKET_COLOR_SCHEMES.get(bucket, {}).get('color', '#7f8c8d') for bucket in bucket_names]
                for metric_key in requested_metric_keys:
                    metric_col_before = metric_feature_name(metric_key, 'before_agg')
                    metric_col_after = metric_feature_name(metric_key, 'after_agg')
                    metric_col_delta = metric_feature_name(metric_key, 'delta')
                    if metric_col_before in method_wide.columns:
                        values = [
                            method_wide[method_wide['bucket'] == bucket][metric_col_before].iloc[0]
                            if not method_wide[method_wide['bucket'] == bucket].empty else None
                            for bucket in bucket_names
                        ]
                        plot_single_metric_bars(
                            bucket_labels,
                            values,
                            f'{run_id} | {method} | {requested_metric_labels.get(metric_key, metric_key)} (before)',
                            'Before',
                            metric_bar_dir / f'{slugify_metric_name(metric_key)}_before_bar.png',
                            colors=bucket_palette,
                        )
                    if metric_col_after in method_wide.columns:
                        values = [
                            method_wide[method_wide['bucket'] == bucket][metric_col_after].iloc[0]
                            if not method_wide[method_wide['bucket'] == bucket].empty else None
                            for bucket in bucket_names
                        ]
                        plot_single_metric_bars(
                            bucket_labels,
                            values,
                            f'{run_id} | {method} | {requested_metric_labels.get(metric_key, metric_key)} (after)',
                            'After',
                            metric_bar_dir / f'{slugify_metric_name(metric_key)}_after_bar.png',
                            colors=bucket_palette,
                        )
                    if metric_col_delta in method_wide.columns:
                        values = [
                            method_wide[method_wide['bucket'] == bucket][metric_col_delta].iloc[0]
                            if not method_wide[method_wide['bucket'] == bucket].empty else None
                            for bucket in bucket_names
                        ]
                        plot_single_metric_bars(
                            bucket_labels,
                            values,
                            f'{run_id} | {method} | {requested_metric_labels.get(metric_key, metric_key)} (delta)',
                            'Delta (after - before)',
                            metric_bar_dir / f'{slugify_metric_name(metric_key)}_delta_bar.png',
                            colors=bucket_palette,
                        )

            if not requested_summary_df.empty:
                requested_method = requested_summary_df[requested_summary_df['method'] == method].copy()
                if not requested_method.empty:
                    requested_method.to_csv(method_dir / 'requested_metrics_table.csv', index=False)
                    export_json_records(
                        method_dir / 'requested_metrics_table.json',
                        requested_method.to_dict(orient='records'),
                    )

        if not wide_df.empty:
            correlation_matrix = build_correlation_matrix(wide_df)
            if not correlation_matrix.empty:
                global_corr_dir = correlation_dir / 'global'
                ensure_directory(global_corr_dir)
                correlation_matrix.to_csv(global_corr_dir / 'correlation.csv')
                export_json_records(
                    global_corr_dir / 'correlation.json',
                    correlation_matrix.reset_index().rename(columns={'index': 'feature'}).to_dict(orient='records'),
                )
                plot_heatmap_matrix(
                    correlation_matrix,
                    f'{run_id} | Global Metric Correlation',
                    global_corr_dir / 'correlation.png',
                    cmap='coolwarm',
                    center=0.0,
                    value_format='.2f',
                    annotate=correlation_matrix.shape[0] <= 20,
                )

        for bucket in bucket_names:
            bucket_wide = wide_df[wide_df['bucket'] == bucket].copy()
            if bucket_wide.empty:
                continue
            correlation_matrix = build_correlation_matrix(bucket_wide)
            if correlation_matrix.empty:
                continue
            bucket_corr_dir = correlation_dir / 'by_forget_set' / bucket
            ensure_directory(bucket_corr_dir)
            correlation_matrix.to_csv(bucket_corr_dir / 'correlation.csv')
            export_json_records(
                bucket_corr_dir / 'correlation.json',
                correlation_matrix.reset_index().rename(columns={'index': 'feature'}).to_dict(orient='records'),
            )
            plot_heatmap_matrix(
                correlation_matrix,
                f'{run_id} | {bucket} Metric Correlation',
                bucket_corr_dir / 'correlation.png',
                cmap='coolwarm',
                center=0.0,
                value_format='.2f',
                annotate=correlation_matrix.shape[0] <= 20,
            )

    print(f'Analysis complete. Results saved in {args.output_dir}')


if __name__ == "__main__":
    run_analysis()
