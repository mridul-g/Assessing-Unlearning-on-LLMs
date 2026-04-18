#!/usr/bin/env python3

import argparse
import csv
import json
import logging
import os
import shutil
import signal
import subprocess
import traceback
import datetime
from pathlib import Path
from collections import defaultdict

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger("pipeline")

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg") 
import torch
from sklearn.metrics import roc_auc_score
from datasets import load_dataset
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict
from transformers.trainer_utils import get_last_checkpoint

from data import get_collators, get_data
from model import get_model
from trainer import load_trainer
from trainer.utils import seed_everything
from evals import get_evaluators


CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE_MODEL = "open-unlearning/tofu_Llama-3.2-1B-Instruct_full"
DEFAULT_FORGET_SPLIT = "full"
DEFAULT_HOLDOUT_SPLIT = "holdout10"
DEFAULT_RETAIN_SPLIT = "null"
DEFAULT_METHODS = [
    "GradAscent",
    "GradDiff",
    "NPO",
    "SimNPO",
    "RMU",
    "UNDIAL",
    "SatImp",
    "WGA",
    "CEU",
    "PDU",
]

UNSUPPORTED_PIPELINE_METHODS = {
    "DPO": "DPO needs pairwise preference data and is not compatible with this JSONL bucket pipeline.",
    "AltPO": "AltPO trainer/config is not available in this repository.",
}

CURRENT_STAGE = "startup"
PROGRESS_LOG_FILE = None


def _format_kv(fields):
    parts = []
    for key, value in fields.items():
        if value is None:
            continue
        parts.append(f"{key}={value}")
    return " ".join(parts)


def log_progress(event, **fields):
    if PROGRESS_LOG_FILE is None:
        return
    payload = {
        "stage": CURRENT_STAGE,
        **fields,
    }
    line = f"[{utc_now()}] {event}"
    suffix = _format_kv(payload)
    if suffix:
        line += f" | {suffix}"
    with open(PROGRESS_LOG_FILE, "a") as f:
        f.write(line + "\n")


def init_progress_log(run_dir, run_id):
    global PROGRESS_LOG_FILE
    PROGRESS_LOG_FILE = Path(run_dir) / "CLEAN_PROGRESS.log"
    ensure_dir(Path(PROGRESS_LOG_FILE).parent)
    with open(PROGRESS_LOG_FILE, "a") as f:
        f.write(f"\n=== run_id={run_id} started_at={utc_now()} ===\n")
    log_progress("run_start", run_id=run_id)


def set_stage(stage_name):
    global CURRENT_STAGE
    CURRENT_STAGE = stage_name
    log_progress("stage_start", stage=stage_name)


def _write_sigterm_debug(frame):
    """Write a minimal SIGTERM diagnostic record to help identify kill context."""
    debug_log = REPO_ROOT / "scripts" / "logs" / "sigterm_debug.log"
    ensure_dir(debug_log.parent)
    payload = {
        "timestamp": utc_now(),
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "stage": CURRENT_STAGE,
        "run_id_env": os.environ.get("RUN_ID"),
    }
    if frame is not None:
        stack = traceback.extract_stack(frame)
        payload["top_frame"] = {
            "file": stack[-1].filename if stack else None,
            "line": stack[-1].lineno if stack else None,
            "name": stack[-1].name if stack else None,
        }

    with open(debug_log, "a") as f:
        f.write(json.dumps(payload) + "\n")


def _sigterm_handler(signum, frame):
    _write_sigterm_debug(frame)
    raise SystemExit(143)


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def compose_cfg(config_name, overrides):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(config_name=config_name, overrides=overrides)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def write_json(path, payload):
    ensure_dir(Path(path).parent)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    log_progress("save_json", path=path)


def ensure_gpu_ready():
    """Require CUDA and try to recover from missing MPS daemon on shared GPU hosts."""
    if torch.cuda.is_available():
        return

    pipe_dir = os.environ.get("CUDA_MPS_PIPE_DIRECTORY", f"/tmp/nvidia-mps-{os.environ.get('USER', 'user')}")
    log_dir = os.environ.get("CUDA_MPS_LOG_DIRECTORY", f"/tmp/nvidia-log-{os.environ.get('USER', 'user')}")
    os.environ["CUDA_MPS_PIPE_DIRECTORY"] = pipe_dir
    os.environ["CUDA_MPS_LOG_DIRECTORY"] = log_dir
    ensure_dir(pipe_dir)
    ensure_dir(log_dir)

    try:
        subprocess.run(["nvidia-cuda-mps-control", "-d"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        log_progress("mps_daemon_started", pipe_dir=pipe_dir, log_dir=log_dir)
    except Exception as exc:
        log_progress("mps_daemon_start_failed", error=str(exc))

    if torch.cuda.is_available():
        log_progress("gpu_ready", cuda_device_count=torch.cuda.device_count())
        return

    raise RuntimeError(
        "CUDA is not available for PyTorch. This pipeline requires GPU. "
        "If running on this cluster, ensure MPS is available and launch with valid CUDA env."
    )


def log_vram_snapshot(point, method=None, bucket=None):
    """Record a tiny VRAM snapshot for debugging accumulation across iterations."""
    if not torch.cuda.is_available():
        log_progress("vram_snapshot", point=point, method=method, bucket=bucket, cuda="unavailable")
        return

    try:
        device_index = torch.cuda.current_device()
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
        allocated_bytes = torch.cuda.memory_allocated(device_index)
        reserved_bytes = torch.cuda.memory_reserved(device_index)
        log_progress(
            "vram_snapshot",
            point=point,
            method=method,
            bucket=bucket,
            device=device_index,
            allocated_mb=round(allocated_bytes / (1024**2), 2),
            reserved_mb=round(reserved_bytes / (1024**2), 2),
            free_mb=round(free_bytes / (1024**2), 2),
            total_mb=round(total_bytes / (1024**2), 2),
        )
    except Exception as exc:
        log_progress(
            "vram_snapshot_failed",
            point=point,
            method=method,
            bucket=bucket,
            error=str(exc),
        )


def is_baseline_complete(baseline_dir, expected_forget_split):
    details = Path(baseline_dir) / "TOFU_EVAL_DETAILS.csv"
    summary = Path(baseline_dir) / "TOFU_EVAL.json"
    
    if not (details.exists() and summary.exists()):
        return False
        
    # If the summary exists, verify it was run on the correct split if possible
    try:
        with open(summary, "r") as f:
            data = json.load(f)
            # Check if any metric has a 'forget' subset that matched a different size
            # This is a bit complex, so we'll just check the file size of the details as a heuristic
            if os.path.getsize(details) < 10000 and expected_forget_split == "full":
                return False
    except Exception:
        return False

    if "exact_memorization" not in data or "mia_gradnorm" not in data:
        return False

    # Check for flat unified checkpoint if it exists (no bifurcation requested)
    checkpoint_root = Path(baseline_dir) / "checkpoints" / "mia_gradnorm"
    unified_ckpt = checkpoint_root / "attack_checkpoint.json"
    
    if unified_ckpt.exists():
        try:
            checkpoint_data = read_json(unified_ckpt)
            return checkpoint_data.get("done", False)
        except Exception:
            return False
            
    # Fallback to bifurcated check for existing runs
    for subset in ("forget", "holdout", "retain"):
        ckpt = checkpoint_root / subset / "attack_checkpoint.json"
        if not ckpt.exists():
            if subset == "retain" and expected_forget_split == "full":
                continue
            return False
        try:
            checkpoint_data = read_json(ckpt)
            if not checkpoint_data.get("done", False):
                return False
        except Exception:
            return False

    return True


def is_bucket_method_complete(bucket_dir):
    marker = Path(bucket_dir) / "RUN_COMPLETE.json"
    return marker.exists()


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def get_last_valid_trainer_checkpoint(output_dir):
    """Return latest checkpoint with trainer_state.json, skipping partial writes."""
    output_path = Path(output_dir)
    candidates = sorted(output_path.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]), reverse=True)
    for checkpoint_dir in candidates:
        trainer_state = checkpoint_dir / "trainer_state.json"
        if trainer_state.exists():
            return str(checkpoint_dir)
        log_progress("checkpoint_skip_incomplete", checkpoint=str(checkpoint_dir), missing="trainer_state.json")
    return None


def load_csv_rows(path):
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def assert_eval_metrics_complete(eval_logs_path, expected_metrics):
    """Ensure evaluator output contains every configured metric key."""
    if not Path(eval_logs_path).exists():
        raise RuntimeError(f"Missing eval logs file: {eval_logs_path}")

    logs = read_json(eval_logs_path)
    missing = [metric_name for metric_name in expected_metrics if metric_name not in logs]
    if missing:
        raise RuntimeError(
            "Post-unlearning evaluation is incomplete. "
            f"Missing metrics ({len(missing)}): {', '.join(sorted(missing))}"
        )


def to_int(value):
    if value is None or value == "":
        return None
    return int(float(value))


def to_float(value):
    if value is None or value == "":
        return None
    return float(value)


def extract_metric_map(rows, metric_name, subset_name="forget", stat_key="score"):
    score_map = {}
    for row in rows:
        if row["metric"] != metric_name:
            continue
        if row["subset"] != subset_name:
            continue
        if row["stat_key"] != stat_key:
            continue
        example_id = to_int(row["example_id"])
        value = to_float(row["value"])
        if example_id is not None and value is not None:
            score_map[example_id] = value
    return score_map


def load_tofu_split(split_name):
    return load_dataset("locuslab/TOFU", name=split_name, split="train")


def write_jsonl_split(dataset, example_ids, output_file, supplemental_dataset=None):
    ensure_dir(Path(output_file).parent)
    with open(output_file, "w") as f:
        for example_id in example_ids:
            item = dataset[int(example_id)]
            # Preserve all available source fields so eval metrics that depend on
            # paraphrases/perturbations can run on bucketized JSONL splits.
            payload = dict(item)
            
            if supplemental_dataset is not None:
                # Handle both Dictionary (dynamic index mapped pool_perturbed) or Dataset (single row lookup)
                if isinstance(supplemental_dataset, dict):
                    supplemental_item = supplemental_dataset.get(int(example_id))
                elif int(example_id) < len(supplemental_dataset):
                    supplemental_item = supplemental_dataset[int(example_id)]
                else:
                    supplemental_item = None
                    
                if supplemental_item is not None:
                    for key, value in supplemental_item.items():
                        if key not in payload:
                            payload[key] = value
                            
            if "paraphrased_answer" not in payload:
                payload["paraphrased_answer"] = payload.get("answer", "")
            if "perturbed_answer" not in payload:
                payload["perturbed_answer"] = payload.get("answer", "")
            if "paraphrased_question" not in payload:
                payload["paraphrased_question"] = payload.get("question", "")
                
            payload["index"] = int(example_id)
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    log_progress("save_jsonl", path=output_file, n_examples=len(example_ids))


def flatten_metric_rows(logs):
    rows = []
    for metric_name, metric_results in sorted(logs.items()):
        if not isinstance(metric_results, dict):
            continue
        agg_value = metric_results.get("agg_value", None)

        def emit_rows(subset_name, subset_results):
            value_by_index = subset_results.get("value_by_index", {}) if isinstance(subset_results, dict) else {}
            for example_id, stats in value_by_index.items():
                if stats is None:
                    rows.append({
                        "metric": metric_name,
                        "subset": subset_name,
                        "example_id": example_id,
                        "stat_key": "value",
                        "value": None,
                        "agg_value": agg_value,
                    })
                    continue
                if isinstance(stats, dict):
                    emitted = False
                    for stat_key, stat_value in stats.items():
                        if isinstance(stat_value, (int, float)) and not isinstance(stat_value, bool):
                            emitted = True
                            rows.append({
                                "metric": metric_name,
                                "subset": subset_name,
                                "example_id": example_id,
                                "stat_key": stat_key,
                                "value": float(stat_value),
                                "agg_value": agg_value,
                            })
                    if not emitted:
                        rows.append({
                            "metric": metric_name,
                            "subset": subset_name,
                            "example_id": example_id,
                            "stat_key": "value",
                            "value": None,
                            "agg_value": agg_value,
                        })
                elif isinstance(stats, (int, float)) and not isinstance(stats, bool):
                    rows.append({
                        "metric": metric_name,
                        "subset": subset_name,
                        "example_id": example_id,
                        "stat_key": "value",
                        "value": float(stats),
                        "agg_value": agg_value,
                    })

        if "value_by_index" in metric_results:
            emit_rows("default", metric_results)
        for subset_name, subset_results in metric_results.items():
            if subset_name in ("agg_value", "value_by_index"):
                continue
            if isinstance(subset_results, dict):
                emit_rows(subset_name, subset_results)
    return rows


def build_bucket_manifest(detailed_csv_path, run_dir, bucket_size, seed):
    log_progress("manifest_build_start", details_csv=detailed_csv_path, bucket_size=bucket_size)
    rows = load_csv_rows(detailed_csv_path)

    def resolve_metric_scores(metric_name, preferred_subsets=("baseline_pool", "forget", "default")):
        for subset_name in preferred_subsets:
            scores = extract_metric_map(rows, metric_name, subset_name=subset_name, stat_key="score")
            if scores:
                return scores, subset_name
        return {}, None

    grad_scores, grad_subset = resolve_metric_scores("mia_gradnorm")
    memo_scores, memo_subset = resolve_metric_scores("exact_memorization")

    if not grad_scores:
        raise RuntimeError(
            "Unable to find mia_gradnorm scores in TOFU_EVAL_DETAILS.csv for subsets "
            "'baseline_pool', 'forget' or 'default'."
        )
    if not memo_scores:
        raise RuntimeError(
            "Unable to find exact_memorization scores in TOFU_EVAL_DETAILS.csv for subsets "
            "'baseline_pool', 'forget' or 'default'."
        )

    common_ids = sorted(set(grad_scores) & set(memo_scores))
    if not common_ids:
        raise RuntimeError(
            "No overlapping examples found between gradnorm and exact memorization scores "
            f"(subsets used: mia_gradnorm={grad_subset}, exact_memorization={memo_subset})."
        )

    grad_values = [grad_scores[example_id] for example_id in common_ids]
    memo_values = [memo_scores[example_id] for example_id in common_ids]

    if torch.cuda.is_available() and len(common_ids) > 0:
        grad_t = torch.tensor(grad_values, dtype=torch.float32, device='cuda')
        memo_t = torch.tensor(memo_values, dtype=torch.float32, device='cuda')
        grad_median = float(torch.quantile(grad_t, 0.5).cpu().item())
        memo_median = float(torch.quantile(memo_t, 0.5).cpu().item())
    else:
        def median(values):
            ordered = sorted(values)
            middle = len(ordered) // 2
            if len(ordered) % 2 == 1:
                return float(ordered[middle])
            return float((ordered[middle - 1] + ordered[middle]) / 2.0)

        grad_median = median(grad_values)
        memo_median = median(memo_values)

    smoke_run = os.environ.get("SMOKE_RUN", "0") == "1"
    if smoke_run:
        bucket_size = min(bucket_size, len(common_ids))
        
    if len(common_ids) < bucket_size:
        raise RuntimeError(
            f"Only {len(common_ids)} overlapping examples available; cannot take top {bucket_size}."
        )

    def rank_scale(score_map, ids):
        ordered = sorted(ids, key=lambda i: (score_map[i], i))
        if len(ordered) == 1:
            return {ordered[0]: 0.5}
        denom = float(len(ordered) - 1)
        return {example_id: rank / denom for rank, example_id in enumerate(ordered)}

    grad_scaled = rank_scale(grad_scores, common_ids)
    memo_scaled = rank_scale(memo_scores, common_ids)

    # Joint scores over rank-scaled values in [0, 1] for each target quadrant.
    joint_scores = {
        "low_grad_low_memo": {
            i: (1.0 - grad_scaled[i]) + (1.0 - memo_scaled[i]) for i in common_ids
        },
        "low_grad_high_memo": {
            i: (1.0 - grad_scaled[i]) + memo_scaled[i] for i in common_ids
        },
        "high_grad_low_memo": {
            i: grad_scaled[i] + (1.0 - memo_scaled[i]) for i in common_ids
        },
        "high_grad_high_memo": {
            i: grad_scaled[i] + memo_scaled[i] for i in common_ids
        },
    }

    ranked_ids = {
        "low_grad_low_memo": sorted(
            common_ids,
            key=lambda i: (-joint_scores["low_grad_low_memo"][i], i),
        ),
        "low_grad_high_memo": sorted(
            common_ids,
            key=lambda i: (-joint_scores["low_grad_high_memo"][i], i),
        ),
        "high_grad_low_memo": sorted(
            common_ids,
            key=lambda i: (-joint_scores["high_grad_low_memo"][i], i),
        ),
        "high_grad_high_memo": sorted(
            common_ids,
            key=lambda i: (-joint_scores["high_grad_high_memo"][i], i),
        ),
    }

    selected = {
        bucket_name: ids[:bucket_size]
        for bucket_name, ids in ranked_ids.items()
    }

    manifest = {
        "seed": seed,
        "bucket_size": bucket_size,
        "thresholds": {
            "mia_gradnorm_median": grad_median,
            "exact_memorization_median": memo_median,
        },
        "all_counts": {name: len(ids) for name, ids in ranked_ids.items()},
        "selected_counts": {name: len(ids) for name, ids in selected.items()},
        "selection": {
            "strategy": "joint_rank_scaled",
            "weights": {"grad": 1.0, "memo": 1.0},
        },
        "selected_example_ids": selected,
    }

    # Save clean instance scores CSV as requested
    scores_csv_path = Path(run_dir) / "instance_scores.csv"
    with open(scores_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["example_id", "grad_norm", "memorization_score", "bucket"])
        
        # Mapping for easy lookup of assigned bucket
        id_to_bucket = {}
        for b_name, b_ids in selected.items():
            for i in b_ids:
                id_to_bucket[i] = b_name
                
        for i in common_ids:
            writer.writerow([
                i, 
                grad_scores.get(i), 
                memo_scores.get(i), 
                id_to_bucket.get(i, "None")
            ])
            
    log_progress("scores_csv_saved", path=scores_csv_path)

    write_json(Path(run_dir) / "split_manifest.json", manifest)
    log_progress(
        "manifest_build_done",
        n_common=len(common_ids),
        selected_per_bucket=bucket_size,
    )
    return manifest, grad_scores, memo_scores


def plot_bucket_scatter(grad_scores, memo_scores, manifest, output_file):
    colors = {
        "low_grad_low_memo": "#7f8c8d",
        "low_grad_high_memo": "#f39c12",
        "high_grad_low_memo": "#3498db",
        "high_grad_high_memo": "#e74c3c",
    }

    plt.figure(figsize=(8, 8))
    for bucket_name, example_ids in manifest["selected_example_ids"].items():
        x = [grad_scores[i] for i in example_ids]
        y = [memo_scores[i] for i in example_ids]
        plt.scatter(x, y, s=16, alpha=0.7, label=bucket_name, color=colors[bucket_name])

    plt.axvline(manifest["thresholds"]["mia_gradnorm_median"], linestyle="--", color="black", alpha=0.5)
    plt.axhline(manifest["thresholds"]["exact_memorization_median"], linestyle="--", color="black", alpha=0.5)
    plt.xlabel("mia_gradnorm")
    plt.ylabel("exact_memorization")
    plt.title("TOFU forget split buckets")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=160)
    plt.close()
    log_progress("save_plot", path=output_file)


def infer_access_key(dataset_name, dataset_cfg):
    access_key = getattr(dataset_cfg, "access_key", None)
    if access_key in ("forget", "holdout", "retain"):
        return access_key

    dataset_name_lower = str(dataset_name).lower()
    if "forget" in dataset_name_lower:
        return "forget"
    if "holdout" in dataset_name_lower or "test" in dataset_name_lower:
        return "holdout"
    if "retain" in dataset_name_lower:
        return "retain"
    return None


def rewrite_eval_datasets(eval_cfg, split_files, retain_logs_path):
    def recurse(node):
        if not hasattr(node, "items"):
            return

        if "retain_logs_path" in node:
            with open_dict(node):
                node.retain_logs_path = retain_logs_path

        if "datasets" in node and hasattr(node.datasets, "items"):
            for dataset_name, dataset_cfg in node.datasets.items():
                if not hasattr(dataset_cfg, "args") or not hasattr(dataset_cfg.args, "hf_args"):
                    continue
                access_key = infer_access_key(dataset_name, dataset_cfg)
                if access_key not in split_files:
                    continue
                hf_args = dataset_cfg.args.hf_args
                with open_dict(hf_args):
                    hf_args.path = "json"
                    hf_args.data_files = str(split_files[access_key])
                    hf_args.split = "train"
                    hf_args.pop("name", None)

        for value in node.values():
            recurse(value)

    recurse(eval_cfg)


def rewrite_data_datasets(data_cfg, split_files):
    with open_dict(data_cfg):
        if "forget" in data_cfg:
            for dataset_cfg in data_cfg.forget.values():
                hf_args = dataset_cfg.args.hf_args
                with open_dict(hf_args):
                    hf_args.path = "json"
                    hf_args.data_files = str(split_files["forget"])
                    hf_args.split = "train"
                    hf_args.pop("name", None)
        if "retain" in data_cfg:
            for dataset_cfg in data_cfg.retain.values():
                hf_args = dataset_cfg.args.hf_args
                with open_dict(hf_args):
                    hf_args.path = "json"
                    hf_args.data_files = str(split_files["retain"])
                    hf_args.split = "train"
                    hf_args.pop("name", None)
        if "test" in data_cfg:
            for dataset_cfg in data_cfg.test.values():
                hf_args = dataset_cfg.args.hf_args
                with open_dict(hf_args):
                    hf_args.path = "json"
                    hf_args.data_files = str(split_files["holdout"])
                    hf_args.split = "train"
                    hf_args.pop("name", None)


def make_split_files(run_dir, manifest, forget_dataset_name, holdout_split, retain_split):
    data_dir = Path(run_dir) / "data"
    ensure_dir(data_dir)

    # Always load the full training pool (4000) and full holdout pool (400)
    # This fulfills the request: "there is no single retain or forget set"
    # We treat it as one Universal Pool from which we slice based on mapped IDs.
    pool_source = load_tofu_split(forget_dataset_name)
    holdout_source = load_tofu_split(holdout_split)
    
    # Load perturbed version for auxiliary columns (e.g., paraphrased_answer)
    try:
        pool_perturbed = {}
        p_split = load_tofu_split(f"{forget_dataset_name}_perturbed")
        for item in p_split: pool_perturbed[int(item["index"])] = dict(item)
    except Exception as e:
        pool_perturbed = None
        logger.warning(f"Could not synthesize perturbed split for {forget_dataset_name}, falling back to non-perturbed pool.")

    smoke_run = os.environ.get("SMOKE_RUN", "0") == "1"

    split_files = {
        "forget": data_dir / "forget.jsonl",
        "holdout": data_dir / "holdout.jsonl",
        "retain": data_dir / "retain.jsonl",
    }
    bucket_retain_files = {}

    for bucket_name, ids in manifest["selected_example_ids"].items():
        if smoke_run: ids = ids[:2]
        
        # Forget Split for this bucket
        logger.info(f"Creating Forget split for bucket {bucket_name}: {len(ids)} forget examples")
        write_jsonl_split(
            pool_source,
            ids,
            data_dir / f"{bucket_name}.jsonl",
            supplemental_dataset=pool_perturbed,
        )
        
        # Retain Split for this bucket (The rest of the training pool)
        # Note: Even if 'forget_dataset_name' is not 'full' in a legacy run, 
        # for these benchmarks we always treat the pool as 0-3999.
        retain_ids = sorted(set(range(len(pool_source))) - set(ids))
        if smoke_run: retain_ids = retain_ids[:2]
        
        logger.info(f"Creating Retain split for bucket {bucket_name}: {len(retain_ids)} examples")
        bucket_retain_file = data_dir / f"{bucket_name}_retain.jsonl"
        write_jsonl_split(
            pool_source,
            retain_ids,
            bucket_retain_file,
            supplemental_dataset=pool_perturbed,
        )
        bucket_retain_files[bucket_name] = bucket_retain_file

    # Save the global holdout for metrics
    h_ids = list(range(len(holdout_source)))
    if smoke_run: h_ids = h_ids[:2]
    write_jsonl_split(holdout_source, h_ids, split_files["holdout"])

    # Default training inputs (for backward compatibility)
    split_files["forget_default"] = data_dir / f"{next(iter(manifest['selected_example_ids']))}.jsonl"
    split_files["bucket_retain_files"] = bucket_retain_files
    log_progress("split_files_ready", data_dir=data_dir)
    return split_files, bucket_retain_files


def run_baseline_eval(
    run_dir,
    model_path,
    forget_split,
    holdout_split,
    retain_split,
    seed,
    eval_batch_size,
):
    baseline_dir = Path(run_dir) / "baseline"
    ensure_dir(baseline_dir)
    log_progress("baseline_eval_start", output_dir=baseline_dir)

    smoke_run = os.environ.get("SMOKE_RUN", "0") == "1"

    # Create proper split files for baseline
    f_data = load_tofu_split(forget_split)
    h_data = load_tofu_split(holdout_split)
    r_data = load_tofu_split(retain_split)
    
    try:
        # User defined split variables get resolved directly to their perturbed HuggingFace equivalents
        f_split_name = f"{forget_split}_perturbed"
        f_perturbed = load_tofu_split(f_split_name)
    except Exception:
        f_perturbed = None
        
    try:
        h_perturbed = load_tofu_split(f"{holdout_split}_perturbed")
    except Exception:
        h_perturbed = None

    try:
        r_perturbed = load_tofu_split(f"{retain_split}_perturbed")
    except Exception:
        r_perturbed = None

    f_ids = list(range(len(f_data)))
    h_ids = list(range(len(h_data)))
    r_ids = list(range(len(r_data)))
    
    if smoke_run:
        logger.info("!!! SMOKE RUN: Limiting baseline to 20 samples (15 train, 5 holdout) !!!")
        f_ids = f_ids[:15]
        h_ids = h_ids[:5]
        r_ids = r_ids[:10]

    baseline_files = {
        "forget": baseline_dir / "baseline_forget.jsonl",
        "holdout": baseline_dir / "baseline_holdout.jsonl",
        "retain": baseline_dir / "baseline_retain.jsonl",
    }
    
    write_jsonl_split(f_data, f_ids, baseline_files["forget"], supplemental_dataset=f_perturbed)
    write_jsonl_split(h_data, h_ids, baseline_files["holdout"], supplemental_dataset=h_perturbed)
    write_jsonl_split(r_data, r_ids, baseline_files["retain"], supplemental_dataset=r_perturbed)

    cfg = compose_cfg(
        "eval",
        [
            f"eval.tofu.forget_split={baseline_files['forget'].stem}",
            f"eval.tofu.holdout_split={baseline_files['holdout'].stem}",
            f"eval.tofu.retain_split={retain_split}",
            "eval.tofu.question_key=question",
            f"seed={seed}",
            f"eval.tofu.batch_size={eval_batch_size}",
            f"paths.output_dir={baseline_dir}",
            f"model.model_args.pretrained_model_name_or_path={model_path}",
            f"model.tokenizer_args.pretrained_model_name_or_path={model_path}",
        ],
    )

    with open_dict(cfg):
        if "mia_reference" in cfg.eval.tofu.metrics:
            cfg.eval.tofu.metrics.mia_reference.reference_model_path = model_path
            
    # Rewrite datasets to use our local JSONL files
    rewrite_eval_datasets(cfg.eval, baseline_files, retain_logs_path=None)

    seed_everything(cfg.seed)
    ensure_gpu_ready()
    model, tokenizer = get_model(cfg.model)
    eval_cfgs = cfg.eval
    evaluators = get_evaluators(eval_cfgs, template_args=cfg.model.template_args, model=model, tokenizer=tokenizer)
    for evaluator in evaluators.values():
        evaluator.evaluate(
            model=model,
            output_dir=str(baseline_dir),
            template_args=cfg.model.template_args,
            tokenizer=tokenizer,
            overwrite=True, 
        )

    log_progress("baseline_eval_done", output_dir=baseline_dir)

    # In 'full' mode, we now have a single flat set of results.
    # We still ensure TOFU_EVAL.json has 'forget' and 'retain' keys for compatibility.
    eval_json = baseline_dir / "TOFU_EVAL.json"
    if eval_json.exists():
        try:
            with open(eval_json, "r") as f:
                eval_logs = json.load(f)
            
            updated = False
            for metric_name, metric_res in eval_logs.items():
                # Map the 'baseline_pool' results to 'forget' and 'retain' and 'holdout' for downstream tools
                if "baseline_pool" in metric_res:
                    pool_val = metric_res["baseline_pool"]
                    metric_res["forget"] = pool_val
                    metric_res["retain"] = pool_val
                    metric_res["holdout"] = pool_val
                    updated = True
            
            if updated:
                with open(eval_json, "w") as f:
                    json.dump(eval_logs, f, indent=4)
                logger.info(f"Broadcasted 'baseline_pool' results to forget/retain/holdout in {eval_json}")
        except Exception as e:
            logger.error(f"Post-processing baseline log failed: {e}")

    return baseline_dir

    try:
        del model
        del evaluators
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass

    return baseline_dir


def run_retain_reference_eval(
    run_dir,
    bucket_name,
    model_path,
    forget_split,
    holdout_split,
    retain_file,
    seed,
    eval_batch_size,
):
    reference_dir = Path(run_dir) / "baseline" / bucket_name / "retain_reference"
    ensure_dir(reference_dir)

    cfg = compose_cfg(
        "eval",
        [
            f"eval.tofu.forget_split={forget_split}",
            f"eval.tofu.holdout_split={holdout_split}",
            f"eval.tofu.retain_split={bucket_name}",
            "eval.tofu.question_key=question",
            f"seed={seed}",
            f"eval.tofu.batch_size={eval_batch_size}",
            f"paths.output_dir={reference_dir}",
            f"model.model_args.pretrained_model_name_or_path={model_path}",
            f"model.tokenizer_args.pretrained_model_name_or_path={model_path}",
        ],
    )

    with open_dict(cfg):
        if "mia_reference" in cfg.eval.tofu.metrics:
            cfg.eval.tofu.metrics.mia_reference.reference_model_path = model_path
        rewrite_eval_datasets(
            cfg.eval,
            {
                "retain": retain_file,
            },
            retain_logs_path=str(reference_dir / "TOFU_EVAL.json"),
        )

    seed_everything(cfg.seed)
    ensure_gpu_ready()
    model, tokenizer = get_model(cfg.model)
    eval_cfgs = cfg.eval
    evaluators = get_evaluators(eval_cfgs, template_args=cfg.model.template_args)
    for evaluator in evaluators.values():
        evaluator.evaluate(
            model=model,
            output_dir=str(reference_dir),
            template_args=cfg.model.template_args,
            tokenizer=tokenizer,
            overwrite=False,
        )

    return reference_dir / "TOFU_EVAL.json"


def create_mapped_performance_report(baseline_csv, unlearn_csv, output_csv):
    """Joins baseline scores with unlearned results on example_id and metric."""
    if not baseline_csv.exists() or not unlearn_csv.exists():
        logger.warning(f"Skipping mapped report: missing {baseline_csv} or {unlearn_csv}")
        return

    try:
        import csv

        # Load baseline scores: {(metric, example_id, stat_key): value}
        # In the new Unified Pool, everything is labeled under 'baseline_pool' but mapped by IDs.
        baseline_scores = {}
        with open(baseline_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["metric"], row["example_id"], row.get("stat_key", "value"))
                baseline_scores[key] = row["value"]

        # Load unlearned results and join
        mapped_rows = []
        with open(unlearn_csv, "r") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                metric = row["metric"]
                eid = row["example_id"]
                stat_key = row.get("stat_key", "value")
                subset = row["subset"]
                unlearn_val = row["value"]
                
                # Fetch baseline value using the same metric, example_id, and stat_key from the universal pool
                baseline_val = baseline_scores.get((metric, eid, stat_key))
                
                try:
                    delta = float(unlearn_val) - float(baseline_val) if unlearn_val and baseline_val else None
                except Exception:
                    delta = None

                mapped_rows.append({
                    "metric": metric,
                    "subset": subset,
                    "example_id": eid,
                    "stat_key": stat_key,
                    "baseline_value": baseline_val,
                    "unlearned_value": unlearn_val,
                    "delta": delta
                })

        if mapped_rows:
            ensure_dir(output_csv.parent)
            fieldnames = ["metric", "subset", "example_id", "stat_key", "baseline_value", "unlearned_value", "delta"]
            with open(output_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(mapped_rows)
            logger.info(f"Saved mapped performance report to {output_csv}")

    except Exception as e:
        logger.error(f"Failed to create mapped performance report: {e}")


def aggregate_bucket_baseline(baseline_details_csv, bucket_forget_ids, bucket_retain_ids, holdout_ids, output_dir):
    """
    Synthesizes a bucket-specific TOFU_EVAL.json by re-aggregating global foundation scores 
    based on the specific partitioning of this bucket.
    """
    if not baseline_details_csv.exists():
        logger.warning(f"Cannot generate bucket baseline: {baseline_details_csv} missing.")
        return None

    try:
        import csv
        from collections import defaultdict
        
        # Load all detailed scores from foundation baseline
        # Format: metric -> subset_type -> example_id -> score
        # Note: In baseline, subsets are 'forget' (all 4000) and 'holdout' (400)
        global_scores = defaultdict(lambda: defaultdict(dict))
        
        with open(baseline_details_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                metric = row["metric"]
                eid = int(row["example_id"])
                val = float(row["value"]) if row["value"] not in (None, "") else None
                global_scores[metric][row["subset"]][eid] = val

        bucket_ids_set = set(bucket_forget_ids)
        retain_ids_set = set(bucket_retain_ids)
        holdout_ids_set = set(holdout_ids)

        bucket_baseline = {}
        
        for metric, subsets in global_scores.items():
            # Extract scores for the new bucket-specific partitions
            # Prefer 'baseline_pool' (new unified flat structure) or fallback to 'forget'/'holdout'
            
            # Map training examples (Forget + Retain)
            train_source = subsets.get("baseline_pool", subsets.get("forget", {}))
            forget_scores = [train_source[eid] for eid in bucket_forget_ids if eid in train_source]
            retain_scores = [train_source[eid] for eid in bucket_retain_ids if eid in train_source]
            
            # Map holdout examples
            holdout_source = subsets.get("baseline_pool", subsets.get("holdout", {}))
            holdout_scores = [holdout_source[eid] for eid in holdout_ids if eid in holdout_source]

            def calc_agg(scores):
                valid = [s for s in scores if s is not None]
                return sum(valid) / len(valid) if valid else None

            def create_subset_dict(scores, ids):
                # Routing logic to find the correct score source for these IDs
                val_by_index = {}
                for eid in ids:
                    source = train_source if (eid in bucket_ids_set or eid in retain_ids_set) else holdout_source
                    if eid in source:
                        val_by_index[str(eid)] = {"score": source[eid]}
                return {
                    "agg_value": calc_agg(scores),
                    "value_by_index": val_by_index
                }

            bucket_baseline[metric] = {
                "forget": create_subset_dict(forget_scores, bucket_forget_ids),
                "retain": create_subset_dict(retain_scores, bucket_retain_ids),
                "holdout": create_subset_dict(holdout_scores, holdout_ids),
            }
            
            # Detect MIA metrics and calculate ROC-AUC
            if "mia" in metric.lower():
                def compute_auc(mem, non_mem):
                    v_mem = [s for s in mem if s is not None]
                    v_non_mem = [s for s in non_mem if s is not None]
                    if not v_mem or not v_non_mem: return 0.5
                    y_true = [0] * len(v_mem) + [1] * len(v_non_mem)
                    y_score = v_mem + v_non_mem
                    try:
                        return float(roc_auc_score(y_true, y_score))
                    except Exception:
                        return 0.5

                bucket_baseline[metric]["auc"] = compute_auc(forget_scores, holdout_scores)
                bucket_baseline[metric]["auc_retain"] = compute_auc(forget_scores, retain_scores)
                # Primary aggregate for MIA is AUC
                bucket_baseline[metric]["agg_value"] = bucket_baseline[metric]["auc"]
                logger.info(f"Computed Baseline ROC-AUC for {metric}: {bucket_baseline[metric]['agg_value']}")
            else:
                # Global aggregate for non-MIA metrics defaults to forget mean
                bucket_baseline[metric]["agg_value"] = bucket_baseline[metric]["forget"]["agg_value"]

        output_json = Path(output_dir) / "TOFU_EVAL.json"
        ensure_dir(output_json.parent)
        with open(output_json, "w") as f:
            json.dump(bucket_baseline, f, indent=4)
        
        # Save bucket-specific DETAILS.csv
        output_csv = Path(output_dir) / "TOFU_EVAL_DETAILS.csv"
        fieldnames = ["metric", "subset", "example_id", "stat_key", "value", "agg_value"]
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for metric, m_data in bucket_baseline.items():
                for subset_name, s_data in m_data.items():
                    if subset_name in ["agg_value", "auc", "auc_retain"]: continue
                    agg_val = s_data["agg_value"]
                    for eid, e_data in s_data["value_by_index"].items():
                        writer.writerow({
                            "metric": metric,
                            "subset": subset_name,
                            "example_id": eid,
                            "stat_key": "value",
                            "value": e_data["score"],
                            "agg_value": agg_val
                        })

        logger.info(f"Generated bucket-specific baseline reference at {output_json} and {output_csv}")
        return output_json

    except Exception as e:
        logger.error(f"Failed to aggregate bucket baseline: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_bucket_training(
    run_dir,
    bucket_name,
    bucket_file,
    retain_file,
    retain_logs_path,
    model_path,
    method_name,
    seed,
    save_steps,
    forget_split,
    holdout_split,
    retain_split,
    eval_batch_size,
    max_steps=None,
    force_train=False,
    metrics_only=False,
):
    bucket_dir = Path(run_dir) / method_name / bucket_name
    ensure_dir(bucket_dir)

    if not force_train and is_bucket_method_complete(bucket_dir):
        log_progress("bucket_train_skip_complete", method=method_name, bucket=bucket_name, bucket_dir=bucket_dir)
        return

    log_progress("bucket_train_start", method=method_name, bucket=bucket_name, bucket_dir=bucket_dir)
    log_vram_snapshot("iter_start", method=method_name, bucket=bucket_name)

    cfg = compose_cfg(
        "train",
        [
            "experiment=unlearn/tofu/default.yaml",
            f"trainer={method_name}",
            f"task_name=tofu_{bucket_name}_{method_name}",
            f"paths.output_dir={bucket_dir}",
            f"+seed={seed}",
            f"forget_split={forget_split}",
            f"holdout_split={holdout_split}",
            f"retain_split={retain_split}",
            f"eval.tofu.batch_size={eval_batch_size}",
            f"model.model_args.pretrained_model_name_or_path={model_path}",
            f"model.tokenizer_args.pretrained_model_name_or_path={model_path}",
        ],
    )

    with open_dict(cfg):
        if "mia_reference" in cfg.eval.tofu.metrics:
            cfg.eval.tofu.metrics.mia_reference.reference_model_path = model_path
        rewrite_data_datasets(
            cfg.data,
            {
                "forget": bucket_file,
                "retain": retain_file,
                "holdout": bucket_file.parent / "holdout.jsonl",
            },
        )
        rewrite_eval_datasets(
            cfg.eval,
            {
                "forget": bucket_file,
                "retain": retain_file,
                "holdout": bucket_file.parent / "holdout.jsonl",
            },
            retain_logs_path=str(retain_logs_path),
        )

        cfg.trainer.args.output_dir = str(bucket_dir)
        cfg.trainer.args.logging_dir = str(bucket_dir / "logs")
        cfg.trainer.args.save_strategy = "steps"
        cfg.trainer.args.save_steps = save_steps
        if max_steps is not None:
            cfg.trainer.args.max_steps = max_steps
        cfg.trainer.args.save_total_limit = 2
        cfg.trainer.args.eval_strategy = "no"
        cfg.trainer.args.eval_on_start = False
        cfg.trainer.args.logging_steps = min(save_steps, 10)
        cfg.trainer.args.do_eval = True
        cfg.trainer.args.do_train = True
        # Unlearning trainers consume nested keys like inputs['forget']/inputs['retain'].
        # Keep all columns to avoid HF Trainer stripping them and producing empty batches.
        cfg.trainer.args.remove_unused_columns = False
        if "method_args" not in cfg.trainer:
            cfg.trainer.method_args = OmegaConf.create({})
        if method_name == "PDU" and "retain_loss_eps" not in cfg.trainer.method_args:
            cfg.trainer.method_args.retain_loss_eps = 0.0

    seed_everything(cfg.trainer.args.seed)
    ensure_gpu_ready()
    log_vram_snapshot("pre_model_load", method=method_name, bucket=bucket_name)
    model, tokenizer = get_model(cfg.model)
    data = get_data(cfg.data, mode="unlearn", tokenizer=tokenizer, template_args=cfg.model.template_args)
    collator = get_collators(cfg.collator, tokenizer=tokenizer)
    evaluators = get_evaluators(cfg.eval, template_args=cfg.model.template_args, model=model, tokenizer=tokenizer)

    trainer, trainer_args = load_trainer(
        trainer_cfg=cfg.trainer,
        model=model,
        train_dataset=data.get("train", None),
        eval_dataset=None if metrics_only else data.get("eval", None),
        processing_class=tokenizer,
        data_collator=collator,
        evaluators=evaluators,
        template_args=cfg.model.template_args,
    )
    log_vram_snapshot("post_trainer_init", method=method_name, bucket=bucket_name)

    # HF's get_last_checkpoint only checks folder naming; resume only from complete checkpoints.
    existing_checkpoint = get_last_valid_trainer_checkpoint(str(bucket_dir))
    import sys
    try:
        if existing_checkpoint:
            log_progress("bucket_train_resume", method=method_name, bucket=bucket_name, checkpoint=existing_checkpoint)
            trainer.train(resume_from_checkpoint=existing_checkpoint)
        else:
            trainer.train()
        log_vram_snapshot("post_train", method=method_name, bucket=bucket_name)
    except Exception as e:
        import traceback
        print(f"\n==========================================")
        print(f"[FATAL TOP-LEVEL CRASH] trainer.train() threw an exception!!")
        print(f"{e}")
        traceback.print_exc()
        print(f"==========================================\n")
        sys.stdout.flush()
        raise e
    trainer.save_state()
    log_progress("save_state", method=method_name, bucket=bucket_name, output_dir=trainer_args.output_dir)
    trainer.save_model(trainer_args.output_dir)
    log_progress("save_model", method=method_name, bucket=bucket_name, output_dir=trainer_args.output_dir)
    trainer.evaluate(metric_key_prefix="after")

    eval_logs_path = Path(trainer_args.output_dir) / f"checkpoint-{trainer.state.global_step}" / "evals" / "TOFU_EVAL.json"
    assert_eval_metrics_complete(
        eval_logs_path,
        list(cfg.eval.tofu.metrics.keys()),
    )

    log_progress("bucket_eval_after_done", method=method_name, bucket=bucket_name)

    # Generate the mapped performance report (Baseline vs Unlearned)
    checkpoint_dir = Path(trainer_args.output_dir) / f"checkpoint-{trainer.state.global_step}"
    unlearn_details_csv = checkpoint_dir / "evals" / "TOFU_EVAL_DETAILS.csv"
    # Use the bucket-specific baseline created earlier
    bucket_baseline_dir = Path(run_dir) / "baseline" / bucket_name
    baseline_details_csv = bucket_baseline_dir / "TOFU_EVAL_DETAILS.csv"
    mapped_out_csv = Path(bucket_dir) / "mapped_performance.csv"
    create_mapped_performance_report(baseline_details_csv, unlearn_details_csv, mapped_out_csv)

    write_json(
        Path(bucket_dir) / "RUN_COMPLETE.json",
        {
            "completed_at": utc_now(),
            "method": method_name,
            "bucket": bucket_name,
            "output_dir": str(bucket_dir),
            "last_checkpoint": get_last_checkpoint(str(bucket_dir)),
        },
    )
    log_progress("bucket_train_done", method=method_name, bucket=bucket_name, bucket_dir=bucket_dir)

    # Prevent massive GPU memory leaks across different methods and buckets in the orchestrator loop
    try:
        del model
        del trainer
        del evaluators
        del collator
        del data
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
        log_vram_snapshot("post_cleanup", method=method_name, bucket=bucket_name)
    except Exception:
        pass


def plot_metric_comparison_from_csv(before_csv, after_csv, output_file, metric_name, subset_name="forget", stat_key="score"):
    def load_metric(path):
        scores = {}
        for row in load_csv_rows(path):
            if row["metric"] != metric_name:
                continue
            if row["subset"] != subset_name:
                continue
            if row["stat_key"] != stat_key:
                continue
            example_id = to_int(row["example_id"])
            value = to_float(row["value"])
            if example_id is not None and value is not None:
                scores[example_id] = value
        return scores

    before_scores = load_metric(before_csv)
    after_scores = load_metric(after_csv)
    common_ids = sorted(set(before_scores) & set(after_scores))
    if not common_ids:
        return

    plt.figure(figsize=(7, 7))
    xs = [before_scores[i] for i in common_ids]
    ys = [after_scores[i] for i in common_ids]
    plt.scatter(xs, ys, s=12, alpha=0.6)
    low = min(xs + ys)
    high = max(xs + ys)
    plt.plot([low, high], [low, high], "r--", alpha=0.7)
    plt.xlabel("Before Unlearning")
    plt.ylabel("After Unlearning")
    plt.title(f"{metric_name} Comparison")
    plt.tight_layout()
    plt.savefig(output_file, dpi=160)
    plt.close()
    log_progress("save_plot", path=output_file, metric=metric_name, subset=subset_name)


def validate_methods(methods):
    if not methods:
        raise RuntimeError("No methods provided. Pass at least one trainer method using --methods.")

    missing_trainer_cfg = []
    for method_name in methods:
        if method_name in UNSUPPORTED_PIPELINE_METHODS:
            raise RuntimeError(
                f"Unsupported method '{method_name}' for this pipeline: "
                f"{UNSUPPORTED_PIPELINE_METHODS[method_name]}"
            )
        trainer_cfg_path = CONFIG_DIR / "trainer" / f"{method_name}.yaml"
        if not trainer_cfg_path.exists():
            missing_trainer_cfg.append((method_name, trainer_cfg_path))

    if missing_trainer_cfg:
        details = ", ".join(f"{name} ({path})" for name, path in missing_trainer_cfg)
        raise RuntimeError(f"Missing trainer config(s): {details}")


def main():
    signal.signal(signal.SIGTERM, _sigterm_handler)

    # Cap CPU thread pools so most compute pressure stays on GPU.
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)

    parser = argparse.ArgumentParser(description="TOFU bucketed unlearning pipeline")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=DEFAULT_BASELINE_MODEL)
    parser.add_argument("--forget-split", type=str, default=DEFAULT_FORGET_SPLIT)
    parser.add_argument("--holdout-split", type=str, default=DEFAULT_HOLDOUT_SPLIT)
    parser.add_argument("--retain-split", type=str, default=DEFAULT_RETAIN_SPLIT)
    parser.add_argument("--bucket-size", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    parser.add_argument("--stage", choices=["baseline", "split", "train", "plot", "all"], default="all")
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=None, help="Limit maximum training steps (useful for smoke runs)")
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--force-baseline", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--metrics-only", action="store_true", help="Skip dataset evaluation, only run custom metrics")
    parser.add_argument("--buckets", nargs="*", default=None, help="Specific bucket names to process (e.g. high_grad_high_memo)")
    args = parser.parse_args()

    validate_methods(args.methods)

    run_id = args.run_id or f"tofu_buckets_{args.seed}"
    run_dir = REPO_ROOT / "results" / run_id
    ensure_dir(run_dir)
    init_progress_log(run_dir, run_id)
    log_progress(
        "run_args",
        stage=args.stage,
        bucket_size=args.bucket_size,
        forget_split=args.forget_split,
        holdout_split=args.holdout_split,
        retain_split=args.retain_split,
        methods=" ".join(args.methods),
    )

    baseline_dir = Path(run_dir) / "baseline"
    baseline_csv = baseline_dir / "TOFU_EVAL_DETAILS.csv"
    baseline_json = baseline_dir / "TOFU_EVAL.json"

    if args.stage in ("baseline", "split", "train", "plot", "all"):
        if args.force_baseline or not is_baseline_complete(baseline_dir, args.forget_split):
            set_stage("baseline_eval")
            run_baseline_eval(
                run_dir=run_dir,
                model_path=args.model_path,
                forget_split=args.forget_split,
                holdout_split=args.holdout_split,
                retain_split=args.retain_split,
                seed=args.seed,
                eval_batch_size=args.eval_batch_size,
            )
        else:
            log_progress("baseline_reuse", baseline_dir=baseline_dir)

    manifest = None
    grad_scores = None
    memo_scores = None
    split_files = None
    bucket_retain_files = None
    bucket_retain_reference_logs = None
    if args.stage in ("split", "train", "plot", "all"):
        set_stage("build_bucket_manifest")
        manifest, grad_scores, memo_scores = build_bucket_manifest(
            baseline_csv,
            run_dir,
            bucket_size=args.bucket_size,
            seed=args.seed,
        )
        set_stage("make_split_files")
        split_files, bucket_retain_files = make_split_files(
            run_dir,
            manifest,
            forget_dataset_name=args.forget_split,
            holdout_split=args.holdout_split,
            retain_split=args.retain_split,
        )
        # Generate split-specific baseline reference logs for each bucket.
        # This fulfills the requirement to calculate metrics differently for each split.
        bucket_retain_reference_logs = {}
        for bucket_name, b_forget_ids in manifest["selected_example_ids"].items():
            # The retain set for 'full' mode is everything else in the pool
            b_retain_ids = sorted(set(range(len(grad_scores))) - set(b_forget_ids))
            holdout_ids = sorted(grad_scores.keys()) # All IDs in baseline details but not in pool are holdout? No, grad_scores only has training pool.
            
            # Re-read the CSV to properly get holdout IDs
            all_baseline_ids = set()
            h_ids = []
            with open(baseline_csv, "r") as f:
                r = csv.DictReader(f)
                for row in r:
                    if row["subset"] == "holdout":
                        h_ids.append(int(row["example_id"]))
            h_ids = sorted(list(set(h_ids)))

            bucket_baseline_dir = Path(run_dir) / "baseline" / bucket_name
            log_path = aggregate_bucket_baseline(
                baseline_csv, 
                b_forget_ids, 
                b_retain_ids, 
                h_ids, 
                bucket_baseline_dir
            )
            bucket_retain_reference_logs[bucket_name] = log_path
        write_json(Path(run_dir) / "run_manifest.json", {
            "run_id": run_id,
            "model_path": args.model_path,
            "forget_split": args.forget_split,
            "holdout_split": args.holdout_split,
            "retain_split": args.retain_split,
            "methods": args.methods,
            "bucket_size": args.bucket_size,
            "seed": args.seed,
        })
        log_progress("run_manifest_saved", path=Path(run_dir) / "run_manifest.json")
        plot_bucket_scatter(
            grad_scores,
            memo_scores,
            manifest,
            Path(run_dir) / "bucket_scatter.png",
        )

    if args.stage in ("train", "all"):
        baseline_retain_logs = baseline_dir / "TOFU_EVAL.json"
        if not baseline_retain_logs.exists():
            raise RuntimeError(f"Baseline retain logs not found at {baseline_retain_logs}")
        log_progress("train_phase_start", baseline_retain_logs=baseline_retain_logs)

        for bucket_name, example_ids in manifest["selected_example_ids"].items():
            if args.buckets and bucket_name not in args.buckets:
                continue
            bucket_file = Path(run_dir) / "data" / f"{bucket_name}.jsonl"
            if not bucket_file.exists():
                raise RuntimeError(f"Missing bucket split file: {bucket_file}")
            retain_file = bucket_retain_files[bucket_name]
            if not retain_file.exists():
                raise RuntimeError(f"Missing bucket retain file: {retain_file}")
            retain_logs_path = bucket_retain_reference_logs[bucket_name]
            for method_name in args.methods:
                bucket_dir = Path(run_dir) / method_name / bucket_name
                if not args.force_train and is_bucket_method_complete(bucket_dir):
                    log_progress("bucket_method_reuse", method=method_name, bucket=bucket_name, bucket_dir=bucket_dir)
                    continue
                set_stage(f"train:{method_name}:{bucket_name}")
                run_bucket_training(
                    run_dir=run_dir,
                    bucket_name=bucket_name,
                    bucket_file=bucket_file,
                    retain_file=retain_file,
                    retain_logs_path=retain_logs_path,
                    model_path=args.model_path,
                    method_name=method_name,
                    seed=args.seed,
                    save_steps=args.save_steps,
                    forget_split=args.forget_split,
                    holdout_split=args.holdout_split,
                    retain_split=args.retain_split,
                    eval_batch_size=args.eval_batch_size,
                    max_steps=args.max_steps,
                    force_train=args.force_train,
                    metrics_only=args.metrics_only,
                )

    if args.stage in ("plot", "all"):
        set_stage("plot")
        for bucket_name in manifest["selected_example_ids"].keys():
            bucket_dir = Path(run_dir) / args.methods[0] / bucket_name
            before_csv = baseline_csv
            after_candidates = sorted((bucket_dir).glob("checkpoint-*/evals/TOFU_EVAL_DETAILS.csv"))
            if not getattr(before_csv, "exists", lambda: False)() or not after_candidates:
                log_progress("plot_skip_missing", bucket=bucket_name)
                continue
            after_csv = after_candidates[-1]
            plot_metric_comparison_from_csv(
                before_csv,
                after_csv,
                Path(run_dir) / f"{bucket_name}_exact_memorization_before_after.png",
                "exact_memorization",
            )
            plot_metric_comparison_from_csv(
                before_csv,
                after_csv,
                Path(run_dir) / f"{bucket_name}_mia_gradnorm_before_after.png",
                "mia_gradnorm",
            )

    write_json(
        Path(run_dir) / "RUN_COMPLETE.json",
        {
            "completed_at": utc_now(),
            "run_id": run_id,
            "stage": args.stage,
            "methods": args.methods,
            "model_path": args.model_path,
        },
    )
    set_stage("complete")
    log_progress("run_complete", run_id=run_id)


if __name__ == "__main__":
    main()