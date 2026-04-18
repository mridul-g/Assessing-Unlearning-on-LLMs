#!/bin/bash
set -euo pipefail

# Run from repository root
cd "$(dirname "$0")/.."

# Cache paths (override through environment when needed)
export HF_HOME="${HF_HOME:-/data/mridulg22/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TORCH_HOME="${TORCH_HOME:-/data/mridulg22/.cache/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/data/mridulg22/.cache/triton}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/data/mridulg22/.cache/pip}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps-$USER}"
export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-/tmp/nvidia-log-$USER}"

# Keep CPU pressure low unless user overrides it.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Defaults are chosen for single-GPU execution.
RUN_ID="${RUN_ID:-tofu_bucket_$(date +"%Y%m%d_%H%M%S")}" 
STAGE="${STAGE:-all}"
MODEL_PATH="${MODEL_PATH:-open-unlearning/tofu_Llama-3.2-1B-Instruct_full}"
FORGET_SPLIT="${FORGET_SPLIT:-full}"
HOLDOUT_SPLIT="${HOLDOUT_SPLIT:-holdout10}"
RETAIN_SPLIT="${RETAIN_SPLIT:-retain90}"
BUCKET_SIZE="${BUCKET_SIZE:-400}"
SEED="${SEED:-42}"
SAVE_STEPS="${SAVE_STEPS:-50}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
FORCE_BASELINE="${FORCE_BASELINE:-0}"
METRICS_ONLY="${METRICS_ONLY:-0}"

# Comma-separated or space-separated list supported,
# e.g. METHODS="GradAscent,NPO,RMU" or METHODS="GradAscent NPO RMU"
# METHODS="${METHODS:-GradAscent,GradDiff,NPO,SimNPO,RMU,UNDIAL,SatImp,WGA,CEU,PDU}"
METHODS="${METHODS:-GradAscent,GradDiff,NPO,SimNPO,RMU,UNDIAL,SatImp, }".  # without ceu and wga, pdu

# CLI overrides (optional)
while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-id)
            RUN_ID="$2"; shift 2 ;;
        --stage)
            STAGE="$2"; shift 2 ;;
        --model-path)
            MODEL_PATH="$2"; shift 2 ;;
        --forget-split)
            FORGET_SPLIT="$2"; shift 2 ;;
        --holdout-split)
            HOLDOUT_SPLIT="$2"; shift 2 ;;
        --retain-split)
            RETAIN_SPLIT="$2"; shift 2 ;;
        --bucket-size)
            BUCKET_SIZE="$2"; shift 2 ;;
        --seed)
            SEED="$2"; shift 2 ;;
        --save-steps)
            SAVE_STEPS="$2"; shift 2 ;;
        --eval-batch-size)
            EVAL_BATCH_SIZE="$2"; shift 2 ;;
        --methods)
            METHODS="$2"; shift 2 ;;
        --force-train)
            FORCE_TRAIN=1; shift 1 ;;
        --force-baseline)
            FORCE_BASELINE=1; shift 1 ;;
        --metrics-only)
            METRICS_ONLY=1; shift 1 ;;
        --smoke-run)
            export SMOKE_RUN=1; shift 1 ;;
        --offline)
            export TRANSFORMERS_OFFLINE=1
            export HF_HUB_OFFLINE=1
            export HF_DATASETS_OFFLINE=1
            echo "!!! OFFLINE MODE ENABLED !!!"
            shift 1 ;;
        --buckets)
            BUCKETS="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1"
            exit 1 ;;
    esac
done

METHODS_NORM="${METHODS//,/ }"
read -r -a METHOD_ARRAY <<< "$METHODS_NORM"

if [[ ${#METHOD_ARRAY[@]} -eq 0 ]]; then
    echo "No methods provided in METHODS=..."
    exit 1
fi

echo "Running TOFU bucket pipeline"
echo "  RUN_ID       : $RUN_ID"
echo "  STAGE        : $STAGE"
echo "  MODEL_PATH   : $MODEL_PATH"
echo "  SPLITS       : $FORGET_SPLIT / $HOLDOUT_SPLIT / $RETAIN_SPLIT"
echo "  BUCKET_SIZE  : $BUCKET_SIZE"
echo "  SEED         : $SEED"
echo "  SAVE_STEPS   : $SAVE_STEPS"
echo "  EVAL_BATCH_SIZE: $EVAL_BATCH_SIZE"
echo "  FORCE_TRAIN  : $FORCE_TRAIN"
echo "  FORCE_BASELINE: $FORCE_BASELINE"
echo "  METHODS      : ${METHOD_ARRAY[*]}"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d >/dev/null 2>&1 || true

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("GPU is required, but torch.cuda.is_available() is False")
print(f"GPU ready: {torch.cuda.get_device_name(0)}")
PY

CMD=(
    python src/tofu_bucket_pipeline.py
    --run-id "$RUN_ID"
    --stage "$STAGE"
    --model-path "$MODEL_PATH"
    --forget-split "$FORGET_SPLIT"
    --holdout-split "$HOLDOUT_SPLIT"
    --retain-split "$RETAIN_SPLIT"
    --bucket-size "$BUCKET_SIZE"
    --seed "$SEED"
    --save-steps "$SAVE_STEPS"
    --eval-batch-size "$EVAL_BATCH_SIZE"
    --methods
)

for method in "${METHOD_ARRAY[@]}"; do
    CMD+=("$method")
done

if [[ "$FORCE_TRAIN" == "1" ]]; then
    CMD+=(--force-train)
fi

if [[ "$FORCE_BASELINE" == "1" ]]; then
    CMD+=(--force-baseline)
fi

if [[ -v BUCKETS ]]; then
    CMD+=(--buckets)
    # Split comma separated buckets if provided
    BUCKETS_NORM="${BUCKETS//,/ }"
    read -r -a BUCKET_ARRAY <<< "$BUCKETS_NORM"
    for b in "${BUCKET_ARRAY[@]}"; do
        CMD+=("$b")
    done
fi

if [[ "${SMOKE_RUN:-0}" == "1" ]]; then
    echo "!!! SMOKE RUN ENABLED !!!"
    BUCKET_SIZE=20
    SAVE_STEPS=5
    CMD+=(--max-steps 5)
fi

if [[ "$METRICS_ONLY" == "1" ]]; then
    CMD+=(--metrics-only)
fi

exec setsid env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${CMD[@]}"
