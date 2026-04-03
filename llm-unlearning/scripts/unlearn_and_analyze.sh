#!/bin/bash

# Navigate to project root
cd "$(dirname "$0")/.."

export HF_HOME="/data/mridulg22/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="/data/mridulg22/.cache/huggingface/hub"
export TRANSFORMERS_CACHE="/data/mridulg22/.cache/huggingface/transformers"
export TORCH_HOME="/data/mridulg22/.cache/torch"
export TRITON_CACHE_DIR="/data/mridulg22/.cache/triton"
export PIP_CACHE_DIR="/data/mridulg22/.cache/pip"


# Configuration
# Allow RUN_ID to be passed as an argument or environment variable, default to timestamp
RUN_ID=${1:-${RUN_ID:-$(date +"%Y%m%d_%H%M%S")}}
RESULTS_DIR="results/run_${RUN_ID}"
echo "Using Results Directory: ${RESULTS_DIR}"
mkdir -p "${RESULTS_DIR}"

MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

# Categorization
GRAD_TRAINERS=("GradAscent" "GradDiff" "NPO" "DPO")
NON_GRAD_TRAINERS=("RMU" "SatImp")

# Benchmarks and Models
MUSE_MODEL="Llama-2-7b-hf"
MUSE_HF_ORG="muse-bench"
TOFU_MODEL="Llama-3.2-1B-Instruct"
TOFU_HF_ORG="open-unlearning"

# #########################################################
# #################### MUSE Unlearning ####################
# #########################################################

MUSE_DATA_SPLITS=("News" "Books")

for data_split in "${MUSE_DATA_SPLITS[@]}"; do
    # 1. Baseline (Before Unlearning)
    echo "Running baseline evaluation for MUSE ${data_split}..."
    MUSE_TARGET_PATH="${MUSE_HF_ORG}/MUSE-${data_split}_target"
    BASELINE_EVAL="${RESULTS_DIR}/muse/baseline/${data_split}/MUSE_EVAL.json"
    
    if [ ! -f "$BASELINE_EVAL" ]; then
        CUDA_VISIBLE_DEVICES=0 python src/eval.py \
            experiment=eval/muse/default.yaml \
            +experiment/eval=comprehensive \
            data_split=${data_split} \
            model=${MUSE_MODEL} \
            model.model_args.pretrained_model_name_or_path=${MUSE_TARGET_PATH} \
            task_name="muse_baseline_${data_split}" \
            paths.output_dir="${RESULTS_DIR}/muse/baseline/${data_split}"
    else
        echo "Baseline evaluation for MUSE ${data_split} already exists. Skipping."
    fi

    # 2. Unlearn and Eval for each method
    for trainer in "${GRAD_TRAINERS[@]}" "${NON_GRAD_TRAINERS[@]}"; do
        echo "Processing ${trainer} for MUSE ${data_split}..."
        
        task_name="muse_${MUSE_MODEL}_${data_split}_${trainer}"
        TRAIN_DIR="${RESULTS_DIR}/muse/${trainer}/${data_split}/checkpoints"
        AFTER_EVAL="${RESULTS_DIR}/muse/${trainer}/${data_split}/evals/MUSE_EVAL.json"

        # Unlearn check
        if [ ! -d "$TRAIN_DIR" ]; then
            CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
                src/train.py --config-name=unlearn.yaml \
                experiment=unlearn/muse/default.yaml \
                model=${MUSE_MODEL} \
                model.model_args.pretrained_model_name_or_path=${MUSE_TARGET_PATH} \
                data_split=${data_split} \
                trainer=${trainer} \
                task_name=${task_name} \
                paths.output_dir="${TRAIN_DIR}"
        else
            echo "Unlearning for ${trainer} (MUSE ${data_split}) already exists. Skipping."
        fi

        # Eval check
        if [ ! -f "$AFTER_EVAL" ]; then
            CUDA_VISIBLE_DEVICES=0 python src/eval.py \
                experiment=eval/muse/default.yaml \
                +experiment/eval=comprehensive \
                data_split=${data_split} \
                model=${MUSE_MODEL} \
                model.model_args.pretrained_model_name_or_path="${TRAIN_DIR}" \
                task_name="${task_name}_after" \
                paths.output_dir="${RESULTS_DIR}/muse/${trainer}/${data_split}/evals"
        else
            echo "Post-unlearning evaluation for ${trainer} (MUSE ${data_split}) already exists. Skipping."
        fi
    done
done

# #########################################################
# #################### TOFU Unlearning ####################
# #########################################################

TOFU_FORGET_SPLITS=("forget01" "forget05" "forget10")
TOFU_RETAIN_SPLITS=("retain99" "retain95" "retain90")

for i in "${!TOFU_FORGET_SPLITS[@]}"; do
    forget_split=${TOFU_FORGET_SPLITS[$i]}
    retain_split=${TOFU_RETAIN_SPLITS[$i]}

    # 1. Baseline (Before Unlearning)
    echo "Running baseline evaluation for TOFU ${forget_split}..."
    TOFU_TARGET_PATH="${TOFU_HF_ORG}/tofu_${TOFU_MODEL}_full"
    BASELINE_EVAL="${RESULTS_DIR}/tofu/baseline/${forget_split}/TOFU_EVAL.json"

    if [ ! -f "$BASELINE_EVAL" ]; then
        CUDA_VISIBLE_DEVICES=0 python src/eval.py \
            experiment=eval/tofu/default.yaml \
            +experiment/eval=comprehensive \
            forget_split=${forget_split} \
            model=${TOFU_MODEL} \
            model.model_args.pretrained_model_name_or_path=${TOFU_TARGET_PATH} \
            task_name="tofu_baseline_${forget_split}" \
            paths.output_dir="${RESULTS_DIR}/tofu/baseline/${forget_split}"
    else
        echo "Baseline evaluation for TOFU ${forget_split} already exists. Skipping."
    fi

    # 2. Unlearn and Eval for each method
    for trainer in "${GRAD_TRAINERS[@]}" "${NON_GRAD_TRAINERS[@]}"; do
        echo "Processing ${trainer} for TOFU ${forget_split}..."
        
        task_name="tofu_${TOFU_MODEL}_${forget_split}_${trainer}"
        TRAIN_DIR="${RESULTS_DIR}/tofu/${trainer}/${forget_split}/checkpoints"
        AFTER_EVAL="${RESULTS_DIR}/tofu/${trainer}/${forget_split}/evals/TOFU_EVAL.json"

        # Unlearn check
        if [ ! -d "$TRAIN_DIR" ]; then
            CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
                src/train.py --config-name=unlearn.yaml \
                experiment=unlearn/tofu/default.yaml \
                trainer=${trainer} \
                task_name=${task_name} \
                model=${TOFU_MODEL} \
                model.model_args.pretrained_model_name_or_path=${TOFU_TARGET_PATH} \
                forget_split=${forget_split} \
                retain_split=${retain_split} \
                paths.output_dir="${TRAIN_DIR}"
        else
            echo "Unlearning for ${trainer} (TOFU ${forget_split}) already exists. Skipping."
        fi

        # Eval check
        if [ ! -f "$AFTER_EVAL" ]; then
            CUDA_VISIBLE_DEVICES=0 python src/eval.py \
                experiment=eval/tofu/default.yaml \
                +experiment/eval=comprehensive \
                forget_split=${forget_split} \
                model=${TOFU_MODEL} \
                model.model_args.pretrained_model_name_or_path="${TRAIN_DIR}" \
                task_name="${task_name}_after" \
                paths.output_dir="${RESULTS_DIR}/tofu/${trainer}/${forget_split}/evals"
        else
            echo "Post-unlearning evaluation for ${trainer} (TOFU ${forget_split}) already exists. Skipping."
        fi
    done
done

# #########################################################
# #################### Final Analysis ####################
# #########################################################

echo "Running final analysis and generating graphs..."
python src/analyze_results.py --results_dir "${RESULTS_DIR}" --output_dir "${RESULTS_DIR}/analysis"
