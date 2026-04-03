#!/bin/bash

# Navigate to project root
cd "$(dirname "$0")/.."

# Environment setup
export HF_HOME="/data/mridulg22/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="/data/mridulg22/.cache/huggingface/hub"
export TRANSFORMERS_CACHE="/data/mridulg22/.cache/huggingface/transformers"

# Configuration
RUN_ID=${1:-${RUN_ID:-"extracted_$(date +"%Y%m%d_%H%M%S")"}}
RESULTS_DIR="results/${RUN_ID}"
echo "Using Results Directory: ${RESULTS_DIR}"
mkdir -p "${RESULTS_DIR}"

# Benchmarks and Models
TOFU_MODEL="Llama-3.2-1B-Instruct"
TOFU_HF_ORG="open-unlearning"
FORGET_SPLIT="forget10"
RETAIN_SPLIT="retain90"

# Mapping: Unlearning Method -> Hugging Face Repository ID
# We use the best performing or standard variants from the open-unlearning organization
declare -A TRAINER_TO_REPO=(
    ["NPO"]="open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr5e-05_beta0.1_alpha2_epoch5"
    ["SimNPO"]="open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr2e-05_b4.5_a1_d1_g0.125_ep10"
    ["GradDiff"]="open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha5_epoch10"
    ["RMU"]="open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr1e-05_layer5_scoeff100_epoch10"
    ["AltPO"]="open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr5e-05_beta0.1_alpha1_epoch10"
    ["IdkDPO"]="open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkDPO_lr2e-05_beta0.5_alpha1_epoch5"
)

# Trainers to evaluate
TRAINERS=("NPO" "SimNPO" "GradDiff" "RMU" "AltPO" "IdkDPO")

# #########################################################
# ################ TOFU Baseline Evaluation ###############
# #########################################################

echo "Running baseline evaluation for TOFU ${FORGET_SPLIT}..."
TOFU_TARGET_PATH="${TOFU_HF_ORG}/tofu_${TOFU_MODEL}_full"
BASELINE_EVAL="${RESULTS_DIR}/tofu/baseline/${FORGET_SPLIT}/TOFU_EVAL.json"

if [ ! -f "$BASELINE_EVAL" ]; then
    CUDA_VISIBLE_DEVICES=0 python src/eval.py \
        experiment=eval/tofu/default.yaml \
        +experiment/eval=comprehensive \
        forget_split=${FORGET_SPLIT} \
        model=${TOFU_MODEL} \
        model.model_args.pretrained_model_name_or_path=${TOFU_TARGET_PATH} \
        task_name="tofu_baseline_${FORGET_SPLIT}" \
        paths.output_dir="${RESULTS_DIR}/tofu/baseline/${FORGET_SPLIT}"
else
    echo "Baseline evaluation for TOFU ${FORGET_SPLIT} already exists. Skipping."
fi

# #########################################################
# ################ TOFU Extraction and Eval ###############
# #########################################################

for trainer in "${TRAINERS[@]}"; do
    REPO_ID=${TRAINER_TO_REPO[$trainer]}
    echo "Evaluating pre-unlearned model for ${trainer} at ${REPO_ID}..."
    
    task_name="tofu_${TOFU_MODEL}_${FORGET_SPLIT}_${trainer}"
    AFTER_EVAL="${RESULTS_DIR}/tofu/${trainer}/${FORGET_SPLIT}/evals/TOFU_EVAL.json"

    # Eval check
    if [ ! -f "$AFTER_EVAL" ]; then
        CUDA_VISIBLE_DEVICES=0 python src/eval.py \
            experiment=eval/tofu/default.yaml \
            +experiment/eval=comprehensive \
            forget_split=${FORGET_SPLIT} \
            model=${TOFU_MODEL} \
            model.model_args.pretrained_model_name_or_path="${REPO_ID}" \
            task_name="${task_name}_extracted" \
            paths.output_dir="${RESULTS_DIR}/tofu/${trainer}/${FORGET_SPLIT}/evals" \
            retain_logs_path="saves/eval/tofu_${TOFU_MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json"
    else
        echo "Evaluation for ${trainer} already exists. Skipping."
    fi
done

# #########################################################
# ##################### Final Analysis ####################
# #########################################################

echo "Running final analysis and generating graphs from extracted results..."
python src/analyze_results.py --results_dir "${RESULTS_DIR}" --output_dir "${RESULTS_DIR}/analysis"
