#!/bin/bash
set -euo pipefail

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

# Run summary
BASELINE_STATUS="not-run"
declare -a EVALUATED_MODELS=()
declare -a SKIPPED_MODELS=()
declare -a FAILED_MODELS=()
ANALYSIS_STATUS="not-run"

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

write_model_status() {
    local status_file="$1"
    local model_name="$2"
    local state="$3"
    local phase="$4"
    mkdir -p "$(dirname "$status_file")"
    cat > "$status_file" <<EOF
{
    "model": "${model_name}",
    "state": "${state}",
    "phase": "${phase}",
    "timestamp": "$(date -Iseconds)"
}
EOF
}

gradnorm_is_complete() {
    local eval_dir="$1"
    local checkpoint_dir="${eval_dir}/checkpoints/mia_gradnorm"

    if [ ! -f "${eval_dir}/TOFU_EVAL.json" ]; then
        return 1
    fi

    python - "$checkpoint_dir" <<'EOF'
import json
import os
import sys

checkpoint_dir = sys.argv[1]
for subset in ("forget", "holdout", "retain"):
    checkpoint_file = os.path.join(checkpoint_dir, subset, "attack_checkpoint.json")
    if not os.path.exists(checkpoint_file):
        raise SystemExit(1)
    with open(checkpoint_file, "r") as f:
        checkpoint = json.load(f)
    if not checkpoint.get("done", False):
        raise SystemExit(1)
raise SystemExit(0)
EOF
}

gradnorm_has_partial_checkpoint() {
    local eval_dir="$1"
    local checkpoint_dir="${eval_dir}/checkpoints/mia_gradnorm"

    python - "$checkpoint_dir" <<'EOF'
import json
import os
import sys

checkpoint_dir = sys.argv[1]
for subset in ("forget", "holdout", "retain"):
    checkpoint_file = os.path.join(checkpoint_dir, subset, "attack_checkpoint.json")
    if not os.path.exists(checkpoint_file):
        continue
    with open(checkpoint_file, "r") as f:
        checkpoint = json.load(f)
    if not checkpoint.get("done", False):
        raise SystemExit(0)
raise SystemExit(1)
EOF
}

tofu_eval_is_complete() {
    local eval_dir="$1"
    local eval_file="${eval_dir}/TOFU_EVAL.json"

    if [ ! -f "$eval_file" ]; then
        return 1
    fi

    if ! gradnorm_is_complete "$eval_dir"; then
        return 1
    fi

    python - "$eval_file" <<'EOF'
import json
import sys

eval_file = sys.argv[1]
with open(eval_file, "r") as f:
    logs = json.load(f)

metric = logs.get("mia_gradnorm")
if not isinstance(metric, dict):
    raise SystemExit(1)
if metric.get("agg_value") is None:
    raise SystemExit(1)

raise SystemExit(0)
EOF
}

# Trainers to evaluate
TRAINERS=("NPO" "SimNPO" "GradDiff" "RMU" "AltPO" "IdkDPO")

# #########################################################
# ################ TOFU Baseline Evaluation ###############
# #########################################################

echo "Running baseline evaluation for TOFU ${FORGET_SPLIT}..."
TOFU_TARGET_PATH="${TOFU_HF_ORG}/tofu_${TOFU_MODEL}_full"
BASELINE_EVAL="${RESULTS_DIR}/tofu/baseline/${FORGET_SPLIT}/TOFU_EVAL.json"
BASELINE_EVAL_DIR="${RESULTS_DIR}/tofu/baseline/${FORGET_SPLIT}"
BASELINE_STATUS_FILE="${RESULTS_DIR}/tofu/baseline/${FORGET_SPLIT}/TOFU_STATUS.json"
BASELINE_ATTEMPT=1

if tofu_eval_is_complete "$BASELINE_EVAL_DIR"; then
    echo "Baseline evaluation for TOFU ${FORGET_SPLIT} already complete. Skipping."
    BASELINE_STATUS="skipped"
    write_model_status "$BASELINE_STATUS_FILE" "baseline" "done" "evaluation"
else
    while ! tofu_eval_is_complete "$BASELINE_EVAL_DIR"; do
        if gradnorm_has_partial_checkpoint "$BASELINE_EVAL_DIR"; then
            write_model_status "$BASELINE_STATUS_FILE" "baseline" "running" "gradnorm_resume_attempt_${BASELINE_ATTEMPT}"
            echo "Running baseline attempt ${BASELINE_ATTEMPT} (checkpoint resume)..."
        else
            write_model_status "$BASELINE_STATUS_FILE" "baseline" "running" "evaluation_attempt_${BASELINE_ATTEMPT}"
            echo "Running baseline attempt ${BASELINE_ATTEMPT} (no gradnorm checkpoint yet)..."
        fi

        if CUDA_VISIBLE_DEVICES=0 python src/eval.py \
            experiment=eval/tofu/default.yaml \
            +experiment/eval/tofu=comprehensive \
            forget_split=${FORGET_SPLIT} \
            model=${TOFU_MODEL} \
            model.model_args.pretrained_model_name_or_path=${TOFU_TARGET_PATH} \
            task_name="tofu_baseline_${FORGET_SPLIT}" \
            paths.output_dir="${BASELINE_EVAL_DIR}"; then
            echo "Baseline attempt ${BASELINE_ATTEMPT} finished without shell error."
        else
            echo "Baseline attempt ${BASELINE_ATTEMPT} was interrupted or failed; retrying."
        fi

        BASELINE_ATTEMPT=$((BASELINE_ATTEMPT + 1))
        sleep 5
    done

    BASELINE_STATUS="success"
    write_model_status "$BASELINE_STATUS_FILE" "baseline" "done" "evaluation"
fi

# #########################################################
# ################ TOFU Extraction and Eval ###############
# #########################################################

for trainer in "${TRAINERS[@]}"; do
    REPO_ID=${TRAINER_TO_REPO[$trainer]}
    echo "Evaluating pre-unlearned model for ${trainer} at ${REPO_ID}..."
    
    task_name="tofu_${TOFU_MODEL}_${FORGET_SPLIT}_${trainer}"
    AFTER_EVAL="${RESULTS_DIR}/tofu/${trainer}/${FORGET_SPLIT}/evals/TOFU_EVAL.json"
    MODEL_STATUS_FILE="${RESULTS_DIR}/tofu/${trainer}/${FORGET_SPLIT}/evals/TOFU_STATUS.json"
    EVAL_DIR="${RESULTS_DIR}/tofu/${trainer}/${FORGET_SPLIT}/evals"
    ATTEMPT=1

    # Eval check
    if tofu_eval_is_complete "$EVAL_DIR"; then
        echo "Evaluation for ${trainer} already exists. Skipping."
        write_model_status "$MODEL_STATUS_FILE" "$trainer" "done" "evaluation"
        SKIPPED_MODELS+=("${trainer}")
    else
        if gradnorm_has_partial_checkpoint "$EVAL_DIR"; then
            echo "Found partial gradnorm checkpoints for ${trainer} (${FORGET_SPLIT}); retry will resume from saved batch."
        elif [ -f "$MODEL_STATUS_FILE" ]; then
            echo "Found status file for ${trainer} (${FORGET_SPLIT}), but no partial gradnorm checkpoint yet; retrying evaluation loop."
        else
            echo "Starting fresh evaluation for ${trainer} (${FORGET_SPLIT})."
        fi

        while ! tofu_eval_is_complete "$EVAL_DIR"; do
            if gradnorm_has_partial_checkpoint "$EVAL_DIR"; then
                write_model_status "$MODEL_STATUS_FILE" "$trainer" "running" "gradnorm_resume_attempt_${ATTEMPT}"
                echo "Running ${trainer} (${FORGET_SPLIT}) attempt ${ATTEMPT} (checkpoint resume)..."
            else
                write_model_status "$MODEL_STATUS_FILE" "$trainer" "running" "evaluation_attempt_${ATTEMPT}"
                echo "Running ${trainer} (${FORGET_SPLIT}) attempt ${ATTEMPT} (no gradnorm checkpoint yet)..."
            fi
            if CUDA_VISIBLE_DEVICES=0 python src/eval.py \
                experiment=eval/tofu/default.yaml \
                +experiment/eval/tofu=comprehensive \
                forget_split=${FORGET_SPLIT} \
                model=${TOFU_MODEL} \
                model.model_args.pretrained_model_name_or_path="${REPO_ID}" \
                task_name="${task_name}_extracted" \
                paths.output_dir="${EVAL_DIR}" \
                retain_logs_path="saves/eval/tofu_${TOFU_MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json"; then
                echo "${trainer} attempt ${ATTEMPT} finished without shell error."
            else
                echo "${trainer} attempt ${ATTEMPT} was interrupted or failed; retrying from checkpoint."
            fi

            if tofu_eval_is_complete "$EVAL_DIR"; then
                EVALUATED_MODELS+=("${trainer}")
                write_model_status "$MODEL_STATUS_FILE" "$trainer" "done" "evaluation"
                break
            fi

            ATTEMPT=$((ATTEMPT + 1))
            sleep 5
        done
    fi
done

# #########################################################
# ##################### Final Analysis ####################
# #########################################################

echo "Running final analysis and generating graphs from extracted results..."
if python src/analyze_results.py --results_dir "${RESULTS_DIR}" --output_dir "${RESULTS_DIR}/analysis"; then
    ANALYSIS_STATUS="success"
else
    ANALYSIS_STATUS="failed"
fi

echo ""
echo "========== Run Summary =========="
echo "Run ID: ${RUN_ID}"
echo "Results Directory: ${RESULTS_DIR}"
echo "Baseline: ${BASELINE_STATUS}"
echo "Evaluated models (${#EVALUATED_MODELS[@]}): ${EVALUATED_MODELS[*]:-none}"
echo "Skipped models (${#SKIPPED_MODELS[@]}): ${SKIPPED_MODELS[*]:-none}"
echo "Failed models (${#FAILED_MODELS[@]}): ${FAILED_MODELS[*]:-none}"
echo "Analysis: ${ANALYSIS_STATUS}"
echo "================================="
