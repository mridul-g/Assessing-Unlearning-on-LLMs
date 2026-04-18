#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p scripts/logs

RUN_ID="${RUN_ID:-tofu_bucket_$(date +"%Y%m%d_%H%M%S")}"
STAGE="${STAGE:-all}"
MODEL_PATH="${MODEL_PATH:-open-unlearning/tofu_Llama-3.2-1B-Instruct_full}"
FORGET_SPLIT="${FORGET_SPLIT:-forget10}"
HOLDOUT_SPLIT="${HOLDOUT_SPLIT:-holdout10}"
RETAIN_SPLIT="${RETAIN_SPLIT:-retain90}"
BUCKET_SIZE="${BUCKET_SIZE:-400}"
SEED="${SEED:-42}"
SAVE_STEPS="${SAVE_STEPS:-50}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
METHODS="${METHODS:-GradAscent,GradDiff,NPO,SimNPO,RMU,UNDIAL,SatImp,WGA,CEU,PDU}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CPU_CORE="${CPU_CORE:-0}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
FORCE_BASELINE="${FORCE_BASELINE:-0}"
METRICS_ONLY="${METRICS_ONLY:-0}"

MAX_RETRIES="${MAX_RETRIES:--1}"  # -1 means infinite retries
RETRY_SLEEP_SEC="${RETRY_SLEEP_SEC:-30}"

WATCHDOG_LOG="scripts/logs/resilient_${RUN_ID}.log"
ATTEMPT=0

echo "[resilient] run_id=${RUN_ID} stage=${STAGE}" | tee -a "$WATCHDOG_LOG"
echo "[resilient] watchdog_log=${WATCHDOG_LOG}" | tee -a "$WATCHDOG_LOG"

while true; do
    ATTEMPT=$((ATTEMPT + 1))
    TS=$(date +%Y%m%d_%H%M%S)
    ATTEMPT_LOG="scripts/logs/${RUN_ID}_attempt_${ATTEMPT}_${TS}.log"

    echo "[resilient] attempt=${ATTEMPT} start=$(date -Iseconds) log=${ATTEMPT_LOG}" | tee -a "$WATCHDOG_LOG"

    set +e
    LAUNCH_PREFIX=()
    if command -v taskset >/dev/null 2>&1; then
        LAUNCH_PREFIX=(taskset -c "$CPU_CORE")
    fi

    env \
        RUN_ID="$RUN_ID" \
        STAGE="$STAGE" \
        MODEL_PATH="$MODEL_PATH" \
        FORGET_SPLIT="$FORGET_SPLIT" \
        HOLDOUT_SPLIT="$HOLDOUT_SPLIT" \
        RETAIN_SPLIT="$RETAIN_SPLIT" \
        BUCKET_SIZE="$BUCKET_SIZE" \
        SEED="$SEED" \
        SAVE_STEPS="$SAVE_STEPS" \
        EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE" \
        METHODS="$METHODS" \
        FORCE_TRAIN="$FORCE_TRAIN" \
        FORCE_BASELINE="$FORCE_BASELINE" \
        METRICS_ONLY="$METRICS_ONLY" \
        CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
        "${LAUNCH_PREFIX[@]}" ./scripts/run_tofu_bucket_pipeline.sh > "$ATTEMPT_LOG" 2>&1
    EXIT_CODE=$?
    set -e

    if [ "$EXIT_CODE" -eq 0 ] && [ -f "results/${RUN_ID}/RUN_COMPLETE.json" ]; then
        echo "[resilient] completed successfully on attempt=${ATTEMPT} at $(date -Iseconds)" | tee -a "$WATCHDOG_LOG"
        ln -sfn "$(basename "$ATTEMPT_LOG")" "scripts/logs/latest_tofu_bucket.log"
        break
    fi

    echo "[resilient] attempt=${ATTEMPT} failed exit_code=${EXIT_CODE} at $(date -Iseconds)" | tee -a "$WATCHDOG_LOG"
    ln -sfn "$(basename "$ATTEMPT_LOG")" "scripts/logs/latest_tofu_bucket.log"

    if [ "$MAX_RETRIES" -ge 0 ] && [ "$ATTEMPT" -ge "$MAX_RETRIES" ]; then
        echo "[resilient] max retries reached (${MAX_RETRIES}), exiting" | tee -a "$WATCHDOG_LOG"
        exit "$EXIT_CODE"
    fi

    echo "[resilient] sleeping ${RETRY_SLEEP_SEC}s before retry" | tee -a "$WATCHDOG_LOG"
    sleep "$RETRY_SLEEP_SEC"
done
