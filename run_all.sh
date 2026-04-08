#!/bin/bash
# =============================================================================
# run_all.sh — Run ALL ASGCN improvement experiments
# =============================================================================
# Usage:
#   bash run_all.sh              # run everything (A B C D E)
#   bash run_all.sh C            # run only improvement C
#   bash run_all.sh C D E        # run C, D and E
#
# Experimental design (for fair comparison):
#   - Fixed seeds : 42, 123, 776  (same for ALL improvements and datasets)
#   - 1 run per seed (repeats=1 inside train.py)
#   - Results for each seed are logged separately
#   - summarize_results.py aggregates mean ± std across seeds
# =============================================================================

# No set -e — handle errors per-experiment
SEEDS=(42 123 776)
DATASETS=(rest14 lap14 rest15 rest16)
DEVICE="cuda"
NUM_EPOCH=100
BATCH_SIZE=32
LOG_STEP=5
SUMMARY_DIR="results"

mkdir -p "${SUMMARY_DIR}"
FAILED_RUNS=()

# --------------------------------------------------------------------------
# Helper: run ONE seed of one experiment
# --------------------------------------------------------------------------
run_experiment() {
    local LABEL=$1      # e.g. "C_gat"
    local WORKDIR=$2    # e.g. "improvement_C_gat"
    local MODEL=$3      # e.g. "asgcn_gat"
    local DATASET=$4
    local SEED=$5
    local EXTRA_ARGS=${6:-""}

    local LOG_FILE="${SUMMARY_DIR}/${LABEL}/${DATASET}_seed${SEED}.txt"
    mkdir -p "${SUMMARY_DIR}/${LABEL}"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [${LABEL}] dataset=${DATASET}  seed=${SEED}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Self-contained header in every log file
    cat > "${LOG_FILE}" << HEADER
================================================
  Improvement : ${LABEL}
  Model       : ${MODEL}
  Dataset     : ${DATASET}
  Seed        : ${SEED}
  Epochs      : ${NUM_EPOCH}
  Batch size  : ${BATCH_SIZE}
  Extra args  : ${EXTRA_ARGS}
  Started at  : $(date '+%Y-%m-%d %H:%M:%S')
================================================
HEADER

    pushd "${WORKDIR}" > /dev/null

    python train.py \
        --model_name    "${MODEL}" \
        --dataset       "${DATASET}" \
        --seed          "${SEED}" \
        --device        "${DEVICE}" \
        --num_epoch     "${NUM_EPOCH}" \
        --batch_size    "${BATCH_SIZE}" \
        --log_step      "${LOG_STEP}" \
        ${EXTRA_ARGS} \
        2>&1 | tee -a "../${LOG_FILE}"
    local EXIT_CODE=${PIPESTATUS[0]}

    popd > /dev/null

    if [ ${EXIT_CODE} -ne 0 ]; then
        echo "  ✗ FAILED (exit ${EXIT_CODE}) → ${LOG_FILE}"
        echo "RESULT: FAILED (exit ${EXIT_CODE})" >> "${LOG_FILE}"
        FAILED_RUNS+=("${LABEL}/${DATASET}/seed${SEED}")
    else
        # Extract final metrics from this run
        local ACC=$(grep "max_test_acc_avg" "${LOG_FILE}" | tail -1 | grep -oP '[0-9]+\.[0-9]+' | head -1)
        local F1=$( grep "max_test_f1_avg"  "${LOG_FILE}" | tail -1 | grep -oP '[0-9]+\.[0-9]+' | head -1)
        echo "  ✓ Done  |  acc=${ACC}  f1=${F1}  seed=${SEED}"
        {
            echo "------------------------------------------------"
            echo "RESULT: seed=${SEED}  acc=${ACC}  f1=${F1}"
        } >> "${LOG_FILE}"
    fi
}

# --------------------------------------------------------------------------
# Which improvements to run
# --------------------------------------------------------------------------
if [ $# -eq 0 ]; then
    RUN_LIST=(A B C D E)
else
    RUN_LIST=("$@")
fi

echo "============================================================"
echo "  ASGCN Experiment Runner"
echo "  Improvements : ${RUN_LIST[*]}"
echo "  Datasets     : ${DATASETS[*]}"
echo "  Seeds        : ${SEEDS[*]}"
echo "  Device       : ${DEVICE}"
echo "  Note         : 1 run per seed, seeds fixed across all improvements"
echo "============================================================"
START_TIME=$(date +%s)

# --------------------------------------------------------------------------
# A — Baseline ASGCN
# --------------------------------------------------------------------------
if [[ " ${RUN_LIST[*]} " == *" A "* ]]; then
    echo ""; echo "▶▶▶  IMPROVEMENT A — Baseline ASGCN"
    for DS in "${DATASETS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            run_experiment "A_baseline" "ASGCN_base" "asgcn" "${DS}" "${SEED}"
        done
    done
fi

# --------------------------------------------------------------------------
# B — BERT-enhanced ASGCN
# --------------------------------------------------------------------------
if [[ " ${RUN_LIST[*]} " == *" B "* ]]; then
    echo ""; echo "▶▶▶  IMPROVEMENT B — BERT-enhanced ASGCN"
    for DS in "${DATASETS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            run_experiment "B_bert" "improvement_B_bert" "asgcn_bert" "${DS}" "${SEED}" \
                "--bert_lr 2e-5 --num_epoch 20 --batch_size 16"
        done
    done
fi

# --------------------------------------------------------------------------
# C — GAT
# --------------------------------------------------------------------------
if [[ " ${RUN_LIST[*]} " == *" C "* ]]; then
    echo ""; echo "▶▶▶  IMPROVEMENT C — Graph Attention Network"
    for DS in "${DATASETS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            run_experiment "C_gat" "improvement_C_gat" "asgcn_gat" "${DS}" "${SEED}"
        done
    done
fi

# --------------------------------------------------------------------------
# D — Soft Mask
# --------------------------------------------------------------------------
if [[ " ${RUN_LIST[*]} " == *" D "* ]]; then
    echo ""; echo "▶▶▶  IMPROVEMENT D — Learnable Soft Graph Masking"
    for DS in "${DATASETS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            run_experiment "D_softmask" "improvement_D_softmask" "asgcn_softmask" "${DS}" "${SEED}"
        done
    done
fi

# --------------------------------------------------------------------------
# E — Contrastive
# --------------------------------------------------------------------------
if [[ " ${RUN_LIST[*]} " == *" E "* ]]; then
    echo ""; echo "▶▶▶  IMPROVEMENT E — Supervised Contrastive Learning"
    for DS in "${DATASETS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            run_experiment "E_contrastive" "improvement_E_contrastive" \
                "asgcn_contrastive" "${DS}" "${SEED}" \
                "--contrastive_lambda 0.1"
        done
    done
fi

# --------------------------------------------------------------------------
# Final summary
# --------------------------------------------------------------------------
END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
HOURS=$(( ELAPSED / 3600 ))
MINS=$(( (ELAPSED % 3600) / 60 ))

echo ""
echo "============================================================"
echo "  All requested experiments finished!"
echo "  Total time: ${HOURS}h ${MINS}m"
echo "  Results in: ${SUMMARY_DIR}/"
if [ ${#FAILED_RUNS[@]} -gt 0 ]; then
    echo ""
    echo "  ✗ FAILED RUNS (${#FAILED_RUNS[@]}):"
    for r in "${FAILED_RUNS[@]}"; do
        echo "      - ${r}"
    done
else
    echo "  ✓ All runs completed successfully"
fi
echo ""
echo "  Run: python summarize_results.py  to see the full table"
echo "============================================================"