#!/bin/bash
# =============================================================================
# run_all.sh — Run ASGCN improvement experiments
# =============================================================================
# Usage:
#   bash run_all.sh              # run all (A B C D E)
#   bash run_all.sh C            # run only C
#   bash run_all.sh C D E        # run multiple
# =============================================================================

SEEDS=(42 123 776)
DATASETS=(rest14 lap14 rest15 rest16)
DEVICE="cuda"
NUM_EPOCH=100
BATCH_SIZE=32
LOG_STEP=5
SUMMARY_DIR="results"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

mkdir -p "${SUMMARY_DIR}"
FAILED_RUNS=()

# --------------------------------------------------------------------------
run_experiment() {
    local LABEL=$1
    local WORKDIR=$2
    local MODEL=$3
    local DATASET=$4
    local SEED=$5
    local EXTRA_ARGS=${6:-""}

    local LOG_FILE="${SUMMARY_DIR}/${LABEL}/${DATASET}_seed${SEED}.txt"
    mkdir -p "${SUMMARY_DIR}/${LABEL}"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [${LABEL}] dataset=${DATASET}  seed=${SEED}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

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

    pushd "${PROJECT_ROOT}/${WORKDIR}" > /dev/null

    python train.py \
        --model_name    "${MODEL}" \
        --dataset       "${DATASET}" \
        --seed          "${SEED}" \
        --device        "${DEVICE}" \
        --num_epoch     "${NUM_EPOCH}" \
        --batch_size    "${BATCH_SIZE}" \
        --log_step      "${LOG_STEP}" \
        ${EXTRA_ARGS} \
        2>&1 | tee -a "${PROJECT_ROOT}/${LOG_FILE}"
    local EXIT_CODE=${PIPESTATUS[0]}

    popd > /dev/null

    if [ ${EXIT_CODE} -ne 0 ]; then
        echo "  ✗ FAILED (exit ${EXIT_CODE}) → ${LOG_FILE}"
        FAILED_RUNS+=("${LABEL}/${DATASET}/seed${SEED}")
    else
        local ACC=$(grep "max_test_acc_avg" "${LOG_FILE}" | tail -1 | grep -oP '[0-9]+\.[0-9]+' | head -1)
        local F1=$(grep  "max_test_f1_avg"  "${LOG_FILE}" | tail -1 | grep -oP '[0-9]+\.[0-9]+' | head -1)
        echo "  ✓ Done  |  acc=${ACC}  f1=${F1}"
        echo "RESULT: seed=${SEED}  acc=${ACC}  f1=${F1}" >> "${LOG_FILE}"
    fi
}

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
echo "  PYTHONPATH   : ${PYTHONPATH}"
echo "============================================================"
START_TIME=$(date +%s)

# A — Baseline Reimplementation
if [[ " ${RUN_LIST[*]} " == *" A "* ]]; then
    echo ""; echo "▶▶▶  A — Baseline Reimplementation"
    for DS in "${DATASETS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            run_experiment "A_baseline" "Baseline_Reimplementation" "asgcn" "${DS}" "${SEED}"
        done
    done
fi

# B — Baseline + BERT
if [[ " ${RUN_LIST[*]} " == *" B "* ]]; then
    echo ""; echo "▶▶▶  B — Baseline + BERT"
    for DS in "${DATASETS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            run_experiment "B_bert" "Baseline_BERT" "asgcn_bert" "${DS}" "${SEED}" \
                "--bert_lr 2e-5 --num_epoch 20 --batch_size 16"
        done
    done
fi

# C — Baseline + GAT
if [[ " ${RUN_LIST[*]} " == *" C "* ]]; then
    echo ""; echo "▶▶▶  C — Baseline + GAT"
    for DS in "${DATASETS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            run_experiment "C_gat" "Baseline_GAT" "asgcn_gat" "${DS}" "${SEED}"
        done
    done
fi

# D — Baseline + SoftMask
if [[ " ${RUN_LIST[*]} " == *" D "* ]]; then
    echo ""; echo "▶▶▶  D — Baseline + SoftMask"
    for DS in "${DATASETS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            run_experiment "D_softmask" "Baseline_SoftMask" "asgcn_softmask" "${DS}" "${SEED}"
        done
    done
fi

# E — Baseline + Contrastive
if [[ " ${RUN_LIST[*]} " == *" E "* ]]; then
    echo ""; echo "▶▶▶  E — Baseline + Contrastive"
    for DS in "${DATASETS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            run_experiment "E_contrastive" "Baseline_Contrastive" "asgcn_contrastive" "${DS}" "${SEED}"
        done
    done
fi

# --------------------------------------------------------------------------
END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
echo ""
echo "============================================================"
echo "  Finished in $(( ELAPSED/3600 ))h $(( (ELAPSED%3600)/60 ))m"
echo "  Results in: ${SUMMARY_DIR}/"
if [ ${#FAILED_RUNS[@]} -gt 0 ]; then
    echo "  ✗ FAILED (${#FAILED_RUNS[@]}):"; for r in "${FAILED_RUNS[@]}"; do echo "      - ${r}"; done
else
    echo "  ✓ All runs completed successfully"
fi
echo "  Run: python summarize_results.py  to generate result tables"
echo "============================================================"
