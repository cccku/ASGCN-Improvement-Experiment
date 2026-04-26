#!/bin/bash
# =============================================================================
# setup.sh — One-time environment and dataset setup
# =============================================================================
# Usage: bash setup.sh
# Tested on Ubuntu 22.04 with conda installed.
# =============================================================================

set -e

ENV_NAME="asgcn"
PYTHON_VERSION="3.10"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo "  ASGCN Project Setup"
echo "  Project root: ${PROJECT_ROOT}"
echo "============================================================"

# --------------------------------------------------------------------------
# 1. Create conda environment
# --------------------------------------------------------------------------
echo "[1/5] Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}"
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"
echo "      Activated: $(which python)"

# --------------------------------------------------------------------------
# 2. Install PyTorch (CUDA 11.8 — compatible with RTX 3090Ti)
# --------------------------------------------------------------------------
echo "[2/5] Installing PyTorch 2.0 + CUDA 11.8..."
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==0.15.2 \
    --index-url https://download.pytorch.org/whl/cu118

# --------------------------------------------------------------------------
# 3. Install core dependencies
# --------------------------------------------------------------------------
echo "[3/5] Installing dependencies..."
pip install \
    numpy==1.24.4 \
    scikit-learn==1.3.2 \
    transformers==4.36.2 \
    tqdm==4.66.3 \
    pandas==2.1.4 \
    openpyxl==3.1.2
pip install "spacy==3.7.2" "thinc==8.2.5"
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl

# --------------------------------------------------------------------------
# 4. Download GloVe embeddings
# --------------------------------------------------------------------------
echo "[4/5] Checking for GloVe embeddings..."
GLOVE_DIR="${PROJECT_ROOT}/glove"
GLOVE_FILE="${GLOVE_DIR}/glove.840B.300d.txt"

if [ ! -f "${GLOVE_FILE}" ]; then
    echo "      Downloading glove.840B.300d..."
    mkdir -p "${GLOVE_DIR}"
    wget -q --show-progress \
        https://nlp.stanford.edu/data/glove.840B.300d.zip \
        -O /tmp/glove.840B.300d.zip
    unzip -q /tmp/glove.840B.300d.zip -d "${GLOVE_DIR}/"
    rm /tmp/glove.840B.300d.zip
    echo "      GloVe saved to ${GLOVE_FILE}"
else
    echo "      GloVe found at ${GLOVE_FILE}, skipping."
fi

# --------------------------------------------------------------------------
# 5. Build dependency graphs and trees from .raw files
# --------------------------------------------------------------------------
echo "[5/5] Building dependency graphs and parse trees..."
echo "      This uses spaCy and may take 5-10 minutes..."

cd "${PROJECT_ROOT}"
python shared/dependency_graph.py
python shared/dependency_tree.py

echo ""
echo "============================================================"
echo "  Setup complete!"
echo "  Activate env : conda activate ${ENV_NAME}"
echo "  Run all      : bash run_all.sh"
echo "  Run one      : bash run_all.sh C"
echo "============================================================"
