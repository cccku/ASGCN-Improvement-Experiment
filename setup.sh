#!/bin/bash
# =============================================================================
# ASGCN Project Setup Script
# Ubuntu 22.04
# =============================================================================

set -e  # Exit on error

ENV_NAME="asgcn"
PYTHON_VERSION="3.10"

echo "============================================================"
echo "  ASGCN Project Environment Setup"
echo "============================================================"

# --------------------------------------------------------------------------
# 1. Create conda environment
# --------------------------------------------------------------------------
echo "[1/6] Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}"

# Activate
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"
echo "      Activated: $(which python)"

# --------------------------------------------------------------------------
# 2. Install PyTorch (CUDA 11.8 — compatible with RTX 3090Ti on Ubuntu 22.04)
# --------------------------------------------------------------------------
echo "[2/6] Installing PyTorch 2.0 + CUDA 11.8..."
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118

# --------------------------------------------------------------------------
# 3. Install core ML / NLP dependencies
# --------------------------------------------------------------------------
echo "[3/6] Installing core dependencies..."
pip install \
    numpy==1.24.4 \
    scikit-learn==1.3.2 \
    transformers==4.36.2 \
    tqdm==4.66.3 \
    pandas==2.1.4 \
    openpyxl==3.1.2

# spaCy pinned separately — 3.7.x needs thinc 8.2.x; install after tqdm
pip install "spacy==3.7.2" "thinc==8.2.5"

# --------------------------------------------------------------------------
# 4. Install spaCy English model (used for dependency parsing)
# --------------------------------------------------------------------------
echo "[4/6] Downloading spaCy English model..."
# Use pip directly with the pinned model wheel — avoids the broken URL
# that 'python -m spacy download' generates for some spacy versions
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl

# --------------------------------------------------------------------------
# 5. Download GloVe embeddings if not present
# --------------------------------------------------------------------------
echo "[5/6] Checking for GloVe embeddings..."
GLOVE_DIR="ASGCN_base"
GLOVE_FILE="${GLOVE_DIR}/glove.42B.300d.txt"

if [ ! -f "${GLOVE_FILE}" ]; then
    echo "      GloVe file not found. Downloading glove.42B.300d..."
    mkdir -p "${GLOVE_DIR}"
    wget -q --show-progress \
        https://nlp.stanford.edu/data/glove.42B.300d.zip \
        -O /tmp/glove.42B.300d.zip
    unzip -q /tmp/glove.42B.300d.zip -d "${GLOVE_DIR}/"
    rm /tmp/glove.42B.300d.zip
    echo "      GloVe downloaded to ${GLOVE_FILE}"
else
    echo "      GloVe found at ${GLOVE_FILE}, skipping download."
fi

# --------------------------------------------------------------------------
# 6. Verify installation
# --------------------------------------------------------------------------
echo "[6/6] Verifying installation..."
python - <<'EOF'
import torch
import transformers
import sklearn
import spacy
print(f"  PyTorch      : {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU          : {torch.cuda.get_device_name(0)}")
print(f"  Transformers : {transformers.__version__}")
print(f"  scikit-learn : {sklearn.__version__}")
print(f"  spaCy        : {spacy.__version__}")
EOF

echo ""
echo "============================================================"
echo "  Setup complete!"
echo "  Activate with:  conda activate ${ENV_NAME}"
echo "  Then run:       bash run_all.sh"
echo "============================================================"
