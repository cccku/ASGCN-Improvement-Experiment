# Improving Aspect-Based Sentiment Classification: Four Enhancements to ASGCN

**NTU AI6103 Deep Learning — Group Project**
Built on top of [Zhang et al., EMNLP 2019](https://aclanthology.org/D19-1464/)

---

## Team (Group 15)

| Member | Student ID | Contribution |
|---|---|---|
| Chang Yuan-Lun | G2510220J | Project coordination and proposal; Baseline + Contrastive Loss implementation; literature review and analysis writing; evaluation code and result analysis; codebase management; presentation |
| Chua Xin Xue | G2510813C | Environment setup and baseline reimplementation; codebase integration and repository management; literature writing; ; powerpoint design and slide finalization; presentation |
| Pang Zishuo | G2503640G | Baseline + Soft Masking implementation; soft adjacency gate design and ablation; literature writing; report editing |
| Chen Ziting | G2510777K | Baseline + BERT implementation; BERT tokenisation pipeline and differential learning rate setup; PowerPoint design and slide drafting |
| Zhang Keyun | G2509663A | Baseline + GAT implementation; graph attention layer design and integration; technical consultation across improvements |



---

## Overview

This project extends **ASGCN** (Aspect-Specific Graph Convolutional Network), a dependency-tree-based model for Aspect-Based Sentiment Analysis (ABSA). Given a sentence and a target aspect term, the model predicts sentiment polarity: **positive**, **negative**, or **neutral**.

We reproduce the original ASGCN baseline and implement four independent improvements, each evaluated across four benchmark datasets and three random seeds.

**Example:**

| Sentence | Aspect | Sentiment |
|---|---|---|
| "The food was great but the service was awful." | food | Positive |
| "The food was great but the service was awful." | service | Negative |

---

## Repository Structure

```
ASGCN-Improvement-Experiment/
├── shared/                          # Shared infrastructure (from original ASGCN)
│   ├── bucket_iterator.py
│   ├── dependency_graph.py          # Generates .graph files from .raw
│   ├── dependency_tree.py           # Generates .tree files from .raw
│   ├── __init__.py
│   └── layers/
│       ├── dynamic_rnn.py           # BiLSTM wrapper
│       └── __init__.py
│
├── datasets/                        # Raw benchmark datasets (.raw files only)
│   ├── semeval14/                   # rest14, lap14 (SemEval 2014 Task 4)
│   ├── semeval15/                   # rest15 (SemEval 2015 Task 12)
│   ├── semeval16/                   # rest16 (SemEval 2016 Task 5)
│   └── acl-14-short-data/           # Twitter (ACL 2014)
│
├── Baseline_Reimplementation/       # A: Original ASGCN reproduced
│   ├── layers -> ../shared/layers   # Symlink to shared layers
│   ├── models/
│   │   ├── asgcn.py                 # Core ASGCN model
│   │   └── ...
│   └── train.py
│
├── Baseline_BERT/                   # B: GloVe replaced with BERT embeddings
│   ├── layers -> ../shared/layers
│   ├── models/
│   │   ├── asgcn_bert.py            # BERT-enhanced ASGCN
│   │   └── ...
│   └── train.py
│
├── Baseline_GAT/                    # C: GCN replaced with Graph Attention Network
│   ├── layers -> ../shared/layers
│   ├── models/
│   │   ├── asgcn_gat.py             # GAT-based ASGCN
│   │   └── ...
│   └── train.py
│
├── Baseline_SoftMask/               # D: Learnable soft graph masking
│   ├── layers -> ../shared/layers
│   ├── models/
│   │   ├── asgcn_softmask.py        # Soft mask ASGCN
│   │   └── ...
│   └── train.py
│
├── Baseline_Contrastive/            # E: Supervised contrastive loss
│   ├── layers -> ../shared/layers
│   ├── models/
│   │   ├── asgcn_contrastive.py     # Contrastive ASGCN
│   │   └── ...
│   └── train.py
│
├── results/                         # All experiment logs (60 files)
│   ├── A_baseline/
│   ├── B_bert/
│   ├── C_gat/
│   ├── D_softmask/
│   ├── E_contrastive/
│   ├── result.txt                   # Formatted summary tables
│   └── results_summary.csv          # Machine-readable results
│
├── data_utils.py                    # Dataset reader for A, C, D, E (GloVe)
├── data_utils_bert.py               # Dataset reader for B (BERT tokenisation)
├── summarize_results.py             # Parses logs → result tables
├── setup.sh                         # One-time environment + dataset setup
├── run_all.sh                       # Run all or selected experiments
└── .gitignore
```

---

## Setup

### Prerequisites

- Ubuntu 22.04 (tested) or compatible Linux
- [Anaconda](https://www.anaconda.com/download) or Miniconda installed
- NVIDIA GPU with CUDA 11.8 support (tested on RTX 3090Ti and Kaggle T4)

### Step 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ASGCN-Improvement-Experiment.git
cd ASGCN-Improvement-Experiment
```

### Step 2 — Run one-time setup

```bash
bash setup.sh
```

This script will:

1. Create a conda environment `asgcn` with Python 3.10
2. Install PyTorch 2.0 + CUDA 11.8
3. Install all dependencies (transformers, scikit-learn, spaCy, etc.)
4. Download GloVe embeddings (`glove.840B.300d`, ~2.0 GB) into `glove/`
5. Build dependency graphs and parse trees from `.raw` dataset files into `datasets/`

> **Note:** Step 4 requires ~2 GB of disk space and a stable internet connection. Step 5 uses spaCy and takes approximately 5–10 minutes.

### Step 3 — Activate the environment

```bash
conda activate asgcn
```

---

## Running Experiments

### Run all five improvements

```bash
bash run_all.sh
```

### Run a specific improvement

```bash
bash run_all.sh A          # Baseline Reimplementation only
bash run_all.sh B          # BERT only
bash run_all.sh C D E      # Multiple
```

### Run a single dataset manually

```bash
# From the project root, cd into the improvement folder
cd Baseline_Reimplementation
python train.py --model_name asgcn --dataset rest14 --seed 42
```

```bash
cd Baseline_BERT
python train.py --model_name asgcn_bert --dataset rest14 --seed 42 \
    --bert_lr 2e-5 --num_epoch 20 --batch_size 16
```

### Key training arguments

| Argument | Default | Description |
|---|---|---|
| `--model_name` | varies | Model to train |
| `--dataset` | `rest14` | `rest14`, `lap14`, `rest15`, `rest16` |
| `--seed` | `776` | Random seed (experiments used: 42, 123, 776) |
| `--num_epoch` | `100` | Max training epochs (early stop at patience=5) |
| `--batch_size` | `32` | Batch size (`16` recommended for BERT) |
| `--learning_rate` | `0.001` | Adam learning rate for GCN/classifier head |
| `--bert_lr` | `2e-5` | BERT encoder learning rate (Improvement B only) |
| `--save` | `False` | Save best model checkpoint to `state_dict/` |

### Regenerate result tables

```bash
python summarize_results.py          # Print tables to terminal
python summarize_results.py --csv    # Also write results_summary.csv
```

---

## Datasets

All datasets are included as `.raw` files and committed to the repository. The `.graph` and `.tree` files are generated during setup and excluded from version control via `.gitignore`.

| Dataset | Domain | Train samples | Test samples | Source |
|---|---|---|---|---|
| **rest14** | Restaurant reviews | 3,608 | 1,120 | SemEval 2014 Task 4 |
| **lap14** | Laptop reviews | 2,966 | 1,328 | SemEval 2014 Task 4 |
| **rest15** | Restaurant reviews | 1,315 | 685 | SemEval 2015 Task 12 |
| **rest16** | Restaurant reviews | 1,240 | 544 | SemEval 2016 Task 5 |

---

## Improvements

| | Improvement | Key Change | Key File |
|---|---|---|---|
| **A** | Baseline Reimplementation | Reproduce original ASGCN on all 4 datasets | `models/asgcn.py` |
| **B** | Baseline + BERT | Replace static GloVe (300d) with `bert-base-uncased` contextual embeddings (768d) | `models/asgcn_bert.py` |
| **C** | Baseline + GAT | Replace GCN aggregation with single-head Graph Attention Network | `models/asgcn_gat.py` |
| **D** | Baseline + SoftMask | Replace binary adjacency matrix with a learned soft gate (MLP per edge) | `models/asgcn_softmask.py` |
| **E** | Baseline + Contrastive | Augment cross-entropy with supervised contrastive loss to improve class separation | `models/asgcn_contrastive.py` |

---

## Results

All results are averaged over **3 random seeds** (42, 123, 776). ± values denote standard deviation across seeds. Best results per column in **bold**.

### Accuracy (%)

| Model | Rest14 | Lap14 | Rest15 | Rest16 |
|---|---|---|---|---|
| ASGCN (paper) | 80.86 | 75.55 | 79.89 | 88.99 |
| A — Baseline (ours) | 81.43 ±0.19 | 75.97 ±0.87 | 79.62 ±0.13 | 88.19 ±0.32 |
| **B — Baseline + BERT** | **85.98 ±0.29** | **80.02 ±0.19** | **84.77 ±0.53** | **92.14 ±0.08** |
| C — Baseline + GAT | 80.80 ±0.25 | 74.96 ±1.05 | 79.83 ±0.39 | 88.49 ±0.34 |
| D — Baseline + SoftMask | 81.49 ±0.47 | 75.74 ±0.40 | 79.48 ±0.35 | 88.29 ±0.38 |
| E — Baseline + Contrastive | 81.16 ±0.30 | 75.98 ±0.20 | 79.60 ±0.20 | 88.29 ±0.23 |

### Macro-F1 (%)

| Model | Rest14 | Lap14 | Rest15 | Rest16 |
|---|---|---|---|---|
| ASGCN (paper) | 72.19 | 71.05 | 61.89 | 67.48 |
| A — Baseline (ours) | 73.05 ±0.40 | 71.99 ±1.26 | 62.76 ±0.65 | 69.44 ±0.19 |
| **B — Baseline + BERT** | **79.80 ±0.29** | **76.48 ±0.33** | **71.46 ±1.90** | **79.04 ±1.57** |
| C — Baseline + GAT | 71.69 ±0.59 | 70.87 ±1.33 | 62.22 ±0.71 | 68.60 ±1.47 |
| D — Baseline + SoftMask | 73.03 ±0.24 | 71.67 ±0.67 | 61.16 ±0.17 | 70.61 ±0.85 |
| E — Baseline + Contrastive | 72.66 ±0.74 | 71.93 ±0.39 | 62.46 ±0.61 | 68.09 ±1.10 |

### Δ vs Paper Baseline (accuracy, percentage points)

| Model | Rest14 | Lap14 | Rest15 | Rest16 |
|---|---|---|---|---|
| A — Baseline (ours) | +0.57 | +0.42 | −0.27 | −0.80 |
| **B — Baseline + BERT** | **+5.12** | **+4.47** | **+4.88** | **+3.15** |
| C — Baseline + GAT | −0.06 | −0.59 | −0.06 | −0.50 |
| D — Baseline + SoftMask | +0.63 | +0.19 | −0.41 | −0.70 |
| E — Baseline + Contrastive | +0.30 | +0.43 | −0.29 | −0.70 |

### Key Findings

**B — Baseline + BERT** is the dominant improvement, yielding +3.15 to +5.12pp accuracy and +4.49 to +9.60pp F1 across all datasets. Replacing static GloVe embeddings with contextual BERT representations substantially improves the model's handling of polysemous words and the minority neutral class.

**C — Baseline + GAT** marginally underperforms the baseline on most datasets. Single-head attention without edge-type awareness provides no reliable advantage over GCN on clean dependency trees, consistent with published findings.

**D — Baseline + SoftMask** shows small gains on rest14 and lap14 but regresses on rest15, the smallest dataset. The learnable edge gates require sufficient training data to learn meaningful adjacency weights.

**E — Baseline + Contrastive** produces the most stable results across seeds (lowest variance), suggesting the contrastive objective acts as a regulariser rather than fundamentally reshaping the representation space.

---

## Base Paper

```bibtex
@inproceedings{zhang2019aspect,
    title     = {Aspect-based Sentiment Classification with Aspect-specific Graph Convolutional Networks},
    author    = {Zhang, Chen and Li, Qiuchi and Song, Dawei},
    booktitle = {Proceedings of EMNLP-IJCNLP 2019},
    pages     = {4568--4578},
    year      = {2019},
    url       = {https://aclanthology.org/D19-1464/}
}
```

## Acknowledgements

Base codebase adapted from [GeneZC/ASGCN](https://github.com/GeneZC/ASGCN).
