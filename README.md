# AI6103 Deep Learning Project
## Improving ASGCN for Aspect-Based Sentiment Classification

NTU AI6103 Group Project — built on top of [Zhang et al., EMNLP 2019](https://aclanthology.org/D19-1464/).

---

## Overview

This project extends **ASGCN** (Aspect-Specific Graph Convolutional Network), a dependency-tree-based model for Aspect-Based Sentiment Analysis (ABSA). Given a sentence and a target aspect term, the model predicts sentiment polarity: positive, negative, or neutral.

We reproduce the original baseline and implement four independent improvements, each evaluated across four benchmark datasets and three random seeds.

---

## Repository Structure

```
project/
├── improvement_A_baseline/     ← Original ASGCN, verified reproduction
├── improvement_B_bert/         ← BERT contextual embeddings
├── improvement_C_gat/          ← GAT replacing GCN layers
├── improvement_D_softmask/     ← Learnable soft adjacency masking
├── improvement_E_contrastive/  ← Contrastive loss training
├── results/
│   ├── A_baseline/
│   │   └── results_logs/       ← Per-seed training logs (.txt)
│   ├── B_bert/
│   ├── C_gat/
│   ├── D_softmask/
│   └── E_contrastive/
├── Results.xlsx                ← Aggregated results spreadsheet
└── 01_Teammate_Template_v2.ipynb  ← Colab notebook template
```

Each improvement folder is self-contained and shares the same structure:

```
improvement_X/
├── models/
│   ├── asgcn.py               ← Original baseline model
│   └── asgcn_<name>.py        ← Modified model for this improvement
├── layers/
│   └── dynamic_rnn.py         ← BiLSTM wrapper (unchanged)
├── datasets/
│   ├── semeval14/             ← rest14, lap14
│   ├── semeval15/             ← rest15
│   └── semeval16/             ← rest16
├── train.py                   ← Training entry point
├── data_utils.py              ← Dataset reader and tokenizer
├── bucket_iterator.py
└── dependency_graph.py
```

---

## Improvements

### A — Baseline
Reproduces the original ASGCN paper results. BiLSTM encoder → 2-layer GCN over dependency tree → aspect-specific masking → attention pooling → softmax classifier.

### B — BERT Embeddings
Replaces static GloVe (300d) embeddings with `bert-base-uncased` contextual embeddings (768d). A linear projection maps BERT output to match the GCN's expected input dimension.

### C — Graph Attention Network
Replaces the symmetric `GraphConvolution` layers with single-head `GraphAttentionLayer` (Veličković et al., 2018). Attention scores are computed as `e_ij = LeakyReLU(a^T [Wh_i || Wh_j])`, normalised with softmax over in-neighbours masked by the adjacency matrix.

### D — Learnable Soft Graph Masking
Keeps the GCN aggregation but replaces the binary adjacency matrix with a learned soft gate. A 2-layer MLP scores each edge from `[h_i; h_j]` and produces a sigmoid weight, allowing the model to suppress noisy parser edges end-to-end.

### E — Contrastive Loss
Augments the standard cross-entropy loss with a supervised contrastive term that pulls together representations of the same sentiment class and pushes apart representations of different classes.

---

## Datasets

All datasets are pre-parsed and included. No additional preprocessing is needed.

| Dataset | Domain | Train | Test | Source |
|---|---|---|---|---|
| rest14 | Restaurant reviews | ~3,608 | ~1,120 | SemEval 2014 Task 4 |
| lap14 | Laptop reviews | ~2,966 | ~1,328 | SemEval 2014 Task 4 |
| rest15 | Restaurant reviews | ~1,315 | ~685 | SemEval 2015 Task 12 |
| rest16 | Restaurant reviews | ~1,240 | ~544 | SemEval 2016 Task 5 |

---

## Requirements

```
Python >= 3.8
PyTorch >= 1.10
transformers >= 4.0   (for B-BERT only)
scikit-learn
numpy
spacy
```

Install dependencies:

```bash
pip install torch transformers scikit-learn numpy spacy
python -m spacy download en_core_web_sm
```

Download GloVe embeddings (required for A, C, D, E):

```bash
# Download from https://nlp.stanford.edu/data/glove.840B.300d.zip
# Extract glove.840B.300d.txt into improvement_X/glove/
```

---

## Usage

### Training a single run

```bash
cd improvement_A_baseline
python train.py --model_name asgcn --dataset rest14 --seed 42 --save True
```

### Training all 4 datasets × 3 seeds

```bash
for dataset in rest14 lap14 rest15 rest16; do
    for seed in 776 42 123; do
        python train.py \
            --model_name asgcn \
            --dataset $dataset \
            --seed $seed \
            --save True \
            2>&1 | tee results_logs/${dataset}_seed${seed}.txt
    done
done
```

### Using the Colab template

Open `01_Teammate_Template_v2.ipynb` in Google Colab. Change `MY_FOLDER` and `MY_NAME` at the top to your assigned values, then run cells 1–7 sequentially. Results are automatically saved back to Google Drive.

### Training arguments

| Argument | Default | Description |
|---|---|---|
| `--model_name` | `asgcn` | Model to train |
| `--dataset` | `twitter` | Dataset: `rest14`, `lap14`, `rest15`, `rest16` |
| `--seed` | `776` | Random seed |
| `--learning_rate` | `0.001` | Adam learning rate |
| `--l2reg` | `1e-5` | L2 regularisation weight |
| `--num_epoch` | `100` | Max training epochs (early stopping at patience=5) |
| `--batch_size` | `32` | Batch size |
| `--hidden_dim` | `300` | BiLSTM and GCN hidden dimension |
| `--embed_dim` | `300` | Input embedding dimension (set to 768 for BERT) |
| `--save` | `False` | Save best model checkpoint |

---

## Results

Mean ± std across seeds 42, 123, 776. Paper values are from Zhang et al. (2019).

### Accuracy (%)

| Model | rest14 | lap14 | rest15 | rest16 |
|---|---|---|---|---|
| Paper (ASGCN) | 80.86 | 75.55 | 79.89 | 88.99 |
| A-Baseline | 81.43 ±0.19 | 75.97 ±0.87 | 79.62 ±0.13 | 88.19 ±0.32 |
| B-BERT | **85.98 ±0.29** | **80.02 ±0.19** | **84.77 ±0.53** | **92.14 ±0.08** |
| C-GAT | 80.80 ±0.25 | 74.96 ±1.05 | 79.83 ±0.39 | 88.49 ±0.34 |
| D-SoftMask | 81.49 ±0.47 | 75.74 ±0.40 | 79.48 ±0.35 | 88.29 ±0.38 |
| E-Contrastive | 81.16 ±0.30 | 75.98 ±0.20 | 79.60 ±0.20 | 88.29 ±0.23 |

### Macro-F1 (%)

| Model | rest14 | lap14 | rest15 | rest16 |
|---|---|---|---|---|
| Paper (ASGCN) | 72.19 | 71.05 | 61.89 | 67.48 |
| A-Baseline | 73.05 ±0.40 | 71.99 ±1.26 | 62.76 ±0.65 | 69.44 ±0.19 |
| B-BERT | **79.80 ±0.29** | **76.48 ±0.33** | **71.46 ±1.90** | **79.04 ±1.57** |
| C-GAT | 71.69 ±0.59 | 70.87 ±1.33 | 62.22 ±0.71 | 68.60 ±1.47 |
| D-SoftMask | 73.03 ±0.24 | 71.67 ±0.67 | 61.16 ±0.17 | 70.61 ±0.85 |
| E-Contrastive | 72.66 ±0.74 | 71.93 ±0.39 | 62.46 ±0.61 | 68.09 ±1.10 |

---

## Key Findings

**B-BERT** is the most impactful improvement, yielding +4–5pp accuracy and +7–10pp F1 across all datasets. Contextual embeddings substantially outperform static GloVe vectors, consistent with published BERT-ABSA literature.

**C-GAT** marginally underperforms the baseline on accuracy. Single-head attention without edge-type awareness does not reliably improve over a well-tuned GCN on clean dependency trees — a finding consistent with prior work.

**D-SoftMask** produces small gains on rest14 and lap14 but shows inconsistent F1 on rest15. The learnable edge gates add an `O(N²)` MLP over limited training data (~1,300 sentences for rest15), which constrains the module's benefit on smaller datasets. A known normalisation interaction between the soft adjacency and the GCN's denominator term also warrants further investigation.

**E-Contrastive** yields modest, consistent improvements in accuracy with low variance, suggesting the contrastive objective regularises training without aggressively reshaping the representation space.

---

## Base Paper

```bibtex
@inproceedings{zhang-etal-2019-aspect,
    title     = "Aspect-based Sentiment Classification with Aspect-specific Graph Convolutional Networks",
    author    = "Zhang, Chen and Li, Qiuchi and Song, Dawei",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing
                 and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month     = nov,
    year      = "2019",
    address   = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url       = "https://aclanthology.org/D19-1464/",
    doi       = "10.18653/v1/D19-1464",
    pages     = "4568--4578",
}
```

---

## Acknowledgements

Base codebase adapted from [GeneZC/ASGCN](https://github.com/GeneZC/ASGCN), which in turn builds on [ABSA-PyTorch](https://github.com/songyouwei/ABSA-PyTorch).
