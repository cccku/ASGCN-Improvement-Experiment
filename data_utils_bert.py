# -*- coding: utf-8 -*-
"""
data_utils_bert.py  —  Improvement B
--------------------------------------
Extends the base ABSADatesetReader to also produce BERT token ids,
attention masks, and token type ids for each sample.

BERT tokenization strategy for ABSA:
  [CLS] text_left aspect text_right [SEP] aspect [SEP]
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^
        segment A (token_type_id=0)       segment B (=1)

This mirrors the standard BERT-for-ABSA input format (Xu et al., 2019).
Segment A = full sentence, Segment B = aspect term.

The existing word-index fields (text_indices, aspect_indices, etc.) and
dependency_graph / dependency_tree are preserved so that the same
BucketIterator and dependency graph infrastructure can be reused.
"""

import os
import pickle
import numpy as np
from transformers import BertTokenizer

# ── reuse helpers from base ────────────────────────────────────────────────
from data_utils import (
    Tokenizer,
    build_embedding_matrix,
    ABSADataset,
)

# Resolve dataset path relative to this file (project root)
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'datasets')

BERT_MODEL_NAME = 'bert-base-uncased'
MAX_SEQ_LEN     = 85   # covers >99% of ABSA sentences; increase if you see warnings


class ABSADatesetReaderBert:
    """
    Drop-in replacement for ABSADatesetReader for improvement B.
    Produces all original fields PLUS:
        bert_token_ids       : LongTensor (MAX_SEQ_LEN,)
        bert_attention_mask  : LongTensor (MAX_SEQ_LEN,)
        bert_token_type_ids  : LongTensor (MAX_SEQ_LEN,)
    """

    # ------------------------------------------------------------------ #
    @staticmethod
    def __read_text__(fnames):
        text = ''
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 3):
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect   = lines[i + 1].lower().strip()
                text_raw = text_left + " " + aspect + " " + text_right
                text    += text_raw + " "
        return text

    # ------------------------------------------------------------------ #
    @staticmethod
    def __read_data__(fname, tokenizer, bert_tokenizer, max_seq_len):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()

        with open(fname + '.graph', 'rb') as f:
            idx2graph = pickle.load(f)
        with open(fname + '.tree', 'rb') as f:
            idx2tree  = pickle.load(f)

        all_data = []
        for i in range(0, len(lines), 3):
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect   = lines[i + 1].lower().strip()
            polarity = int(lines[i + 2].strip()) + 1

            # ── original word-index fields (unchanged) ─────────────────
            text_indices     = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)
            context_indices  = tokenizer.text_to_sequence(text_left + " " + text_right)
            aspect_indices   = tokenizer.text_to_sequence(aspect)
            left_indices     = tokenizer.text_to_sequence(text_left)
            dependency_graph = idx2graph[i]
            dependency_tree  = idx2tree[i]

            # ── BERT tokenization ──────────────────────────────────────
            # Format: [CLS] sentence [SEP] aspect [SEP]
            #   segment A = sentence tokens  (token_type_id = 0)
            #   segment B = aspect tokens    (token_type_id = 1)
            sentence = (text_left + " " + aspect + " " + text_right).strip()
            enc = bert_tokenizer(
                sentence,
                aspect,
                max_length          = max_seq_len,
                padding             = 'max_length',
                truncation          = True,
                return_tensors      = None,   # return plain Python lists
                return_token_type_ids = True,
            )
            bert_token_ids      = enc['input_ids']
            bert_attention_mask = enc['attention_mask']
            bert_token_type_ids = enc['token_type_ids']

            data = {
                # original fields
                'text_indices'      : text_indices,
                'context_indices'   : context_indices,
                'aspect_indices'    : aspect_indices,
                'left_indices'      : left_indices,
                'polarity'          : polarity,
                'dependency_graph'  : dependency_graph,
                'dependency_tree'   : dependency_tree,
                # BERT fields
                'bert_token_ids'      : bert_token_ids,
                'bert_attention_mask' : bert_attention_mask,
                'bert_token_type_ids' : bert_token_type_ids,
            }
            all_data.append(data)
        return all_data

    # ------------------------------------------------------------------ #
    def __init__(self, dataset='rest14', embed_dim=300):
        print(f"[BERT] preparing {dataset} dataset ...")

        fname = {
            'twitter': {
                'train': os.path.join(DATASET_DIR, 'acl-14-short-data', 'train.raw'),
                'test' : os.path.join(DATASET_DIR, 'acl-14-short-data', 'test.raw'),
            },
            'rest14': {
                'train': os.path.join(DATASET_DIR, 'semeval14', 'restaurant_train.raw'),
                'test' : os.path.join(DATASET_DIR, 'semeval14', 'restaurant_test.raw'),
            },
            'lap14': {
                'train': os.path.join(DATASET_DIR, 'semeval14', 'laptop_train.raw'),
                'test' : os.path.join(DATASET_DIR, 'semeval14', 'laptop_test.raw'),
            },
            'rest15': {
                'train': os.path.join(DATASET_DIR, 'semeval15', 'restaurant_train.raw'),
                'test' : os.path.join(DATASET_DIR, 'semeval15', 'restaurant_test.raw'),
            },
            'rest16': {
                'train': os.path.join(DATASET_DIR, 'semeval16', 'restaurant_train.raw'),
                'test' : os.path.join(DATASET_DIR, 'semeval16', 'restaurant_test.raw'),
            },
        }

        # ── word-index tokenizer (reused from base for non-BERT fields) ─
        text = ABSADatesetReaderBert.__read_text__(
            [fname[dataset]['train'], fname[dataset]['test']])

        word2idx_file = dataset + '_word2idx.pkl'
        if os.path.exists(word2idx_file):
            print(f"loading {dataset} tokenizer...")
            with open(word2idx_file, 'rb') as f:
                word2idx  = pickle.load(f)
                tokenizer = Tokenizer(word2idx=word2idx)
        else:
            tokenizer = Tokenizer()
            tokenizer.fit_on_text(text)
            with open(word2idx_file, 'wb') as f:
                pickle.dump(tokenizer.word2idx, f)

        # embedding_matrix kept for API compatibility (not used by BERT model)
        self.embedding_matrix = build_embedding_matrix(
            tokenizer.word2idx, embed_dim, dataset)

        # ── BERT tokenizer ──────────────────────────────────────────────
        print(f"loading {BERT_MODEL_NAME} tokenizer ...")
        bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

        self.train_data = ABSADataset(
            ABSADatesetReaderBert.__read_data__(
                fname[dataset]['train'], tokenizer, bert_tokenizer, MAX_SEQ_LEN))
        self.test_data  = ABSADataset(
            ABSADatesetReaderBert.__read_data__(
                fname[dataset]['test'],  tokenizer, bert_tokenizer, MAX_SEQ_LEN))

        print(f"  train samples : {len(self.train_data)}")
        print(f"  test  samples : {len(self.test_data)}")