# -*- coding: utf-8 -*-
"""
Improvement B: BERT-enhanced ASGCN
------------------------------------
Replaces the GloVe + BiLSTM encoder with a pretrained BERT encoder.
BERT contextual embeddings are passed through the same two GCN layers
and position-weighted attention pooling as the base ASGCN.

Key changes vs baseline:
  - embed + text_lstm  →  BertModel (bert-base-uncased)
  - inputs_cols gains  'bert_token_ids', 'bert_attention_mask', 'bert_token_type_ids'
  - hidden_dim fixed at 768 (BERT hidden size); fc input = 768
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        return output


class ASGCN_BERT(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(ASGCN_BERT, self).__init__()
        self.opt = opt
        # BERT encoder — embedding_matrix unused but kept for API compatibility
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        bert_dim = 768  # bert-base hidden size

        self.gc1 = GraphConvolution(bert_dim, bert_dim)
        self.gc2 = GraphConvolution(bert_dim, bert_dim)
        self.fc  = nn.Linear(bert_dim, opt.polarities_dim)
        self.dropout = nn.Dropout(0.3)

    # ------------------------------------------------------------------ #
    #  Helpers (identical logic to base, adapted for bert_dim)            #
    # ------------------------------------------------------------------ #
    def position_weight(self, x, aspect_double_idx, text_len, aspect_len):
        batch_size, seq_len = x.shape[0], x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        text_len_np  = text_len.cpu().numpy()
        aspect_len_np = aspect_len.cpu().numpy()
        weight = [[] for _ in range(batch_size)]
        for i in range(batch_size):
            context_len = max(text_len_np[i] - aspect_len_np[i], 1)
            for j in range(aspect_double_idx[i, 0]):
                weight[i].append(1 - (aspect_double_idx[i, 0] - j) / context_len)
            for j in range(aspect_double_idx[i, 0], aspect_double_idx[i, 1] + 1):
                weight[i].append(0)
            for j in range(aspect_double_idx[i, 1] + 1, text_len_np[i]):
                weight[i].append(1 - (j - aspect_double_idx[i, 1]) / context_len)
            for j in range(text_len_np[i], seq_len):
                weight[i].append(0)
        weight = torch.tensor(weight, dtype=torch.float).unsqueeze(2).to(self.opt.device)
        return weight * x

    def mask(self, x, aspect_double_idx):
        batch_size, seq_len = x.shape[0], x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        mask = [[] for _ in range(batch_size)]
        for i in range(batch_size):
            for j in range(aspect_double_idx[i, 0]):
                mask[i].append(0)
            for j in range(aspect_double_idx[i, 0], aspect_double_idx[i, 1] + 1):
                mask[i].append(1)
            for j in range(aspect_double_idx[i, 1] + 1, seq_len):
                mask[i].append(0)
        mask = torch.tensor(mask, dtype=torch.float).unsqueeze(2).to(self.opt.device)
        return mask * x

    # ------------------------------------------------------------------ #
    #  Forward                                                             #
    # ------------------------------------------------------------------ #
    def forward(self, inputs):
        # inputs: bert_token_ids, bert_attention_mask, bert_token_type_ids,
        #         aspect_indices, left_indices, dependency_graph
        (bert_token_ids, bert_attention_mask, bert_token_type_ids,
         aspect_indices, left_indices, adj) = inputs

        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        left_len   = torch.sum(left_indices   != 0, dim=-1)
        aspect_double_idx = torch.cat(
            [left_len.unsqueeze(1), (left_len + aspect_len - 1).unsqueeze(1)], dim=1)

        # BERT forward — last_hidden_state: (batch, MAX_SEQ_LEN, 768)
        bert_out = self.bert(
            input_ids      = bert_token_ids,
            attention_mask = bert_attention_mask,
            token_type_ids = bert_token_type_ids,
        ).last_hidden_state
        bert_out = self.dropout(bert_out)

        # adj is (batch, graph_len, graph_len) where graph_len = original text len
        # Trim BERT output to match graph size so GCN matmul dimensions align
        graph_len = adj.size(1)
        text_out  = bert_out[:, :graph_len, :]   # (batch, graph_len, 768)

        # text_len = real tokens capped at graph_len
        text_len = torch.sum(
            bert_attention_mask[:, :graph_len] != 0, dim=1)

        x = F.relu(self.gc1(self.position_weight(text_out, aspect_double_idx, text_len, aspect_len), adj))
        x = F.relu(self.gc2(self.position_weight(x,        aspect_double_idx, text_len, aspect_len), adj))
        x = self.mask(x, aspect_double_idx)

        alpha_mat = torch.matmul(x, text_out.transpose(1, 2))
        alpha     = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        x         = torch.matmul(alpha, text_out).squeeze(1)
        return self.fc(x)