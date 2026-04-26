# -*- coding: utf-8 -*-
"""
Improvement C: ASGCN with Graph Attention Network (GAT)
---------------------------------------------------------
Replaces the vanilla symmetric GCN aggregation with a single-head
Graph Attention layer (Veličković et al., 2018). Attention coefficients
are learned per edge, so the model can down-weight noisy dependency arcs.

Key changes vs baseline:
  - GraphConvolution  →  GraphAttentionLayer
  - Everything else (BiLSTM encoder, position weighting, masking,
    attention pooling, fc) is identical to ASGCN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.dynamic_rnn import DynamicLSTM


class GraphAttentionLayer(nn.Module):
    """
    Single-head Graph Attention layer.
    Computes   e_ij = LeakyReLU( a^T [W h_i || W h_j] )
    then normalises with softmax over in-neighbours (masked by adj > 0).
    """
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.W      = nn.Linear(in_features, out_features, bias=False)
        self.a      = nn.Parameter(torch.FloatTensor(2 * out_features, 1))
        self.leaky  = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a.data.view(1, -1))

    def forward(self, h, adj):
        """
        h   : (B, N, in_features)
        adj : (B, N, N)  — 1 where edge exists, 0 otherwise
        """
        Wh = self.W(h)                              # (B, N, out)
        B, N, D = Wh.shape

        # Pairwise concatenation: (B, N, N, 2F)
        Wh_i = Wh.unsqueeze(2).expand(-1, -1, N, -1)
        Wh_j = Wh.unsqueeze(1).expand(-1, N, -1, -1)
        e = self.leaky(torch.cat([Wh_i, Wh_j], dim=-1).matmul(self.a).squeeze(-1))
        # e: (B, N, N)

        # Mask out non-edges with -inf before softmax
        zero_vec   = -9e15 * torch.ones_like(e)
        attention  = torch.where(adj > 0, e, zero_vec)
        attention  = F.softmax(attention, dim=2)         # row-wise
        attention  = self.dropout(attention)

        return torch.matmul(attention, Wh)               # (B, N, out)


class ASGCN_GAT(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(ASGCN_GAT, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float))
        self.text_lstm = DynamicLSTM(
            opt.embed_dim, opt.hidden_dim,
            num_layers=1, batch_first=True, bidirectional=True)

        dim = 2 * opt.hidden_dim
        self.gat1 = GraphAttentionLayer(dim, dim, dropout=0.1)
        self.gat2 = GraphAttentionLayer(dim, dim, dropout=0.1)
        self.fc   = nn.Linear(dim, opt.polarities_dim)
        self.text_embed_dropout = nn.Dropout(0.3)

    # ------------------------------------------------------------------ #
    def position_weight(self, x, aspect_double_idx, text_len, aspect_len):
        batch_size, seq_len = x.shape[0], x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        text_len_np   = text_len.cpu().numpy()
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
    def forward(self, inputs):
        text_indices, aspect_indices, left_indices, adj = inputs
        text_len   = torch.sum(text_indices   != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        left_len   = torch.sum(left_indices   != 0, dim=-1)
        aspect_double_idx = torch.cat(
            [left_len.unsqueeze(1), (left_len + aspect_len - 1).unsqueeze(1)], dim=1)

        text     = self.embed(text_indices)
        text     = self.text_embed_dropout(text)
        text_out, _ = self.text_lstm(text, text_len)

        x = F.relu(self.gat1(self.position_weight(text_out, aspect_double_idx, text_len, aspect_len), adj))
        x = F.relu(self.gat2(self.position_weight(x,        aspect_double_idx, text_len, aspect_len), adj))
        x = self.mask(x, aspect_double_idx)

        alpha_mat = torch.matmul(x, text_out.transpose(1, 2))
        alpha     = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        x         = torch.matmul(alpha, text_out).squeeze(1)
        return self.fc(x)