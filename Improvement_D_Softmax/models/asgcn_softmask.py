# -*- coding: utf-8 -*-
"""
Improvement D: ASGCN with Learnable Soft Graph Masking
--------------------------------------------------------
Replaces the hard binary dependency adjacency matrix with a
*soft mask* learned end-to-end. A small MLP scores each edge
(h_i, h_j) and produces a soft weight in (0,1) via sigmoid.
This soft-masked adjacency is then used in the standard GCN
aggregation, allowing the model to suppress noisy parser edges
and amplify informative ones.

Reference idea: Zhang et al., "Relational Graph Attention Network
for Aspect-Based Sentiment Classification" (2020).

Key changes vs baseline:
  - New SoftMaskLayer computes element-wise gate on adj
  - Two GCN layers now receive soft-adj instead of binary adj
  - Everything else unchanged
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.dynamic_rnn import DynamicLSTM


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
        denom  = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        return output


class SoftMaskLayer(nn.Module):
    """
    Computes a soft gate for each edge in the adjacency matrix.
    For each pair (i, j) with adj[i,j] = 1:
        gate(i,j) = sigmoid( MLP([h_i; h_j]) )
    Returns adj_soft = adj * gate  (non-edges stay zero).
    """
    def __init__(self, hidden_dim):
        super(SoftMaskLayer, self).__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h, adj):
        """
        h   : (B, N, H)
        adj : (B, N, N)
        """
        B, N, H = h.shape
        h_i = h.unsqueeze(2).expand(-1, -1, N, -1)   # (B, N, N, H)
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1)   # (B, N, N, H)
        edge_feat = torch.cat([h_i, h_j], dim=-1)     # (B, N, N, 2H)
        gate = torch.sigmoid(self.edge_mlp(edge_feat).squeeze(-1))  # (B,N,N)
        adj_soft = adj * gate
        return adj_soft


class ASGCN_SoftMask(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(ASGCN_SoftMask, self).__init__()
        self.opt  = opt
        self.embed = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float))
        self.text_lstm = DynamicLSTM(
            opt.embed_dim, opt.hidden_dim,
            num_layers=1, batch_first=True, bidirectional=True)

        dim = 2 * opt.hidden_dim
        self.soft_mask = SoftMaskLayer(dim)
        self.gc1 = GraphConvolution(dim, dim)
        self.gc2 = GraphConvolution(dim, dim)
        self.fc  = nn.Linear(dim, opt.polarities_dim)
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

        text = self.embed(text_indices)
        text = self.text_embed_dropout(text)
        text_out, _ = self.text_lstm(text, text_len)

        # Compute soft adjacency from LSTM output
        adj_soft = self.soft_mask(text_out, adj)

        x = F.relu(self.gc1(self.position_weight(text_out, aspect_double_idx, text_len, aspect_len), adj_soft))
        x = F.relu(self.gc2(self.position_weight(x,        aspect_double_idx, text_len, aspect_len), adj_soft))
        x = self.mask(x, aspect_double_idx)

        alpha_mat = torch.matmul(x, text_out.transpose(1, 2))
        alpha     = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        x         = torch.matmul(alpha, text_out).squeeze(1)
        return self.fc(x)