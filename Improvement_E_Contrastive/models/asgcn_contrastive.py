# -*- coding: utf-8 -*-
"""
Improvement E: ASGCN with Supervised Contrastive Loss
-------------------------------------------------------
Augments the standard cross-entropy training with a supervised
contrastive objective (Khosla et al., 2020) applied to the
aspect-level representation produced by ASGCN.

The contrastive head projects the GCN output into a
lower-dimensional L2-normalised space. Representations of the
same sentiment class are pulled together; different classes are
pushed apart. The final loss is:

    L = L_CE + lambda * L_SupCon

where lambda = 0.1 by default (configurable via opt.contrastive_lambda).

Key changes vs baseline:
  - ProjectionHead (2-layer MLP → L2-norm)
  - SupConLoss computes the supervised contrastive loss
  - ASGCN_Contrastive.forward() returns (logits, proj_feat)
  - train.py variant unpacks both and computes combined loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.dynamic_rnn import DynamicLSTM


# ======================================================================
# Graph Convolution (unchanged from base)
# ======================================================================
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


# ======================================================================
# Projection Head for contrastive learning
# ======================================================================
class ProjectionHead(nn.Module):
    """2-layer MLP with ReLU → L2-normalised output."""
    def __init__(self, in_dim, hidden_dim=128, out_dim=64):
        super(ProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        z = self.net(x)
        return F.normalize(z, dim=-1)   # L2-normalise


# ======================================================================
# Supervised Contrastive Loss
# ======================================================================
class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., 2020).
    Works on a batch of L2-normalised features z (B, D) and
    integer labels y (B,).

    For each anchor i:
        L_i = -1/|P(i)| * sum_{p in P(i)} log(
                  exp(z_i · z_p / tau) /
                  sum_{a != i} exp(z_i · z_a / tau)
              )
    """
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.tau = temperature

    def forward(self, features, labels):
        """
        features : (B, D)  — L2-normalised
        labels   : (B,)    — integer class labels
        """
        B = features.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=features.device)

        # Similarity matrix (B, B)
        sim = torch.matmul(features, features.T) / self.tau

        # Mask out self-comparisons
        self_mask = torch.eye(B, dtype=torch.bool, device=features.device)
        sim.masked_fill_(self_mask, -9e15)

        # Log-softmax over all negatives+positives
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)

        # Positive mask: same label, not self
        labels   = labels.unsqueeze(1)
        pos_mask = (labels == labels.T) & ~self_mask

        # Average log-prob over positives
        pos_count = pos_mask.sum(dim=1).clamp(min=1)
        loss = -(log_prob * pos_mask.float()).sum(dim=1) / pos_count
        return loss.mean()


# ======================================================================
# ASGCN_Contrastive Model
# ======================================================================
class ASGCN_Contrastive(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(ASGCN_Contrastive, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float))
        self.text_lstm = DynamicLSTM(
            opt.embed_dim, opt.hidden_dim,
            num_layers=1, batch_first=True, bidirectional=True)

        dim = 2 * opt.hidden_dim
        self.gc1  = GraphConvolution(dim, dim)
        self.gc2  = GraphConvolution(dim, dim)
        self.fc   = nn.Linear(dim, opt.polarities_dim)
        self.proj = ProjectionHead(dim, hidden_dim=128, out_dim=64)
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
        mask_list = [[] for _ in range(batch_size)]
        for i in range(batch_size):
            for j in range(aspect_double_idx[i, 0]):
                mask_list[i].append(0)
            for j in range(aspect_double_idx[i, 0], aspect_double_idx[i, 1] + 1):
                mask_list[i].append(1)
            for j in range(aspect_double_idx[i, 1] + 1, seq_len):
                mask_list[i].append(0)
        mask = torch.tensor(mask_list, dtype=torch.float).unsqueeze(2).to(self.opt.device)
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

        x = F.relu(self.gc1(self.position_weight(text_out, aspect_double_idx, text_len, aspect_len), adj))
        x = F.relu(self.gc2(self.position_weight(x,        aspect_double_idx, text_len, aspect_len), adj))
        x = self.mask(x, aspect_double_idx)

        alpha_mat = torch.matmul(x, text_out.transpose(1, 2))
        alpha     = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        feat      = torch.matmul(alpha, text_out).squeeze(1)   # (B, 2H)

        logits = self.fc(feat)
        proj   = self.proj(feat)   # (B, 64) — used for contrastive loss
        return logits, proj