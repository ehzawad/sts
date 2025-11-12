"""Unified tag classifier architecture (embedding + pattern branches)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class UnifiedTagClassifier(nn.Module):
    """
    Two-branch architecture (embedding + pattern) fused into final logits.
    """

    def __init__(self, embedding_dim=1024, pattern_dim=390, num_tags=195, dropout=0.5):
        super().__init__()

        # Embedding branch
        self.emb_fc1 = nn.Linear(embedding_dim, 512)
        self.emb_bn1 = nn.BatchNorm1d(512)
        self.emb_fc2 = nn.Linear(512, 256)
        self.emb_bn2 = nn.BatchNorm1d(256)

        # Pattern branch
        self.pattern_fc1 = nn.Linear(pattern_dim, 128)
        self.pattern_bn1 = nn.BatchNorm1d(128)
        self.pattern_fc2 = nn.Linear(128, 64)
        self.pattern_bn2 = nn.BatchNorm1d(64)

        # Fusion
        fusion_dim = 256 + 64
        self.fusion_fc1 = nn.Linear(fusion_dim, 128)
        self.fusion_bn1 = nn.BatchNorm1d(128)
        self.output = nn.Linear(128, num_tags)

        self.dropout = nn.Dropout(dropout)

    def forward(self, embeddings, patterns):
        # Embedding branch
        emb = F.relu(self.emb_bn1(self.emb_fc1(embeddings)))
        emb = self.dropout(emb)
        emb = F.relu(self.emb_bn2(self.emb_fc2(emb)))
        emb = self.dropout(emb)

        # Pattern branch
        pat = F.relu(self.pattern_bn1(self.pattern_fc1(patterns)))
        pat = self.dropout(pat)
        pat = F.relu(self.pattern_bn2(self.pattern_fc2(pat)))
        pat = self.dropout(pat)

        combined = torch.cat([emb, pat], dim=1)
        fused = F.relu(self.fusion_bn1(self.fusion_fc1(combined)))
        fused = self.dropout(fused)

        return self.output(fused)
