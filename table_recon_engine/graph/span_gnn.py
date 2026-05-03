from __future__ import annotations

import torch
from torch import nn

from table_recon_engine.graph.grid_graph import EDGE_FEATURE_DIM, NODE_FEATURE_DIM


class SpanEdgeClassifier(nn.Module):
    """Small pure-PyTorch graph network for adjacent-cell merge prediction."""

    def __init__(
        self,
        node_dim: int = NODE_FEATURE_DIM,
        edge_dim: int = EDGE_FEATURE_DIM,
        hidden_dim: int = 128,
        message_layers: int = 3,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.message_layers = message_layers
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.message_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(message_layers)
            ]
        )
        self.update_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                )
                for _ in range(message_layers)
            ]
        )
        self.edge_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return node_features.new_zeros((0,))
        h = self.node_encoder(node_features)
        src = edge_index[0]
        dst = edge_index[1]
        for message_mlp, update_mlp in zip(self.message_mlps, self.update_mlps):
            pair = torch.cat([h[src], h[dst], edge_features], dim=1)
            message = message_mlp(pair)
            aggregated = h.new_zeros(h.shape)
            aggregated.index_add_(0, dst, message)
            aggregated.index_add_(0, src, message)
            degree = h.new_zeros((h.shape[0], 1))
            ones = h.new_ones((message.shape[0], 1))
            degree.index_add_(0, dst, ones)
            degree.index_add_(0, src, ones)
            aggregated = aggregated / degree.clamp_min(1.0)
            h = h + update_mlp(torch.cat([h, aggregated], dim=1))
        logits = self.edge_classifier(torch.cat([h[src], h[dst], edge_features], dim=1)).squeeze(1)
        return logits


def build_model_from_checkpoint(checkpoint: dict[str, object]) -> SpanEdgeClassifier:
    config = dict(checkpoint.get("model_config", {}))
    model = SpanEdgeClassifier(**config)
    model.load_state_dict(checkpoint["model"])
    return model
