import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(max_len, d_model)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(1), persistent=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        seq_len = tokens.size(0)
        return self.dropout(tokens + self.encoding[:seq_len])


class StructureDecoder(nn.Module):
    """Transformer decoder for autoregressive HTML structure prediction."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_len: int = 1024,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__()
        self.pad_token_id = pad_token_id
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.positional = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=dropout)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, vocab_size)

    @staticmethod
    def causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if token_ids.dim() != 2:
            raise ValueError(f"token_ids must have shape (B, T), got {token_ids.shape}")

        tgt = token_ids.transpose(0, 1)
        tgt_key_padding_mask = token_ids.eq(self.pad_token_id)
        embeddings = self.embedding(tgt) * math.sqrt(self.embedding.embedding_dim)
        embeddings = self.positional(embeddings)

        hidden = self.decoder(
            tgt=embeddings,
            memory=memory,
            tgt_mask=self.causal_mask(tgt.size(0), tgt.device),
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        hidden = self.output_norm(hidden).transpose(0, 1).contiguous()
        logits = self.classifier(hidden)
        return logits, hidden
