import torch
from torch import nn

from table_recon_engine.models.decoder import StructureDecoder
from table_recon_engine.models.encoder import VisionEncoder
from table_recon_engine.models.regression_head import RegressionHead


class TSREngine(nn.Module):
    """End-to-end table structure recognition model."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        decoder_layers: int = 3,
        pad_token_id: int = 0,
        max_len: int = 1024,
    ) -> None:
        super().__init__()
        self.encoder = VisionEncoder(out_dim=d_model)
        self.decoder = StructureDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=decoder_layers,
            pad_token_id=pad_token_id,
            max_len=max_len,
        )
        self.regression_head = RegressionHead(d_model=d_model)

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        memory, memory_mask = self.encoder(images)
        logits, decoder_states = self.decoder(
            token_ids=input_ids,
            memory=memory,
            memory_key_padding_mask=memory_mask,
        )
        pred_boxes = self.regression_head(decoder_states)
        return {"logits": logits, "boxes": pred_boxes}
