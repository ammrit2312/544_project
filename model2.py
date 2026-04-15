from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel

from attention2 import Decoder


class GECModel(nn.Module):
    """
    Full grammar error correction model:
      encoder (frozen XLM-R / mXLM-R) -> projection -> custom Transformer decoder

    Expected usage:
      - source_ids/source_mask: noisy sentence tokens for the encoder
      - target_ids: shifted-right decoder input tokens
      - encoder_output is produced internally by the encoder
    """

    def __init__(
        self,
        encoder_name: str,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        pad_idx: int,
        freeze_encoder: bool = True,
        max_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(encoder_name)
        encoder_hidden_size = self.encoder.config.hidden_size

        self.proj = nn.Linear(encoder_hidden_size, d_model)
        self.decoder = Decoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout,
            pad_idx=pad_idx,
        )

        self.freeze_encoder = freeze_encoder
        if self.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def encode(
        self,
        source_ids: torch.Tensor,
        source_mask: Optional[torch.Tensor] = None,
    ):
        """
        source_ids: [batch, src_len]
        source_mask: [batch, src_len] with 1 for real tokens and 0 for pad tokens
        """
        encoder_outputs = self.encoder(
            input_ids=source_ids,
            attention_mask=source_mask,
        )
        hidden = encoder_outputs.last_hidden_state  # [batch, src_len, encoder_hidden_size]
        hidden = self.proj(hidden)                  # [batch, src_len, d_model]
        return hidden

    def build_encoder_padding_mask(
        self,
        source_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if source_mask is None:
            return None
        # Decoder cross-attention expects broadcastable shape: [batch, 1, 1, src_len]
        return source_mask.unsqueeze(1).unsqueeze(2)

    def forward(
        self,
        source_ids: torch.Tensor,
        target_ids: torch.Tensor,
        source_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        source_ids: [batch, src_len]
        target_ids: [batch, tgt_len]   (decoder input, typically shifted right)
        source_mask: [batch, src_len]   (1 for real tokens, 0 for pad)

        returns:
            logits: [batch, tgt_len, vocab_size]
        """
        encoder_output = self.encode(source_ids, source_mask)
        encoder_padding_mask = self.build_encoder_padding_mask(source_mask)
        logits = self.decoder(
            tokens=target_ids,
            encoder_output=encoder_output,
            encoder_padding_mask=encoder_padding_mask,
        )
        return logits

    @torch.no_grad()
    def generate(
        self,
        source_ids: torch.Tensor,
        source_mask: Optional[torch.Tensor],
        bos_id: int,
        eos_id: int,
        max_new_tokens: int = 64,
    ) -> torch.Tensor:
        """
        Greedy decoding for inference.

        source_ids: [batch, src_len]
        source_mask: [batch, src_len]

        returns:
            generated token ids: [batch, generated_len]
        """
        self.eval()
        device = source_ids.device

        encoder_output = self.encode(source_ids, source_mask)
        encoder_padding_mask = self.build_encoder_padding_mask(source_mask)

        batch_size = source_ids.size(0)
        generated = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            logits = self.decoder(
                tokens=generated,
                encoder_output=encoder_output,
                encoder_padding_mask=encoder_padding_mask,
            )
            next_token_logits = logits[:, -1, :]  # [batch, vocab_size]
            next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_tokens], dim=1)

            if torch.all(next_tokens.squeeze(1) == eos_id):
                break

        return generated


if __name__ == "__main__":
    # Small sanity check
    batch_size = 2
    src_len = 8
    tgt_len = 6

    model = GECModel(
        encoder_name="xlm-roberta-base",
        vocab_size=250002,
        d_model=256,
        num_heads=8,
        num_layers=2,
        d_ff=1024,
        pad_idx=1,
        freeze_encoder=True,
        max_len=512,
    )

    source_ids = torch.randint(0, 100, (batch_size, src_len))
    target_ids = torch.randint(0, 100, (batch_size, tgt_len))
    source_mask = torch.ones(batch_size, src_len, dtype=torch.long)

    logits = model(source_ids, target_ids, source_mask)
    print(logits.shape)  # expected: [2, 6, 250002]
