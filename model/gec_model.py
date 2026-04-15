import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from model.decoder import Decoder

class GECModel(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048,
                 num_layers=4, max_len=128, dropout=0.1,
                 freeze_encoder: bool = True):
        super().__init__()

        self.freeze_encoder = freeze_encoder

        self.encoder     = AutoModel.from_pretrained('xlm-roberta-base')
        self.tokenizer   = AutoTokenizer.from_pretrained('xlm-roberta-base')
        vocab_size       = self.tokenizer.vocab_size

        for param in self.encoder.parameters():
            param.requires_grad = not freeze_encoder

        # Project encoder 768-dim output → decoder d_model
        self.projection  = nn.Linear(768, d_model)

        # Your custom decoder
        self.decoder     = Decoder(
            d_model    = d_model,
            num_heads  = num_heads,
            d_ff       = d_ff,
            num_layers = num_layers,
            vocab_size = vocab_size,
            max_len    = max_len,
            dropout    = dropout
        )

        # Weight tying: share embedding and output projection weights
        # Reduces trainable params by ~vocab_size * d_model
        self.decoder.output_proj.weight = self.decoder.embedding.weight

    def encode(self, input_ids, attention_mask):
        if self.freeze_encoder:
            with torch.no_grad():
                encoder_out = self.encoder(
                    input_ids      = input_ids,
                    attention_mask = attention_mask
                ).last_hidden_state
        else:
            encoder_out = self.encoder(
                input_ids      = input_ids,
                attention_mask = attention_mask
            ).last_hidden_state
        return self.projection(encoder_out)

    def make_tgt_mask(self, tgt):
        # Causal mask — decoder can't look ahead
        tgt_len = tgt.size(1)
        mask = torch.tril(torch.ones(tgt_len, tgt_len, device=tgt.device)).unsqueeze(0).unsqueeze(0)
        return mask   # [1, 1, tgt_len, tgt_len]

    def forward(self, input_ids, attention_mask, tgt):
        encoder_out = self.encode(input_ids, attention_mask)
        tgt_mask    = self.make_tgt_mask(tgt)
        return self.decoder(tgt, encoder_out, tgt_mask=tgt_mask)