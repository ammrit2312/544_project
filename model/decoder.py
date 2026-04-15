import torch
import torch.nn as nn
from model.attention import MultiHeadAttention

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff         = FeedForward(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.norm3      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x, encoder_out, src_mask=None, tgt_mask=None):
        # 1. masked self attention
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        # 2. cross attention over encoder output
        x = self.norm2(x + self.dropout(self.cross_attn(x, encoder_out, encoder_out, src_mask)))
        # 3. feed forward
        x = self.norm3(x + self.dropout(self.ff(x)))
        return x


class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, vocab_size, max_len=128, dropout=0.1):
        super().__init__()
        self.embedding   = nn.Embedding(vocab_size, d_model)
        self.pos_embed   = nn.Embedding(max_len, d_model)
        self.layers      = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm        = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)
        self.dropout     = nn.Dropout(dropout)

    def forward(self, tgt, encoder_out, src_mask=None, tgt_mask=None):
        batch, tgt_len = tgt.size()
        positions = torch.arange(tgt_len, device=tgt.device).unsqueeze(0)
        x = self.dropout(self.embedding(tgt) + self.pos_embed(positions))
        for layer in self.layers:
            x = layer(x, encoder_out, src_mask, tgt_mask)
        x = self.norm(x)
        return self.output_proj(x)  # [batch, tgt_len, vocab_size]