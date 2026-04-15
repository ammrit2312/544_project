import math
from typing import Optional

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, d_model]
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)  # [batch, num_heads, seq_len, d_k]

    def scaled_dot_product(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        # Q: [B, H, Q_len, d_k]
        # K: [B, H, K_len, d_k]
        # V: [B, H, K_len, d_k]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            # mask should be broadcastable to [B, H, Q_len, K_len]
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        return torch.matmul(attn, V), attn

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        x, attn = self.scaled_dot_product(Q, K, V, mask)

        batch_size, _, seq_len, _ = x.size()
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_o(x), attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Masked self-attention
        self_attn_out, _ = self.self_attn(x, x, x, self_mask)
        x = self.norm1(x + self.dropout(self_attn_out))

        # Cross-attention: decoder queries attend to encoder keys/values
        cross_attn_out, _ = self.cross_attn(x, encoder_output, encoder_output, cross_mask)
        x = self.norm2(x + self.dropout(cross_attn_out))

        # Feed-forward network
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))

        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, d_model]
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        max_len: int = 5000,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        self.layers = nn.ModuleList(
            [DecoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        self.out_proj = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        # Lower-triangular mask: tokens can only attend to current and previous tokens
        return torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)

    def generate_padding_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [batch, seq_len]
        return (tokens != self.pad_idx).unsqueeze(1).unsqueeze(2)

    def forward(
        self,
        tokens: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        tokens: [batch, tgt_len]
        encoder_output: [batch, src_len, d_model]
        encoder_padding_mask: [batch, 1, 1, src_len] or None
        """
        device = tokens.device
        batch_size, tgt_len = tokens.size()

        x = self.embedding(tokens) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        causal_mask = self.generate_causal_mask(tgt_len, device)  # [1, 1, tgt_len, tgt_len]
        tgt_padding_mask = self.generate_padding_mask(tokens)      # [batch, 1, 1, tgt_len]
        self_mask = causal_mask * tgt_padding_mask

        cross_mask = encoder_padding_mask

        for layer in self.layers:
            x = layer(x, encoder_output, self_mask=self_mask, cross_mask=cross_mask)

        logits = self.out_proj(x)  # [batch, tgt_len, vocab_size]
        return logits


if __name__ == "__main__":
    # Tiny sanity check
    batch_size = 2
    src_len = 7
    tgt_len = 6
    vocab_size = 100
    d_model = 64
    num_heads = 8
    num_layers = 2
    d_ff = 256

    decoder = Decoder(vocab_size, d_model, num_heads, num_layers, d_ff)

    tokens = torch.randint(0, vocab_size, (batch_size, tgt_len))
    encoder_output = torch.randn(batch_size, src_len, d_model)
    encoder_padding_mask = torch.ones(batch_size, 1, 1, src_len)

    out = decoder(tokens, encoder_output, encoder_padding_mask)
    print(out.shape)  # expected: [2, 6, 100]
