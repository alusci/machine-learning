import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from multi_head_attention import MultiHeadSelfAttention


class FeedForward(nn.Module):
    def __init__(self, embed_dim, expansion=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * expansion),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * expansion, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.ffn = FeedForward(embed_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, self_mask=None):
        x = self.norm1(x + self.self_attn(x, self_mask)[0])
        x = self.norm2(x + self.ffn(x))
        return x


class Decoder(nn.Module):
    def __init__(
        self, vocab_size, embed_dim, num_heads, num_layers, max_len=512, dropout=0.1
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len)
        self.layers = nn.ModuleList(
            [DecoderBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.ln = nn.LayerNorm(embed_dim)
        self.output = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, mask=None):
        x = self.embed(x)
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x, self_mask=mask)
        x = self.ln(x)
        return self.output(x)
