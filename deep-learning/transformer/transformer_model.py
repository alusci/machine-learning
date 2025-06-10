import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from multi_head_attention import MultiHeadSelfAttention


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
        # This works thanks to broadcasting
        # Brodcasting works only if the first dimesions is equal or 1
        pe = pe.unsqueeze(0)  # (1, max_len, embed_dim)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x


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


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.ffn = FeedForward(embed_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.attn(x, mask)[0])
        x = self.norm2(x + self.ffn(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self, vocab_size, embed_dim, num_heads, num_layers, max_len=512, dropout=0.1
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len)
        self.layers = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.ln = nn.LayerNorm(embed_dim)
        self.output = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, mask=None):
        x = self.embed(x)
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.ln(x)
        return self.output(x)


if __name__ == "__main__":

    # Model config
    vocab_size = 1000
    embed_dim = 64
    num_heads = 4
    num_layers = 2
    seq_len = 10
    batch_size = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the model
    model = Transformer(
        vocab_size, embed_dim, num_heads, num_layers, max_len=seq_len
    ).to(device)

    # Dummy input: batch of token indices
    input_tokens = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)

    # Optional: Create a causal mask
    def generate_causal_mask(seq_len):
        return torch.tril(torch.ones((1, 1, seq_len, seq_len), device=device))

    mask = generate_causal_mask(seq_len)

    # Run the model
    with torch.no_grad():
        output_logits = model(input_tokens, mask=mask)

    # Output shape: (batch_size, seq_len, vocab_size)
    print("Output shape:", output_logits.shape)
