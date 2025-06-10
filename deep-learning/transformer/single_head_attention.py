import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim

        # Linear layers for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x: (batch, seq_len, embed_dim)
        B, T, D = x.size()

        # Project Q, K, V
        q = self.q_proj(x)  # (B, T, D)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (D**0.5)  # (B, T, T)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, T, T)
        attn_weights = self.dropout(attn_weights)

        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)  # (B, T, D)

        return self.out_proj(attn_output), attn_scores


if __name__ == "__main__":
    # Example input
    x = torch.rand(2, 5, 32)  # batch=2, seq_len=5, embed_dim=32

    attn = SelfAttention(embed_dim=32)
    out, attn_weights = attn(x)

    print(out.shape)  # → torch.Size([2, 5, 32])
    print(attn_weights.shape)  # → torch.Size([2, 5, 5])
