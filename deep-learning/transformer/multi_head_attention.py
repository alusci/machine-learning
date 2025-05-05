import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Combined Q, K, V projection
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, D = x.size()

        # Project to Q, K, V and reshape for multi-head
        qkv = self.qkv_proj(x)  # (B, T, 3*D)
        q, k, v = qkv.chunk(3, dim=-1)  # Each: (B, T, D)

        def reshape_for_heads(tensor):
            # Reshape to (B, T, nh, hs) and then transpose to (B, nh, T, hs)
            return tensor.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) 
        
        q = reshape_for_heads(q)  # (B, nh, T, hs)
        k = reshape_for_heads(k)
        v = reshape_for_heads(v)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, nh, T, T)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)  # (B, nh, T, hs)

        # Recombine heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)

        return self.out_proj(attn_output), attn_weights
    
if __name__ == "__main__":
    
    # Unmasked example input
    attn = MultiHeadSelfAttention(embed_dim=64, num_heads=8)
    x = torch.rand(2, 10, 64)  # batch=2, seq_len=10, embed_dim=64
    out, attn_weights = attn(x)
    print(out.shape)  # torch.Size([2, 10, 64])
    print(attn_weights.shape)
    print(attn_weights.detach().numpy()[0][0])  # torch.Size([2, 8, 10, 10])

    # Masked example input
    def generate_causal_mask(seq_len, device="cpu"):
        return torch.tril(torch.ones((seq_len, seq_len), device=device))  # float 0/1
    
    mask = generate_causal_mask(10).unsqueeze(0)
    out, attn_weights = attn(x, mask=mask)
    print(out.shape)  # torch.Size([2, 10, 64])
    print(attn_weights.shape)
    print(attn_weights.detach().numpy()[0][0])
