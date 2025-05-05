import torch
from decoder_only import Decoder

# Toy vocabulary and tokenizer
vocab = {
    "hello": 0, "world": 1, "!": 2, "<bos>": 3, "<eos>": 4, "<pad>": 5,
    "how": 6, "are": 7, "you": 8, "?": 9, "i": 10, "am": 11, "fine": 12
}
inv_vocab = {v: k for k, v in vocab.items()}

def encode(text):
    return [vocab.get(t, vocab["<pad>"]) for t in text.split()]

def decode(tokens):
    return " ".join(inv_vocab.get(t, "<unk>") for t in tokens)

# Initialize model
vocab_size = len(vocab)
embed_dim = 32
num_heads = 4
num_layers = 2
max_len = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Decoder(vocab_size, embed_dim, num_heads, num_layers, max_len).to(device)
model.eval()

# Start with <bos> token
input_ids = torch.tensor([[vocab["<bos>"]]], device=device)

# Generate up to 10 tokens
for _ in range(10):
    seq_len = input_ids.shape[1]

    # Create causal mask
    mask = torch.tril(torch.ones((1, 1, seq_len, seq_len), device=device))

    with torch.no_grad():
        logits = model(input_ids, mask=mask)  # (B, T, V)
        next_token_logits = logits[:, -1, :]  # last token
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)

    input_ids = torch.cat([input_ids, next_token], dim=1)

    # Stop at <eos>
    if next_token.item() == vocab["<eos>"]:
        break

# Print generated tokens
generated_tokens = input_ids[0].tolist()
print("Generated:", decode(generated_tokens))
