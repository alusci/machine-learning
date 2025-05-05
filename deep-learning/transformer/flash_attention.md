⚡ Summary: The Essence of FlashAttention

FlashAttention is a faster, more memory-efficient implementation of the attention mechanism in Transformer models. It computes exact attention (not approximate) using clever memory and math optimizations.

⸻

🚀 Key Techniques

1. Tiling
	•	Breaks large attention matrices into smaller blocks (tiles).
	•	Each tile fits into fast GPU shared memory.
	•	Reduces global memory reads/writes and improves speed.

2. Shared Memory
	•	Uses fast, on-chip shared memory instead of slower global memory.
	•	Allows efficient reuse of data like keys and values across threads.

⸻

🧠 Standard Attention (for context)

Attention(Q, K, V) = softmax(Q × K^T / sqrt(d_k)) × V

	•	Requires computing the full Q × K^T matrix.
	•	Needs large memory, especially for long sequences (O(n²) space).

⸻

✅ FlashAttention’s Softmax Trick

Instead of computing the full attention matrix, FlashAttention processes blocks of keys/values and updates the result incrementally, using a streaming softmax.
For each query row i, it maintains:
	•	m_i: the running maximum logit seen so far.
	•	l_i: the running denominator of softmax (sum of exponentials).
	•	o_i: the running attention output.

Each tile is processed as follows:
	•	Compute partial logits (Q × K_tile^T).
	•	Update m_i, l_i, and o_i using a rescaling trick:
	•	Reweight old values to match the new max.
	•	Add new contributions from the tile.
	•	Final result is numerically stable and memory-efficient.

⸻

💡 Benefits
	•	Trains models with much longer sequences (4K–32K tokens).
	•	Reduces memory usage and improves speed by 2–4×.
	•	Delivers exact results — no approximations.
	•	Used in large models like GPT-4, LLaMA 2, and Mistral.
