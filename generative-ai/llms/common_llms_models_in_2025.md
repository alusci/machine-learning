📊 Typical Parameter Counts in LLMs

| Model (Public/Reported) | Params (Billions) | Notes |
|-------------------------|-------------------|-------|
| GPT-2 | 1.5B | OpenAI's 2019 model |
| GPT-3 | 175B | First massive LLM to go viral |
| GPT-3.5 (turbo) | ~154B (est.) | Optimized variant of GPT-3 |
| GPT-4 | >500B (est.) | Exact size undisclosed, possibly MoE |
| GPT-4o | ??? | Unreleased, but likely 1T+ total weights, MoE architecture |
| LLaMA 2 (Meta) | 7B / 13B / 70B | Open weights for all |
| Claude 3 (Anthropic) | ~>100B (est.) | Claude 3 Opus likely >175B |
| PaLM (Google) | 540B | Pathways model |
| Gemini 1.5 (Google) | Unknown | Possibly MoE, 1T+ parameters |
| Mistral (7B / Mixtral) | 7B / 12.9B active | Sparse MoE: 12.9B active of 46.7B total |
| Command R+ (Cohere) | ~35B (est.) | Retrieval-augmented model |
| Falcon | 7B / 40B | Trained on RefinedWeb |
| Chinchilla (DeepMind) | 70B | Rebalanced compute/data vs GPT-3 |

---

## 🧠 Key Notes

- **Sparse MoE (Mixture of Experts)**: Models like GPT-4, Mixtral, and Gemini may have trillions of total parameters, but only activate a subset per inference, e.g., 2 of 8 experts → faster & cheaper.
- **Typical Production Models**:
  - General-purpose models: 70B to 540B
  - Efficient/compact models: 7B to 13B
  - Specialized RAG/coding/chat: 13B to 70B

---

## Summary

- Entry-level LLM: 7–13B
- Mid-range production model: 30–70B
- SOTA foundation models: >100B (some sparse MoE >1T parameters)

