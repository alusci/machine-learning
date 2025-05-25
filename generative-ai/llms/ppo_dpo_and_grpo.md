# LLM Alignment Methods Comparison

## 🥇 PPO – Proximal Policy Optimization

**Type:** Reinforcement Learning from Human Feedback (RLHF)

**Goal:** Maximize a reward signal (e.g., from a reward model trained on human preferences) while keeping the updated policy close to the original model.

### How it works:
- Reward model is trained on human preferences (e.g., ranking completions).
- The policy (language model) is updated using policy gradients, with a regularization term to prevent the model from drifting too far from the original (reference) policy.
- Uses KL-divergence to constrain the updated policy.

### Used in:
- OpenAI's InstructGPT
- Anthropic's Claude fine-tuning pipeline (initially)

### Pros:
- Strong theoretical foundation.
- Good control over policy drift.

### Cons:
- Complex to implement (requires a full RL loop).
- Expensive to train.

---

## 🥈 DPO – Direct Preference Optimization

**Type:** Direct loss-based fine-tuning

**Goal:** Learn from preference data (e.g., ranked pairs) without needing a separate reward model or full RL setup.

### How it works:
- Uses a pairwise preference dataset (chosen vs. rejected completions).
- Minimizes a loss function that encourages the model to assign higher log-probability to the preferred outputs.
- Includes a KL regularization term to maintain closeness to the base model.

### Key advantage:
- No need to train a reward model.
- Simpler than PPO, more efficient.

### Pros:
- Easier to implement than PPO.
- Works well on large models with minimal tuning.
- Competitive performance with PPO.

### Cons:
- Assumes pairwise data (vs. scalar rewards).
- Less flexible than PPO for arbitrary reward functions.

---

## 🥉 GRPO – Generalized Rejection Sampling Policy Optimization

**Type:** Hybrid of DPO and reward modeling

**Goal:** Generalize DPO by allowing arbitrary reward functions rather than just pairwise preferences.

### How it works:
- Instead of being limited to binary "chosen vs. rejected" pairs, GRPO can operate with general reward distributions.
- Can approximate DPO when using pairwise preferences, or PPO when using scalar rewards.
- Optimizes a Bayesian posterior-like objective, derived from probabilistic principles.

### Pros:
- General framework—subsumes DPO and approaches PPO.
- More flexible for multi-signal or real-valued feedback.

### Cons:
- More complex than DPO.
- Newer, fewer public implementations and benchmarks.

---

## Summary Table

| Method | Type | Needs Reward Model? | Uses Pairs? | Complexity | Flexibility |
|--------|------|---------------------|------------|------------|-------------|
| PPO | RLHF | ✅ Yes | ❌ No | High | High |
| DPO | Supervised (loss-based) | ❌ No | ✅ Yes | Low | Medium |
| GRPO | Probabilistic optimization | ❌ (flexible input) | Optional | Medium | High |

