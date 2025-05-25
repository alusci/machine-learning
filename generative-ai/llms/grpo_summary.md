# Group Relative Policy Optimization (GRPO)

## 🧠 What is GRPO?

**Group Relative Policy Optimization (GRPO)** is a reinforcement learning technique used to fine-tune language models for better reasoning capabilities. It was popularized in models like DeepSeek-R1 and removes the need for a critic or value model.

---

## 🎯 Key Concepts

### 1. No Value Model Needed
Unlike PPO, GRPO doesn’t use a value function. Instead, it compares completions in a group by their relative quality.

### 2. Group-Based Scoring
- A group of completions is generated per prompt.
- Each completion receives a heuristic or rule-based reward.
- Scores are compared relatively within the group.

### 3. Relative Advantage
Each completion’s reward is centered relative to the group:

```
A_i = r_i - mean(r)
```

This encourages the model to prefer better-than-average completions.

### 4. Policy Update (REINFORCE with baseline)
The update rule is:

```
L_GRPO = -Σ A_i * log π(y_i | x)
```

This reinforces higher-scoring completions more than lower-scoring ones.

---

## ⚙️ Comparison

| Method | Value Model? | Reward Source | Comparison Style | Stability |
|--------|--------------|----------------|------------------|-----------|
| PPO    | ✅ Yes        | Reward model   | Individual        | Less stable |
| DPO    | ❌ No         | Preference pairs | Pairwise         | Stable |
| GRPO   | ❌ No         | Rule-based or exact-match | Group-relative | Very stable |

---

## 📚 Example: GRPO on GSM8k

1. Generate 4 completions per math problem.
2. Score them:
   - Correct answer: 1.0
   - Partially correct: 0.5
   - Wrong/invalid: 0.0
3. Compute relative advantage per sample.
4. Update the model using reward-weighted log-likelihood.

---

GRPO enables efficient and interpretable reinforcement learning for reasoning tasks in LLMs without requiring complex critic models.


---

# 🔄 GRPO vs PPO: Key Differences

## Core Comparison Table

| Aspect | **PPO (Proximal Policy Optimization)** | **GRPO (Group Relative Policy Optimization)** |
|--------|-----------------------------------------|-----------------------------------------------|
| **Critic/Value Function** | ✅ Required to estimate expected return | ❌ Not needed |
| **Reward Type** | Absolute reward from external model (or human) | Relative score within a group |
| **Update Signal** | Advantage = reward - value estimate | Advantage = reward - group mean |
| **Loss Function** | Uses clipped surrogate objective with advantage estimates from a value model | Uses REINFORCE with mean-normalized group rewards |
| **Sampling** | Single output per prompt | Multiple completions per prompt (group-wise) |
| **Stability** | Can be unstable due to poor value estimation | More stable due to relative normalization |
| **Implementation Complexity** | Higher (requires training a value network and dealing with reward variance) | Lower (no critic, simpler reward handling) |

---

## 🧠 What Does "No Critic Needed" Mean?

In **PPO**, the advantage is calculated as:

```
A_t = R_t - V(s_t)
```

Where `V(s_t)` is a value estimate provided by a separate learned model (critic). This critic needs to be trained alongside the policy and is sensitive to errors, leading to instability.

In **GRPO**, there is no need for a value model. Instead, the advantage is computed **relatively**:

```
A_i = r_i - mean(r)
```

This relative scoring uses the average reward in a group as the baseline, simplifying training and improving stability.

---

## 🧪 Analogy

- **PPO**: “I scored 90, but I expected 95 based on my value function. Not satisfied.”
- **GRPO**: “I scored 90, the group average was 70. I did well!”

GRPO provides a **more grounded and comparative learning signal** without the need for value estimation, making it especially useful in tasks involving multiple reasoning paths or completions.

