## 🔧 **Generation Control Parameters**

### 1. 🥵 **Temperature**
- **What it is**: Controls the **randomness** of the output.
- **Range**: Typically `0.0` to `1.0` (sometimes up to 2.0).
- **How it works**: Higher temperature means more random sampling from the probability distribution over tokens.

| Temperature | Behavior |
|-------------|----------|
| 0.0         | Deterministic and focused (always picks the highest probability token) |
| 0.7         | Balanced creativity |
| 1.0+        | Creative and diverse, but can become incoherent |

> 🔍 **Use when**: You want more **creative** or **diverse** output. Lower temp for factual consistency, higher for brainstorming.

---

### 2. 🎲 **Top P (Nucleus Sampling)**
- **What it is**: Limits token choices to the **smallest set** whose cumulative probability is ≥ P.
- **Range**: `0.0` to `1.0`
- **How it works**: Instead of picking from all possible tokens, it only samples from the top-N most probable ones that add up to probability `p`.

| Top P | Behavior |
|-------|----------|
| 0.9   | Only consider top 90% of cumulative token probability |
| 1.0   | No filtering — all tokens can be sampled (similar to not using Top P) |

> 🔍 **Use when**: You want to **limit the randomness** to only the most likely tokens. Lower `top_p` = tighter, more deterministic results.

---

### 3. 📏 **Max Length / Max Tokens**
- **What it is**: The **maximum number of tokens** the model is allowed to generate.
- **Includes**: All output tokens, not input.

> 🔍 **Use when**: You want to **limit verbosity** or **control response length**. Essential in production to avoid runaway generations or cost overruns.

---

### 4. 🛑 **Stop Sequences**
- **What it is**: A list of tokens/strings that **halt generation** when encountered.
- **Example**: `["\nUser:", "</end>", "###"]`

> 🔍 **Use when**: You're designing **chat-like** interactions, or want to **truncate at a certain delimiter** (like end of a sentence or section).

---

### 5. 🔁 **Frequency Penalty**
- **What it is**: Penalizes **repetition of tokens** that have already been generated.
- **Range**: `-2.0` to `2.0` (OpenAI)
- **Effect**: Higher value = model less likely to repeat same lines or words.

> 🔍 **Use when**: You're getting **repetitive output** (e.g. "The cat is... the cat is... the cat is...").

---

### 6. 🌍 **Presence Penalty**
- **What it is**: Encourages the model to **introduce new concepts** by penalizing already-mentioned ones.
- **Range**: `-2.0` to `2.0`
- **Effect**: Higher value = model avoids reusing ideas, boosting novelty.

> 🔍 **Use when**: You want the model to **diversify topics** or **avoid looping** around the same ideas.

---

## 🔁 **How They Work Together**
You typically **tune `temperature` and `top_p` together**, and use **penalties** when output feels repetitive or boring.

| Use Case                 | Recommended Settings                       |
|--------------------------|--------------------------------------------|
| Factual QA               | `temperature=0`, `top_p=1.0`               |
| Creative writing         | `temperature=0.9`, `top_p=0.9`             |
| Chatbots                 | `temperature=0.7`, `frequency_penalty=0.5` |
| Structured answers       | `stop=["\n"]` or `stop=["###"]`            |

