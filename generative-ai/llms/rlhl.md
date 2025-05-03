## ✅ Step-by-Step: Tiny GPT Training with RLHF

---

### 🔹 1. Load a Base Model (pretrained small one)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
```

---

### 🔹 2. Supervised Fine-Tuning (SFT)

Let’s say we have instruction-response pairs:

```python
from datasets import Dataset

data = [
    {"prompt": "Translate to French: Hello", "response": "Bonjour"},
    {"prompt": "Summarize: Large Language Models are...", "response": "LLMs are powerful text generators."}
]

# Prepare dataset for fine-tuning
def preprocess(example):
    full_prompt = f"### Instruction:\n{example['prompt']}\n### Response:\n{example['response']}"
    tokens = tokenizer(full_prompt, truncation=True, padding="max_length", max_length=128)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

dataset = Dataset.from_list(data).map(preprocess)
```

---

### 🔹 3. Train with SFT using `Trainer`

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    per_device_train_batch_size=2,
    output_dir="./sft-checkpoint",
    num_train_epochs=3,
    logging_steps=1,
    save_steps=5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer
)

trainer.train()
```

---

### 🔹 4. Add a Reward Model (mocked for demo)

In practice, this is trained separately, often using human comparisons.

```python
def simple_reward_fn(output_text):
    if "Bonjour" in output_text:
        return 1.0
    return 0.0
```

---

### 🔹 5. RLHF with PPO (via `trl`)

```python
from trl import PPOTrainer, PPOConfig
from transformers import pipeline

# Reward pipeline (can be a real model instead)
ref_model = AutoModelForCausalLM.from_pretrained("gpt2")
config = PPOConfig(model_name="gpt2", learning_rate=1e-5, batch_size=2)

ppo_trainer = PPOTrainer(config=config, model=model, ref_model=ref_model, tokenizer=tokenizer)

# Fake prompts to simulate dialogue
prompts = ["Translate to French: Hello", "Summarize: LLMs are..."]

for prompt in prompts:
    query_tensor = tokenizer(prompt, return_tensors="pt").input_ids
    response_tensor = model.generate(query_tensor, max_new_tokens=20)
    response_text = tokenizer.decode(response_tensor[0], skip_special_tokens=True)

    reward = simple_reward_fn(response_text)
    ppo_trainer.step([query_tensor[0]], [response_tensor[0]], [reward])
```

---

## 🔁 Summary

| Phase    | Description                                |
| -------- | ------------------------------------------ |
| SFT      | Teaches the model to follow instructions   |
| Reward   | Trained or hand-crafted reward signal      |
| RLHF/PPO | Optimizes behavior aligned with the reward |


