## 🧱 1. Model Quantization

### 🔍 What is it?

Quantization reduces the precision of model weights and activations (e.g., from float32 to int8 or float16), shrinking memory and speeding up inference.

### 🔢 Common Types:

| Type | Weights/Activations | Typical Use |
|------|---------------------|-------------|
| FP32 (baseline) | 32-bit float | Full precision |
| FP16 / BF16 | 16-bit float | Fast GPU inference |
| INT8 | 8-bit integer | Edge devices, CPUs |
| 4-bit (e.g. GPTQ) | 4-bit quantized weights | Squeeze massive models |

### ✅ Benefits
- 2–4× smaller memory footprint
- Faster inference
- Enables on-device LLMs (e.g., mobile, laptops)

### ⚠️ Tradeoffs
- Can lose some accuracy or generate slightly worse text
- Requires calibration (or quantization-aware training)

### 🛠 Tools
- bitsandbytes (4-bit/8-bit for HuggingFace)
- onnxruntime, TensorRT, Intel Neural Compressor
- torch.quantization, ggml, mlc-llm

---

## 📘 2. Model Distillation

### 🔍 What is it?

Distillation is the process of training a smaller model (student) to mimic a larger, pre-trained model (teacher).

### 🔧 How it works:
- Student model is trained on the logits, soft labels, or embeddings of the teacher, not raw data alone.
- Goal: capture teacher's behavior in a smaller, faster model.

### ✅ Benefits
- Keeps performance close to the teacher
- Greatly reduces model size and latency
- Improves generalization due to softened supervision

### 🧠 Real-World Examples

| Student Model | Teacher Model |
|---------------|---------------|
| DistilBERT | BERT-base |
| TinyLlama | LLaMA-2 |
| DistilGPT-2 | GPT-2 |
| MiniLM | RoBERTa / BERT |
| OpenChat (3.5/4) | GPT-3.5-turbo / GPT-4 |

---

## 🔁 Distillation vs Quantization

| Aspect | Quantization | Distillation |
|--------|-------------|--------------|
| Changes Weights | Yes (numerically) | Yes (re-trained) |
| Training Needed | Sometimes | Yes (train student) |
| Speedup | High | Medium-High |
| Accuracy Loss | Possible (minor) | Often less than quant. |
