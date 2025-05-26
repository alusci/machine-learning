# 🔌 Activation Functions in Neural Networks

Activation functions introduce **non-linearity** into neural networks, allowing them to learn complex functions and patterns. Without them, a neural network would simply be a linear model, no matter how many layers it has.

---

## 🔷 1. ReLU (Rectified Linear Unit)

**Formula:**
```
f(x) = max(0, x)
```

**Pros:**
- Simple and fast
- Helps reduce the vanishing gradient problem
- Works well in practice for many deep learning tasks

**Cons:**
- Can cause the **"dying ReLU"** problem: neurons may output 0 and stop updating

---

## 🔶 2. Leaky ReLU

**Formula:**
```
f(x) = x if x > 0 else αx (typically α = 0.01)
```

**Pros:**
- Mitigates the dying ReLU issue
- Maintains a small gradient for x < 0

**Cons:**
- α is fixed and not learned

---

## 🟢 3. Parametric ReLU (PReLU)

**Formula:**
```
f(x) = x if x > 0 else αx, with α learned during training
```

**Pros:**
- More flexible than Leaky ReLU
- Learnable α adapts to data

**Cons:**
- Slight increase in model complexity due to extra parameters

---

## 🟠 4. ELU (Exponential Linear Unit)

**Formula:**
```
f(x) = x if x > 0 else α * (exp(x) - 1)
```

**Pros:**
- Smooth and differentiable
- Zero-centered outputs (helps convergence)

**Cons:**
- Computationally more expensive than ReLU

---

## 🔵 5. GELU (Gaussian Error Linear Unit)

**Formula (approx):**
```
f(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))
```

**Pros:**
- Smooth, non-linear
- Used in Transformers like BERT and GPT

**Cons:**
- More computational overhead

---

## 🟣 6. Sigmoid

**Formula:**
```
f(x) = 1 / (1 + e^(-x))
```

**Pros:**
- Outputs between 0 and 1, useful for probabilities
- Used in binary classification (final layer)

**Cons:**
- Vanishing gradient for large |x|
- Not zero-centered

---

## 🔘 7. Tanh (Hyperbolic Tangent)

**Formula:**
```
f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
```

**Pros:**
- Zero-centered (better than sigmoid)
- Stronger gradients for inputs near 0

**Cons:**
- Still suffers from vanishing gradients at extremes

---

## 📋 Summary Table

| Activation | Output Range | Zero-Centered | Common Use Cases                |
|------------|--------------|----------------|----------------------------------|
| ReLU       | [0, ∞)       | ❌             | Default for hidden layers        |
| Leaky ReLU | (-∞, ∞)      | ❌             | Robust alternative to ReLU       |
| PReLU      | (-∞, ∞)      | ❌             | Adaptive variant of Leaky ReLU   |
| ELU        | (-α, ∞)      | ✅             | Faster convergence in some cases |
| GELU       | (-∞, ∞)      | ✅             | Transformers (BERT, GPT, etc.)   |
| Sigmoid    | (0, 1)       | ❌             | Output for binary classification |
| Tanh       | (-1, 1)      | ✅             | Older RNNs, shallow networks      |

---

## ✅ Interview Tip

> "I typically start with ReLU because it's simple and effective, but if I encounter dead neurons or I'm working with Transformer architectures, I switch to Leaky ReLU or GELU for smoother gradients and better convergence."