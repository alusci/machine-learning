# LLM Evaluation

## 🔍 1. Benchmark Datasets

### 📚 General Reasoning & Knowledge
- **MMLU (Massive Multitask Language Understanding)**
  - Covers 57 tasks across domains like history, law, STEM, etc.
  - Goal: Test general knowledge and reasoning.
  - Metric: Accuracy
- **ARC (AI2 Reasoning Challenge)**
  - Science questions from elementary/middle school exams.
  - Two variants: Easy and Challenge sets.
  - Metric: Accuracy
- **OpenBookQA**
  - Multiple-choice science questions, requiring external knowledge.
  - Metric: Accuracy
- **BoolQ**
  - Yes/no questions from real queries, with evidence passages.
  - Metric: Accuracy
- **HellaSwag**
  - Commonsense reasoning for next-sentence prediction.
  - Metric: Accuracy

### 📖 Reading Comprehension / QA
- **SQuAD v2.0**
  - Extractive QA with unanswerable questions included.
  - Metrics: F1, EM (Exact Match)
- **Natural Questions (NQ)**
  - Real questions from Google search, requiring document-level understanding.
  - Metrics: F1, Recall@k (for long answers)
- **TriviaQA**
  - QA dataset of trivia-style questions.
  - Metric: Accuracy
- **HotpotQA**
  - Multihop QA with sentence-level supporting facts.
  - Metrics: F1, EM, Supporting Fact F1

### 🧠 Advanced Reasoning & Math
- **GSM8K**
  - Grade-school math word problems.
  - Metric: Accuracy (often with CoT prompting)
- **MATH**
  - Harder math problems across algebra, calculus, geometry, etc.
  - Metric: Accuracy
- **DROP**
  - Reading comprehension requiring discrete reasoning (e.g., date differences).
  - Metrics: EM, F1

### 🌍 Multilingual Understanding
- **XGLUE / XTREME / XTREME-UP**
  - Cross-lingual understanding and generation across many languages.
  - Tasks: QA, NER, classification, retrieval.
  - Metric: Task-dependent (F1, accuracy, etc.)
- **FLORES-200**
  - Used for multilingual machine translation evaluation.
  - Metric: BLEU

### 📄 Text Generation & Summarization
- **CNN/DailyMail, XSum**
  - Summarization datasets with varying abstraction levels.
  - Metric: ROUGE, BLEU
- **WMT (Machine Translation)**
  - Annual benchmark for translation quality.
  - Metrics: BLEU, COMET, chrF

---

## 📏 Evaluation Methods

### ✅ Automatic Metrics
- **Accuracy / F1 / EM**: For classification, QA, and math.
- **BLEU / ROUGE / METEOR / chrF**: For text generation, summarization, and translation.
- **Perplexity**: For evaluating fluency (often on held-out corpora like WikiText).
- **Pass@k**: For code generation (e.g., HumanEval).

### 🧠 Human Evaluation
- **Use cases**: Open-ended generation, dialogue quality, hallucination detection.
- **Dimensions evaluated**: Helpfulness, factuality, coherence, safety, and harmlessness.

### 📊 Leaderboards & Evaluation Platforms
- **OpenLLM Leaderboard (by Hugging Face & EleutherAI)**
  - Uses automated evals on a subset of benchmarks (e.g., ARC, HellaSwag, TruthfulQA, etc.).
- **HELM (Holistic Evaluation of Language Models)**
  - Framework for broad, standardized LLM evaluation across capabilities and risks.
- **MT-Bench**
  - Multi-turn benchmark with GPT-4 as judge, used to test chat capabilities.
- **LMEvalHarness (EleutherAI)**
  - Python-based framework for running standardized benchmark suites.

---

## 🔧 How BLEU Score Works

### 🧱 1. N-gram Precision
BLEU calculates the overlap of n-grams (sequences of n words) between the candidate translation and reference translation(s).
- For example, with the sentence:
  - Candidate: "the cat is on the mat"
  - Reference: "there is a cat on the mat"
- BLEU compares unigrams (1-grams), bigrams (2-grams), etc.
- Precision is the number of matched n-grams divided by the number of n-grams in the candidate.

### 🧺 2. Clipping
To avoid rewarding repeated words, BLEU uses clipped counts:
- Each n-gram in the candidate is only counted up to the maximum number of times it appears in any reference.

Example:
- Candidate: "the the the the"
- Reference: "the"

→ BLEU only allows one "the" to match, avoiding a perfect score from spamming.

### 📉 3. Brevity Penalty (BP)
To avoid favoring short translations, BLEU penalizes candidates that are too short compared to references.
- If the candidate is shorter than the reference, a penalty < 1 is applied.
- Formula:
BP = \begin{cases}
1 & \text{if } c > r \\
e^{1 - r/c} & \text{if } c \leq r
\end{cases}
where c = length of candidate, r = effective reference length.

### 📈 4. Final BLEU Score
\text{BLEU} = BP \cdot \exp \left( \sum_{n=1}^N w_n \cdot \log p_n \right)
- p_n: clipped precision for n-grams
- w_n: weight for each n-gram (commonly uniform, e.g., 0.25 for 1-4 grams)
- BLEU typically uses up to 4-grams (BLEU-4)

---

### ✅ Example
Candidate: "the cat is on mat"
Reference: "the cat is on the mat"
- 1-gram precision: 5 matches / 5 candidate words = 1.0
- 2-gram precision: 3 matches / 4 candidate bigrams = 0.75
- Brevity Penalty < 1 due to shorter candidate
- Final BLEU will be less than 1.0 due to missing "the" before "mat"

---

### 📌 Key Points
- BLEU ≠ human judgment, but correlates reasonably for translation tasks.
- Favors exact matches; struggles with paraphrasing or semantics.
- Works best with multiple references.

---

## 🔍 How ROUGE Works

ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is a set of metrics commonly used to evaluate automatic summarization, text generation, and translation, especially when multiple correct outputs are possible.

Unlike BLEU, which focuses on precision of n-grams, ROUGE is recall-oriented, emphasizing how much of the reference text is captured by the generated output.

---

### 📏 1. ROUGE-N (e.g., ROUGE-1, ROUGE-2)
- Measures n-gram overlap between candidate and reference text.
- ROUGE-1 = unigram overlap (individual words)
- ROUGE-2 = bigram overlap (pairs of consecutive words)

Formulas:
- Precision = Overlap n-grams / n-grams in candidate
- Recall = Overlap n-grams / n-grams in reference
- F1-score = Harmonic mean of precision and recall

#### 🧠 Example
Candidate: "the cat sat on the mat"
Reference: "the cat is on the mat"
- ROUGE-1 recall = 5/6 (5 words in common, 6 in reference)

---

### 📏 2. ROUGE-L (Longest Common Subsequence)
- Measures the length of the Longest Common Subsequence (LCS) between candidate and reference.
- LCS captures sentence-level structure better than n-grams.

#### 🧠 Example
Candidate: "the cat sat on the mat"
Reference: "on the mat sat the cat"

→ N-gram overlap is low, but LCS is high: "the cat sat on the mat"
- ROUGE-L Recall = LCS length / reference length
- ROUGE-L Precision = LCS length / candidate length
- ROUGE-L F1 = Harmonic mean

---

### 📏 3. ROUGE-S / ROUGE-SU (Skip-bigram)
- Measures overlap of skip-bigrams (pairs of words in order but not necessarily adjacent).
- ROUGE-SU includes unigrams too.

Used less frequently, but helpful for capturing partial word order matches.

---

### ✅ Key Characteristics

| Feature          | BLEU                | ROUGE                        |
|------------------|---------------------|------------------------------|
| Focus            | Precision           | Recall (mainly)              |
| Use Cases        | Translation, code gen | Summarization, generation    |
| Metrics          | n-gram precision, brevity pen. | n-gram recall, LCS, skip-bigram |
| Multiple Refs    | Supported           | Supported                    |
| Sensitivity      | Harsh on synonyms/paraphrase | Somewhat better with ROUGE-L |

---

## 🛠️ Example in Python (using rouge-score)

