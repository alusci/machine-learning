## RAGAS

RAGAS is an evaluation framework designed to assess the performance of **Retrieval-Augmented Generation (RAG) systems**. It helps measure how well your system retrieves relevant documents and generates accurate responses. Here’s how it works:

### **Key Metrics in RAGAS**
RAGAS evaluates RAG pipelines using three core metrics:
1. **Faithfulness** – Ensures the generated response aligns with the retrieved context.
2. **Relevance** – Measures how well the retrieved documents match the query.
3. **Answer Correctness** – Assesses the accuracy of the final response.

### **How RAGAS Works**
1. **Collect Evaluation Data**  
   - Queries (e.g., "Where is the function `parse_config` defined?")
   - Retrieved Contexts (Documents fetched by your RAG system)
   - Generated Answers (System-generated responses)
   - Expected Answers (Ground truth responses)

2. **Run Evaluation**  
   RAGAS compares the retrieved context and generated answers against expected answers using its built-in metrics.

3. **Analyze Results**  
   - Low **faithfulness** scores indicate hallucinations.
   - Low **relevance** scores suggest poor document retrieval.
   - Low **answer correctness** scores mean the final response is inaccurate.

### **Example Usage**
Here’s a simple Python script to evaluate your RAG system:
```python
from ragas import evaluate
from ragas.metrics import faithfulness, relevance, answer_correctness

queries = ["Where is the function `parse_config` defined?"]
retrieved_contexts = [["def parse_config(): ..."]]  # Retrieved docs
generated_answers = ["The function `parse_config` is defined in config.py"]
expected_answers = ["The function `parse_config` is defined in config.py"]

results = evaluate(
    queries=queries,
    retrieved_contexts=retrieved_contexts,
    generated_answers=generated_answers,
    expected_answers=expected_answers,
    metrics=[faithfulness, relevance, answer_correctness]
)

print(results)
```

### **Why Use RAGAS?**
- **Improves retrieval quality** by identifying weak document matches.
- **Reduces hallucinations** in generated responses.
- **Optimizes RAG workflows** for better accuracy.

Great question! RAGAS calculates its metrics using a combination of **LLM-based evaluations** and **traditional scoring methods**. Here’s a breakdown of how the key metrics are computed:

### **1. Faithfulness**
- **Goal:** Ensures the generated answer aligns with the retrieved context.
- **Calculation:**  
  1. Extracts statements from the generated answer using an LLM.
  2. Checks whether each statement is supported by the retrieved context (Yes/No).
  3. Computes the ratio of context-supported statements to total statements.

### **2. Relevance**
- **Goal:** Measures how well the retrieved documents match the query.
- **Calculation:**  
  - Uses an LLM to assess semantic similarity between the query and retrieved documents.
  - Scores relevance based on how well the retrieved content answers the query.

### **3. Answer Correctness**
- **Goal:** Evaluates the accuracy of the final response.
- **Calculation:**  
  - Compares the generated answer with the expected answer using LLM-based scoring.
  - Uses semantic similarity and factual correctness checks.

### **Additional Metrics**
RAGAS also supports:
- **Context Precision** – Measures how much of the retrieved context is actually useful.
- **Context Recall** – Evaluates whether enough relevant information was retrieved.
- **Noise Sensitivity** – Detects irrelevant or misleading retrieved content.



