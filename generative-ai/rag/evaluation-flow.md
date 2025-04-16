### ✅ **1. Generate questions from context**
This is a strong starting point and ensures questions are answerable given the context.

**Tips**:
- Use prompts like: *"Generate a question someone might ask based on the following passage."*
- You can bias toward either *factual*, *reasoning*, or *extrapolative* questions depending on your RAG use case.

**Code Sample**:
```Python
question_gen_template = PromptTemplate(
    input_variables=["context"],
    template="""
Given the following context, generate a question that could be asked based only on this information.

Context:
{context}

Question:
"""
)

question_gen_chain = LLMChain(llm=llm, prompt=question_gen_template)
```

---

### ✅ **2. Assign groundedness score to questions**
You're evaluating whether the question is *anchored* in the context or speculative.

**Tip**:
- Define a rubric (e.g.):
  - 5 = Directly answerable from context
  - 3 = Vaguely inspired by context
  - 1 = Unrelated or speculative
- You can also automate this with a consistent LLM prompt.

**Code Sample**

```Python
question_groundedness_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Context:
{context}

Question:
{question}

Rate the groundedness of the question from 1 (completely ungrounded) to 5 (fully grounded in the context). 
Explain your reasoning in 1-2 sentences.

Score:
"""
)

question_score_chain = LLMChain(llm=llm, prompt=question_groundedness_template)
```

---

### ✅ **3. Generate answers based on question + context**
You're essentially creating **gold reference answers** that are known to be grounded.

**Code Sample**
```Python
answer_gen_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Context:
{context}

Question:
{question}

Based only on the above context, provide a concise and informative answer.

Answer:
"""
)

answer_gen_chain = LLMChain(llm=llm, prompt=answer_gen_template)

```

---

### ✅ **4. Assign groundedness score to answers**
Evaluating whether the **generated answer** is well-supported by the given context.

Same rubric applies. Consider flagging:
- **Incomplete** but grounded answers (e.g. 3-4)
- **Hallucinations** or unsupported claims (1-2)

**Code Sample**
```Python
answer_groundedness_template = PromptTemplate(
    input_variables=["context", "question", "answer"],
    template="""
Context:
{context}

Question:
{question}

Answer:
{answer}

Rate the groundedness of the answer from 1 (completely ungrounded) to 5 (fully grounded in the context). 
Explain your reasoning in 1-2 sentences.

Score:
"""
)

answer_score_chain = LLMChain(llm=llm, prompt=answer_groundedness_template)

```

---

### ✅ **5. Generate answers using RAG system**
This is your test-time system. You're doing **inference + retrieval**, which is exactly what you want to evaluate.

---

### ✅ **6. Evaluate RAG answer’s groundedness with template**

This prompt format is direct and leads to structured, explainable judgments:

```text
Context: <insert>
Question: <insert>
Answer: <RAG answer>

Rate the groundedness of the answer from 1 (completely ungrounded) to 5 (fully grounded in the context). Explain why.
```

**Pro Tip**:
To increase consistency and reduce variance:
- Add your rubric explicitly in the prompt.
- Ask for a score **and a short explanation** (as you’ve done).
- Optionally ask for a binary `Is this hallucinated? Yes/No`.

---

### 🧠 Optional Enhancements
- **Include the RAG-retrieved context vs. original context**: Compare how retrieval quality affects grounding.
- **Compute delta** between gold answer score and RAG answer score.
- **Human-in-the-loop QA** for a sample of items, to calibrate LLM ratings.
