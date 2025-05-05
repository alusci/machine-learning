🧠 What Is Beam Search?

Beam search keeps track of the top-k most likely sequences at each generation step instead of just the single best one (like greedy decoding).

Think of it as breadth-first search in sequence space, keeping only the best beam_width candidates.

⸻

🪄 Core Idea:

At each decoding step:
	1.	For each active sequence in the beam, compute logits for the next token
	2.	Expand each by all possible next tokens → many candidate sequences
	3.	Select the top k overall sequences (based on total score)
	4.	Repeat until:
	•	<eos> is generated
	•	Max length is reached

⸻

📊 Example: Beam Width = 2

Say at time t=1 you have:

[<bos>] → ["The", "A"]

At time t=2, suppose you expand each:

"The" → ["The cat", "The dog"]
"A"   → ["A car", "A man"]

Now you rank all 4 and keep the top 2:

["The cat", "A man"]  ← beam size = 2



⸻

✨ Why Use It?

✅ Advantages:
	•	Avoids greedy mistakes
	•	Produces more coherent, global sequences
	•	Especially useful in:
	•	Translation
	•	Summarization
	•	Question answering

⚠️ Disadvantages:
	•	More computationally expensive (beam width × model calls)
	•	Can still lack diversity (→ solutions: diverse beam search, top-k hybrid)

⸻

📈 How Is It Scored?

Typically uses log-probabilities:

score = sum(log(P(token_i)))  # across time steps

Optionally normalized:

score /= (sequence_length ** length_penalty)



⸻

🧪 Code (HuggingFace style)

model.generate(
    input_ids,
    max_length=50,
    num_beams=5,          # beam width
    early_stopping=True,
    no_repeat_ngram_size=2,
    length_penalty=1.0
)



⸻

✅ TL;DR

Beam search explores multiple candidate sequences at each step, balancing likelihood and diversity. It’s better than greedy, less random than sampling, and often used in high-quality generation tasks.
