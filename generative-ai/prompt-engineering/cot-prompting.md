## Chain of Thought Prompting

In essence, this approach introduces a sequence of intermediate reasoning steps in the zero or few-shot prompting process. We commonly add the phrase Let's think **step-by-step**. 

There is also **Auto-CoT**: this approach leverages LLMs to generate reasoning chains for demonstrations automatically. 

### Example

Q: Maria has 8 markers. She buys 2 more packs. Each pack has 4 markers. How many markers does she have now?

#### Zero shot answer
A: The answer is 16

#### CoT answer
A: Maria starts with 8 markers and she buys 2 more packs.  
A: Each pack has 4 markers  
A: She therefore bought 2 * 4 = 8 markers  
A: The answer is 16  

## Self-Consistency Prompting Technique

It involves asking the model the same prompt multiple times and taking the majority result as the final answer.

### Scenario

An email is received reporting a major security vulnerabilities
By using self-consistency prompting, the model can be prompted the same email multiple times to ensurea consistent classification. 
By taking the majority result, the model can provide more reliable classification. 

* Diverse Reasoning Paths
* Majority Voting System
* Streamlined implementation

* Benefits:
    *  Reduced Bias
    * Improved Accuracy
    * Enhanced Critical Thinking


#### Example

```
prompt = """
Let's consider which is heavier: 1000 feathers or a 30-pound weight.
I'll think through this in a few different ways and then decide which answer seems most consistent.

1. First line of reasoning: A single feather is very light, almost weightless.
So, 1000 feathers might still be quite light, possibly lighter than a 30-pound weight.

2. Second line of reasoning: 1000 is a large number, and when you add up the weight of so many feathers,
it could be quite heavy. Maybe it's heavier than a 30-pound weight.

3. Third line of reasoning: The average weight of a feather is very small. Even 1000 feathers would not add up to 30 pounds.

Considering these reasonings, the most consistent answer is:
"""

```
