# NanoReasoner-R1

For research blog() i want to show how Test time compute worksA minimal, verifiable experiment showing how **GRPO (verifier-based RL)** can **compress test-time compute (best-of-k sampling)** into **higher pass@1** on arithmetic problems.

The core idea:
- At inference time, *sampling helps* (best-of-k finds correct answers more often).
- With GRPO, we train the policy so that *the correct answer becomes more likely at pass@1* (policy compression).

## Dataset

I used DeepMind Mathematics (cleaned JSONL, verifier-friendly) dataset which contain 1,90,000 train and 10,000 eval problems which i uploaded at:

- Hugging Face dataset: `sachin52/deepmind_math`

## Results

Here is the results with baseline and GRPO: 

| Method   | pass@1 | pass@8 | gap   |
|----------|--------|--------|-------|
| Baseline | 0.067  | 0.232  | 0.165 |
| GRPO     | 0.127  | 0.224  | 0.097 |

**Pass@1 improvement: 1.90×**
 

Each example is a JSON object:
```json
{"problem": "...", "answer": "..."}

