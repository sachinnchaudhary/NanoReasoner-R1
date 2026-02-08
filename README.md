# NanoReasoner-R1

A minimal, verifiable experiment showing how **GRPO (verifier-based RL)** can **compress test-time compute (best-of-k sampling)** into **higher pass@1** on arithmetic problems.

The core idea:
- At inference time, *sampling helps* (best-of-k finds correct answers more often).
- With GRPO, we train the policy so that *the correct answer becomes more likely at pass@1* (policy compression).

## Dataset

DeepMind Mathematics (cleaned JSONL, verifier-friendly):

- Hugging Face dataset: `sachin52/deepmind_math`

Here is the results with baseline and GRPO: 

=== SUMMARY ===
Baseline pass@1: 0.067, pass@8: 0.232, gap: 0.165
GRPO     pass@1: 0.127, pass@8: 0.224, gap: 0.097
Pass@1 improvement: 1.90×    

Each example is a JSON object:
```json
{"problem": "...", "answer": "..."}

