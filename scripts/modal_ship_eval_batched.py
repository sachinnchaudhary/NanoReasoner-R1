import json, re, random, time
from pathlib import Path
import modal

app = modal.App("nanoreasoner-ship-eval-batched")

GPU = "A100"
PY = "3.11"

DATA_REPO = "sachin52/deepmind_math"
EVAL_FILE = "eval.jsonl"

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("nanoreasoner-checkpoints", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version=PY)
    .pip_install("torch", "transformers", "accelerate", "huggingface_hub")
)

def norm_num(s: str) -> str:
    s = str(s).strip().replace(",", "").replace(" ", "")
    if s in {"-0", "-0.0", "-0.00"}:
        s = "0"
    return s

def make_prompt(problem: str) -> str:
    return (
        "Solve the problem. Return only the final number formatted exactly as:\n"
        "<answer>NUMBER</answer>\n\n"
        f"Problem: {problem}\n"
    )

def extract_answer_last(text: str):
    # Take LAST <answer>...</answer> (prompt itself contains one)
    matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if not matches:
        return None
    return norm_num(matches[-1])

def cut_after_last_answer(text: str) -> str:
    low = text.lower()
    j = low.rfind("</answer>")
    if j == -1:
        return text
    return text[: j + len("</answer>")]

@app.function(
    gpu=GPU,
    image=image,
    timeout=60 * 1200,
    volumes={"/root/.cache/huggingface": hf_cache, "/root/checkpoints": ckpt_vol},
)
def ship_eval_batched(
    # model selection
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",   # HF model id
    ckpt_dir: str = "",                              # if set, load from this local dir instead

    # eval params
    n_eval: int = 1000,
    k: int = 8,
    batch_size: int = 16,
    max_new_tokens: int = 32,
    temperature: float = 0.3,
    top_p: float = 0.9,
    seed: int = 0,

    # logging
    tag: str = "baseline_qwen15",
    out_metrics_path: str = "/root/checkpoints/ship_eval/metrics.jsonl",
):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, AutoModelForCausalLM

    t0 = time.time()

    # --- data download ---
    data_dir = Path("/root/deepmind_data")
    data_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=DATA_REPO, repo_type="dataset", local_dir=str(data_dir), allow_patterns=[EVAL_FILE])

    rows = []
    with (data_dir / EVAL_FILE).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    rng = random.Random(seed)
    rng.shuffle(rows)
    rows = rows[: min(n_eval, len(rows))]

    prompts = [make_prompt(ex["problem"]) for ex in rows]
    golds = [norm_num(ex["answer"]) for ex in rows]

    # --- model load ---
    if ckpt_dir.strip():
        load_source = ckpt_dir
    else:
        load_source = model_id

    tok = AutoTokenizer.from_pretrained(load_source, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        load_source,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    @torch.no_grad()
    def greedy_batch(batch_prompts):
        inputs = tok(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,
        )
        texts = tok.batch_decode(gen, skip_special_tokens=False)
        return [cut_after_last_answer(t) for t in texts]

    @torch.no_grad()
    def sample_kminus1_batch(batch_prompts, k_minus_1: int):
        if k_minus_1 <= 0:
            return []
        inputs = tok(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=k_minus_1,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,
        )
        texts = tok.batch_decode(gen, skip_special_tokens=False)
        return [cut_after_last_answer(t) for t in texts]

    total = 0
    fmt_ok = 0
    pass1 = 0
    passk = 0

    k_minus_1 = max(0, k - 1)

    for i in range(0, len(prompts), batch_size):
        b_prompts = prompts[i:i+batch_size]
        b_golds = golds[i:i+batch_size]
        B = len(b_prompts)

        # greedy pass@1
        greedy_texts = greedy_batch(b_prompts)
        greedy_ans = [extract_answer_last(t) for t in greedy_texts]

        ok_any = [False] * B
        for j in range(B):
            total += 1
            if greedy_ans[j] is not None:
                fmt_ok += 1
                if greedy_ans[j] == b_golds[j]:
                    pass1 += 1
                    ok_any[j] = True

        # sampled k-1 in one call
        sampled_texts = sample_kminus1_batch(b_prompts, k_minus_1)
        # sampled_texts length = B*(k-1)
        for j in range(B):
            if ok_any[j]:
                continue
            chunk = sampled_texts[j*k_minus_1:(j+1)*k_minus_1] if k_minus_1 > 0 else []
            for t in chunk:
                ak = extract_answer_last(t)
                if ak is not None and ak == b_golds[j]:
                    ok_any[j] = True
                    break

        passk += sum(ok_any)

        if (i // batch_size) % 10 == 0:
            print(f"progress: {min(i+batch_size, len(prompts))}/{len(prompts)}", flush=True)

    elapsed = time.time() - t0

    result = {
        "tag": tag,
        "load_source": load_source,
        "model_id": model_id,
        "ckpt_dir": ckpt_dir,
        "n_eval_used": total,
        "k": k,
        "batch_size": batch_size,
        "fmt_rate": fmt_ok / total if total else 0.0,
        "pass@1": pass1 / total if total else 0.0,
        "pass@k_bestof": passk / total if total else 0.0,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "elapsed_sec": elapsed,
    }

    # append to metrics file in Modal volume
    out_path = Path(out_metrics_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
    ckpt_vol.commit()

    print(json.dumps(result, indent=2))
    print(f"WROTE: {out_metrics_path}", flush=True)
    return result

@app.local_entrypoint()
def main(
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
    ckpt_dir: str = "",
    n_eval: int = 1000,
    k: int = 8,
    batch_size: int = 16,
    max_new_tokens: int = 32,
    temperature: float = 0.3,
    top_p: float = 0.9,
    seed: int = 0,
    tag: str = "baseline_qwen15",
    out_metrics_path: str = "/root/checkpoints/ship_eval/metrics.jsonl",
):
    ship_eval_batched.remote(
        model_id=model_id,
        ckpt_dir=ckpt_dir,
        n_eval=n_eval,
        k=k,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        tag=tag,
        out_metrics_path=out_metrics_path,
    )
