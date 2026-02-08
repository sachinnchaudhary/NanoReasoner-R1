import json, random, re
from pathlib import Path
import modal

app = modal.App("nanoreasoner-qwen-eval-deepmind")

GPU = "A100"
PY = "3.11"

DATA_REPO = "sachin52/deepmind_math"
EVAL_FILE = "eval.jsonl"

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version=PY)
    .pip_install(
        "torch",
        "transformers",
        "huggingface_hub",
        "accelerate",
    )
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

def extract_answer(text: str):
    matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if not matches:
        return None
    return norm_num(matches[-1])

def cut_after_answer(text: str) -> str:
    low = text.lower()
    j = low.rfind("</answer>") 
    if j == -1:
        return text
    return text[: j + len("</answer>")]

@app.function(
    gpu=GPU,
    image=image,
    timeout=60 * 600,
    volumes={"/root/.cache/huggingface": hf_cache},
)
def eval_qwen(
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    n_eval: int = 1000,
    k: int = 8,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    top_p: float = 0.9,
    seed: int = 0,
):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # Download eval.jsonl from your HF dataset repo
    data_dir = Path("/root/deepmind_data")
    data_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=DATA_REPO,
        repo_type="dataset",
        local_dir=str(data_dir),
        allow_patterns=[EVAL_FILE],
    )
    eval_path = data_dir / EVAL_FILE

    rows = []
    with eval_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    rng = random.Random(seed)
    rng.shuffle(rows)
    rows = rows[: min(n_eval, len(rows))]

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)

    # Force eager attention to avoid SDPA/sliding-window surprises
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    @torch.no_grad()
    def gen_text(prompt: str, do_sample: bool, temp: float, p: float):
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=(temp if do_sample else None),
            top_p=(p if do_sample else None),
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,
        )
        text = tok.decode(gen[0], skip_special_tokens=False)
        return cut_after_answer(text)

    total = 0
    fmt_ok = 0
    pass1 = 0
    passk = 0

    for ex in rows:
        gold = norm_num(ex["answer"])
        prompt = make_prompt(ex["problem"])

        # pass@1 (greedy)
        t1 = gen_text(prompt, do_sample=False, temp=0.0, p=1.0)
        a1 = extract_answer(t1)

        if a1 is not None:
            fmt_ok += 1
            if a1 == gold:
                pass1 += 1

        ok_any = (a1 is not None and a1 == gold)

        # best-of-k INCLUDING greedy
        for _ in range(max(0, k - 1)):
            tk = gen_text(prompt, do_sample=True, temp=temperature, p=top_p)
            ak = extract_answer(tk)
            if ak is not None and ak == gold:
                ok_any = True
                break

        if ok_any:
            passk += 1

        total += 1

    out = {
        "model": model_id,
        "n_eval_used": total,
        "k": k,
        "fmt_rate": (fmt_ok / total) if total else 0.0,
        "pass@1": (pass1 / total) if total else 0.0,
        "pass@k_bestof": (passk / total) if total else 0.0,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
    }
    print(json.dumps(out, indent=2))
    return out

@app.local_entrypoint()
def main(
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    n_eval: int = 1000,
    k: int = 8,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    top_p: float = 0.9,
    seed: int = 0,
):
    eval_qwen.remote(
        model_id=model_id,
        n_eval=n_eval,
        k=k,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    )