import json, random, re, time
from pathlib import Path
import modal

app = modal.App("nanoreasoner-grpo-qwen15-deepmind-resume")

GPU = "A100"
PY = "3.11"

DATA_REPO = "sachin52/deepmind_math"
TRAIN_FILE = "train.jsonl"
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

def extract_answer(text: str):
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
    timeout=60 * 960,  # ✅ 16 hours
    volumes={"/root/.cache/huggingface": hf_cache, "/root/checkpoints": ckpt_vol},
)
def grpo_resume(
    # resume controls
    resume_dir: str = "/root/checkpoints/grpo_qwen15_deepmind_1k/ckpt_latest",
    start_step: int = 780,         # where you left off (set based on logs)
    more_steps: int = 300,         # how many more steps to run

    # training knobs (keep same as before unless you want change)
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
    k: int = 4,
    grad_accum: int = 8,
    lr: float = 5e-6,
    temperature: float = 0.3,
    top_p: float = 0.9,
    max_new_tokens: int = 32,
    len_penalty: float = 0.001,

    # eval / saving
    eval_every: int = 100,
    eval_n: int = 200,
    eval_k: int = 8,
    seed: int = 0,

    # output
    run_name: str = "grpo_qwen15_deepmind_1k",
):
    import torch
    import torch.nn.functional as F
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, AutoModelForCausalLM

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # data
    data_dir = Path("/root/deepmind_data")
    data_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=DATA_REPO, repo_type="dataset", local_dir=str(data_dir),
                      allow_patterns=[TRAIN_FILE, EVAL_FILE])

    def load_jsonl(p: Path):
        rows = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    train_rows = load_jsonl(data_dir / TRAIN_FILE)
    eval_rows = load_jsonl(data_dir / EVAL_FILE)
    rng = random.Random(seed)
    rng.shuffle(train_rows)

    print("train rows:", len(train_rows), "eval rows:", len(eval_rows), flush=True)

    # load tokenizer
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)

    # resume model weights
    resume_path = Path(resume_dir)
    if not resume_path.exists():
        raise RuntimeError(f"resume_dir not found: {resume_dir}")

    print("Resuming from:", str(resume_path), flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        str(resume_path),
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    # helpers
    @torch.no_grad()
    def sample_one(prompt: str):
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=1,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,
        )
        full_text = tok.decode(gen[0], skip_special_tokens=False)
        full_text = cut_after_last_answer(full_text)
        prompt_ids = inputs["input_ids"][0].tolist()
        out_ids = gen[0].tolist()[len(prompt_ids):]
        return out_ids, full_text, prompt_ids

    def logprob_with_grad(prompt_ids, sampled_ids):
        if len(sampled_ids) == 0:
            return torch.tensor(0.0, device=model.device)
        inp = prompt_ids + sampled_ids[:-1]
        tgt = sampled_ids
        x = torch.tensor([inp], device=model.device, dtype=torch.long)
        y = torch.tensor([tgt], device=model.device, dtype=torch.long)
        out = model(input_ids=x)
        logp = F.log_softmax(out.logits, dim=-1)
        gathered = torch.gather(logp, dim=-1, index=y.unsqueeze(-1)).squeeze(-1)
        return gathered.sum()

    def reward(gold: str, text: str, out_ids):
        ans = extract_answer(text)
        correct = 1.0 if (ans is not None and ans == gold) else 0.0
        return float(correct - len_penalty * len(out_ids))

    @torch.no_grad()
    def eval_bestof(n: int, k_best: int):
        model.eval()
        rows = eval_rows[: min(n, len(eval_rows))]
        total = 0
        fmt_ok = 0
        pass1 = 0
        passk = 0

        for ex in rows:
            gold = norm_num(ex["answer"])
            prompt = make_prompt(ex["problem"])
            inputs = tok(prompt, return_tensors="pt").to(model.device)

            # greedy
            gen = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.eos_token_id,
            )
            text1 = cut_after_last_answer(tok.decode(gen[0], skip_special_tokens=False))
            a1 = extract_answer(text1)
            if a1 is not None:
                fmt_ok += 1
                if a1 == gold:
                    pass1 += 1
            ok_any = (a1 is not None and a1 == gold)

            # sample k-1
            for _ in range(max(0, k_best - 1)):
                _, t, _ = sample_one(prompt)
                ak = extract_answer(t)
                if ak is not None and ak == gold:
                    ok_any = True
                    break

            if ok_any:
                passk += 1
            total += 1

        model.train()
        return {
            "n": total,
            "fmt_rate": fmt_ok / total if total else 0.0,
            "pass@1": pass1 / total if total else 0.0,
            f"pass@{k_best}": passk / total if total else 0.0,
        }

    out_dir = Path("/root/checkpoints") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"

    def log_line(step_i: int, payload: dict):
        row = {"stage": "GRPO", "step": step_i, **payload}
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        ckpt_vol.commit()
        print(f"  METRICS_WROTE: {metrics_path}", flush=True)

    t0 = time.time()
    ptr = 0

    end_step = start_step + more_steps
    print(f"\n=== RESUME GRPO === start_step={start_step} more_steps={more_steps} -> end_step={end_step}", flush=True)

    for step in range(start_step + 1, end_step + 1):
        opt.zero_grad(set_to_none=True)

        total_loss = 0.0
        avgR = 0.0
        total_rolls = 0
        eq_groups = 0
        hit_groups = 0
        avg_len = 0.0

        for _ in range(grad_accum):
            ex = train_rows[ptr % len(train_rows)]
            ptr += 1
            gold = norm_num(ex["answer"])
            prompt = make_prompt(ex["problem"])

            sampled_ids_list = []
            rewards = []
            prompt_ids_saved = None

            for _s in range(k):
                out_ids, text, prompt_ids = sample_one(prompt)
                prompt_ids_saved = prompt_ids
                r = reward(gold, text, out_ids)
                sampled_ids_list.append(out_ids)
                rewards.append(r)
                avgR += r
                total_rolls += 1

            rmax, rmin = max(rewards), min(rewards)
            if abs(rmax - rmin) < 1e-9:
                eq_groups += 1
            if rmax > 0.5:
                hit_groups += 1
            avg_len += sum(len(x) for x in sampled_ids_list) / k

            r_mean = sum(rewards) / k
            adv = [r - r_mean for r in rewards]
            adv_t = torch.tensor(adv, device=model.device, dtype=torch.float32)

            for j, new_ids in enumerate(sampled_ids_list):
                logp = logprob_with_grad(prompt_ids_saved, new_ids)
                loss = -(adv_t[j] * logp) / grad_accum
                loss.backward()
                total_loss += float(loss.detach().cpu().item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if total_rolls:
            avgR /= total_rolls
            avg_len /= grad_accum

        if step % 20 == 0:
            elapsed_min = (time.time() - t0) / 60.0
            print(
                f"[rl {step:04d}] loss~{total_loss:.6f} avgR={avgR:+.3f} "
                f"eq_groups={eq_groups}/{grad_accum} hit_groups={hit_groups}/{grad_accum} avg_len={avg_len:.1f} "
                f"elapsed_min={elapsed_min:.1f}",
                flush=True,
            )

        if eval_every and (step % eval_every == 0 or step == end_step):
            m = eval_bestof(eval_n, eval_k)
            m["elapsed_min"] = (time.time() - t0) / 60.0
            print(f"  EVAL@{step}: fmt={m['fmt_rate']:.3f} pass@1={m['pass@1']:.3f} pass@{eval_k}={m[f'pass@{eval_k}']:.3f} n={m['n']}", flush=True)
            log_line(step, m)

            # save latest
            latest_path = out_dir / "ckpt_latest"
            model.save_pretrained(latest_path)
            tok.save_pretrained(latest_path)
            ckpt_vol.commit()
            print(f"  CKPT_SAVED: {latest_path}", flush=True)

    print("Done resume run. Metrics:", metrics_path, flush=True)
    return {"metrics": str(metrics_path), "last_step": end_step}

@app.local_entrypoint()
def main(
    resume_dir: str = "/root/checkpoints/grpo_qwen15_deepmind_1k/ckpt_latest",
    start_step: int = 780,
    more_steps: int = 300,
    run_name: str = "grpo_qwen15_deepmind_1k",
):
    grpo_resume.remote(
        resume_dir=resume_dir,
        start_step=start_step,
        more_steps=more_steps,
        run_name=run_name,
    )
