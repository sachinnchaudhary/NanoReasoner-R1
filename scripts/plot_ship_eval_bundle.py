import json
from pathlib import Path
import matplotlib.pyplot as plt

def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def paper_style():
    plt.rcParams.update({
        "figure.dpi": 170,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 2.4,
    })

def get_by_tag(rows):
    by = {}
    for r in rows:
        by[r["tag"]] = r
    return by

def save_fig(fig, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    print("Saved:", out_path)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--out_dir", default="outputs/plots")
    args = ap.parse_args()

    paper_style()
    rows = read_jsonl(Path(args.metrics))
    by = get_by_tag(rows)

    # Expect these tags
    base = by["baseline_qwen15"]
    grpo = by["grpo_latest"]

    labels = ["Baseline", "GRPO"]
    pass1 = [base["pass@1"], grpo["pass@1"]]
    pass8 = [base["pass@k_bestof"], grpo["pass@k_bestof"]]
    gap = [pass8[0] - pass1[0], pass8[1] - pass1[1]]
    ratio = pass1[1] / max(pass1[0], 1e-9)

    out_dir = Path(args.out_dir)

    # Plot A: pass@1 and pass@8 bars
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = [0, 1]
    ax.bar([i-0.18 for i in x], pass1, width=0.36, label="pass@1")
    ax.bar([i+0.18 for i in x], pass8, width=0.36, label="best-of-8 pass@8")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("accuracy")
    ax.set_title("DeepMind Math (n=1000) — Baseline vs GRPO (Qwen2.5-1.5B-Instruct)")
    ax.legend(frameon=False, loc="upper left")
    save_fig(fig, out_dir / "ship_eval_pass1_pass8.png")

    # Plot B: gap compression
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(labels, gap)
    ax.set_ylabel("pass@8 − pass@1")
    ax.set_title("Test-time compute headroom shrinks after GRPO (compression)")
    save_fig(fig, out_dir / "ship_eval_gap.png")

    # Plot C: pass@1 ratio
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.bar(["pass@1 improvement"], [ratio])
    ax.set_ylabel("GRPO pass@1 / Baseline pass@1")
    ax.set_title(f"Pass@1 improved by ~{ratio:.2f}×")
    save_fig(fig, out_dir / "ship_eval_pass1_ratio.png")

    # Print a compact summary (for your blog)
    print("\n=== SUMMARY ===")
    print(f"Baseline pass@1: {pass1[0]:.3f}, pass@8: {pass8[0]:.3f}, gap: {gap[0]:.3f}")
    print(f"GRPO     pass@1: {pass1[1]:.3f}, pass@8: {pass8[1]:.3f}, gap: {gap[1]:.3f}")
    print(f"Pass@1 improvement: {ratio:.2f}×")

if __name__ == "__main__":
    main()
