import json
from pathlib import Path
import matplotlib.pyplot as plt

def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and line.startswith("{"):
                rows.append(json.loads(line))
    return rows

def paper_style():
    plt.rcParams.update({
        "figure.dpi": 200,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linestyle": "-",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 2.2,
    })

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--out", default="outputs/plots/ship_eval_paper.png")
    args = ap.parse_args()

    paper_style()

    rows = read_jsonl(Path(args.metrics))
    by_tag = {r["tag"]: r for r in rows}

    base = by_tag["baseline_qwen15"]
    grpo = by_tag["grpo_latest"]

    labels = ["Baseline", "GRPO"]
    pass1 = [base["pass@1"], grpo["pass@1"]]
    pass8 = [base["pass@k_bestof"], grpo["pass@k_bestof"]]
    gap = [pass8[0] - pass1[0], pass8[1] - pass1[1]]

    # Summary numbers
    p1_gain = pass1[1] - pass1[0]
    p1_ratio = pass1[1] / max(pass1[0], 1e-12)
    gap_reduction = (gap[0] - gap[1]) / max(gap[0], 1e-12)

    fig = plt.figure(figsize=(10.5, 8.5))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 1.0, 0.9], hspace=0.35)

    # --- Panel 1: pass@1 vs pass@8 ---
    ax1 = fig.add_subplot(gs[0, 0])
    x = [0, 1]
    w = 0.36
    ax1.bar([i - w/2 for i in x], pass1, width=w, label="pass@1")
    ax1.bar([i + w/2 for i in x], pass8, width=w, label="best-of-8 pass@8")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Accuracy")
    ax1.set_title("DeepMind Math (n=1000) — Test-time compute vs policy compression")
    ax1.legend(frameon=False, loc="upper left")

    # annotate improvements
    ax1.text(
        0.02, 0.05,
        f"pass@1 gain: +{p1_gain:.3f}  ({p1_ratio:.2f}×)",
        transform=ax1.transAxes
    )

    # --- Panel 2: headroom gap ---
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.bar(labels, gap)
    ax2.set_ylabel("pass@8 − pass@1")
    ax2.set_title("Search headroom shrinks after GRPO (compression)")

    ax2.text(
        0.02, 0.78,
        f"headroom reduction: {gap_reduction*100:.1f}%",
        transform=ax2.transAxes
    )

    # --- Panel 3: compact table-like annotation ---
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.axis("off")

    txt = (
        f"Baseline  : pass@1={pass1[0]:.3f} | pass@8={pass8[0]:.3f} | gap={gap[0]:.3f}\n"
        f"GRPO      : pass@1={pass1[1]:.3f} | pass@8={pass8[1]:.3f} | gap={gap[1]:.3f}\n"
        f"Interpretation: GRPO moved probability mass from 'rare correct samples' into the default output (pass@1)."
    )
    ax3.text(0.01, 0.85, txt, va="top")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    print("Saved:", out)

if __name__ == "__main__":
    main()
