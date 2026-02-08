import json
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt

def load_metrics(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def paper_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "-",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", type=str, default="outputs/metrics/metrics.jsonl")
    ap.add_argument("--out", type=str, default="outputs/plots/phase_progress.png")
    args = ap.parse_args()

    metrics_path = Path(args.metrics)
    rows = load_metrics(metrics_path)
    if not rows:
        raise SystemExit(f"No rows found in {metrics_path}")

    paper_style()

    # Group by tag
    by_tag = defaultdict(list)
    for r in rows:
        by_tag[r["tag"]].append(r)

    # sort each tag series by step
    for tag in by_tag:
        by_tag[tag] = sorted(by_tag[tag], key=lambda x: x.get("step", 0))

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.flatten()

    panels = [
        ("answerability_at_1", "Answerability@1"),
        ("pass_at_1", "Pass@1"),
        ("pass_at_k", "Pass@k"),
        ("proto_leak_rate", "Proto leak rate"),
    ]

    for ax, (key, title) in zip(axes, panels):
        for tag, series in by_tag.items():
            x = [s.get("step", 0) for s in series]
            y = [s[key] for s in series]
            ax.plot(x, y, linewidth=2, label=tag)
        ax.set_title(title)
        ax.set_xlabel("training step")
        ax.set_ylabel(key)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path)
    print("Saved:", out_path)

if __name__ == "__main__":
    main()
