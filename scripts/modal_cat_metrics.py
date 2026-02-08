import modal

app = modal.App("nanoreasoner-cat-metrics")
ckpt_vol = modal.Volume.from_name("nanoreasoner-checkpoints", create_if_missing=True)

@app.function(volumes={"/root/checkpoints": ckpt_vol})
def cat(path: str):
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        raise RuntimeError(f"File not found: {path}")
    print(p.read_text(encoding="utf-8"), end="")

@app.local_entrypoint()
def main(path: str):
    cat.remote(path)