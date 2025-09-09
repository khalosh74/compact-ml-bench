from __future__ import annotations
import subprocess, sys, os, json, shutil
from pathlib import Path
import typer, yaml

app = typer.Typer(add_completion=False)
ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"

def sh(args: list[str], env: dict[str,str] | None = None):
    # Inherit stdout/stderr so you see progress live
    typer.secho(">> " + " ".join(args), fg=typer.colors.CYAN)
    r = subprocess.run(args, cwd=ROOT, env=env)
    if r.returncode != 0:
        raise typer.Exit(r.returncode)

def load_cfg(config: Path | None) -> dict:
    if not config:
        return {}
    with open(config, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def override(cfg: dict, **kwargs) -> dict:
    out = dict(cfg)
    for k, v in kwargs.items():
        if v is not None:
            out[k] = v
    return out

def ensure_exists(path: Path, what: str):
    if not path.exists():
        typer.secho(f"[ERROR] Missing {what}: {path}", fg=typer.colors.RED)
        raise typer.Exit(2)

@app.command()
def env():
    """Environment/status checks."""
    sh([sys.executable, str(SCRIPTS / "status.py")])
    sh([sys.executable, str(SCRIPTS / "diagnose.py")])

@app.command()
def data(config: Path = typer.Option(None, help="YAML config")):
    cfg = load_cfg(config)
    dataset = cfg.get("dataset", "cifar10")
    data_dir = cfg.get("data_dir", "data")
    sh([sys.executable, str(SCRIPTS / "download_data.py"), "--dataset", dataset, "--out", data_dir])

@app.command()
def train(
    config: Path = typer.Option(None, help="YAML config"),
    model: str | None = None,
    epochs: int | None = None,
    batch: int | None = None,
    workers: int | None = None,
    seed: int | None = None,
    outbase: str | None = None,
):
    cfg = override(load_cfg(config), model=model, epochs=epochs, batch=batch,
                   workers=workers, seed=seed, outbase=outbase)
    model = cfg.get("model", "resnet18")
    epochs = int(cfg.get("epochs", 40))
    batch  = int(cfg.get("batch", 1024))
    workers= int(cfg.get("workers", 16))
    seed   = int(cfg.get("seed", 42))
    outbase= cfg.get("outbase", "baseline_40e")

    out_dir = ROOT / "runs" / outbase
    sh([sys.executable, str(SCRIPTS / "train.py"),
        "--model", model,
        "--epochs", str(epochs),
        "--batch-size", str(batch),
        "--seed", str(seed),
        "--num-workers", str(workers),
        "--out", str(out_dir)])
    ensure_exists(out_dir / "best.pt", "baseline checkpoint")

@app.command("bench")
def bench_baseline(
    config: Path = typer.Option(None, help="YAML config"),
    outbase: str | None = None,
    warmup: int | None = None,
    repeat: int | None = None,
):
    cfg = override(load_cfg(config), outbase=outbase, warmup=warmup, repeat=repeat)
    outbase = cfg.get("outbase", "baseline_40e")
    warmup  = int(cfg.get("warmup", 60))
    repeat  = int(cfg.get("repeat", 600))
    ckpt = ROOT / "runs" / outbase / "best.pt"
    ensure_exists(ckpt, "baseline checkpoint")

    # GPU
    sh([sys.executable, "-u", str(SCRIPTS / "bench.py"),
        "--checkpoint", str(ckpt),
        "--device", "gpu", "--warmup", str(warmup), "--repeat", str(repeat), "--verbose", "1"])

    # CPU (single thread for stability)
    env = dict(os.environ)
    env["MKL_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    out_json = ROOT / "outputs" / f"bench_{outbase}_cpu.json"
    sh([sys.executable, "-u", str(SCRIPTS / "bench.py"),
        "--checkpoint", str(ckpt),
        "--device", "cpu", "--warmup", str(warmup), "--repeat", str(repeat),
        "--threads", "1", "--out", str(out_json)], env=env)

@app.command()
def prune(
    config: Path = typer.Option(None, help="YAML config"),
    outbase: str | None = None,
    pruned_name: str | None = None,
    prune_ratio: float | None = None,
    prune_epochs: int | None = None,
    batch: int | None = None,
    workers: int | None = None,
):
    cfg = override(load_cfg(config), outbase=outbase, pruned_name=pruned_name,
                   prune_ratio=prune_ratio, prune_epochs=prune_epochs,
                   batch=batch, workers=workers)
    outbase     = cfg.get("outbase", "baseline_40e")
    pruned_name = cfg.get("pruned_name", "resnet18_struct30")
    prune_ratio = float(cfg.get("prune_ratio", 0.3))
    prune_epochs= int(cfg.get("prune_epochs", 10))
    batch       = int(cfg.get("batch", 1024))
    workers     = int(cfg.get("workers", 16))

    ckpt = ROOT / "runs" / outbase / "best.pt"
    ensure_exists(ckpt, "baseline checkpoint")

    # Ensure torch-pruning is available (best-effort)
    try:
        import torch_pruning  # type: ignore
    except Exception:
        sh([sys.executable, "-m", "pip", "install", "torch-pruning==1.6.0"])

    out_dir = ROOT / "runs" / pruned_name
    sh([sys.executable, str(SCRIPTS / "structured_prune.py"),
        "--checkpoint", str(ckpt),
        "--ratio", str(prune_ratio),
        "--epochs", str(prune_epochs),
        "--batch-size", str(batch),
        "--num-workers", str(workers),
        "--out", str(out_dir)])
    ensure_exists(out_dir / "structured.ts", "TorchScript (structured)")

@app.command("bench-ts")
def bench_torchscript(
    config: Path = typer.Option(None, help="YAML config"),
    pruned_name: str | None = None,
    warmup: int | None = None,
    repeat: int | None = None,
):
    cfg = override(load_cfg(config), pruned_name=pruned_name, warmup=warmup, repeat=repeat)
    pruned_name = cfg.get("pruned_name", "resnet18_struct30")
    warmup  = int(cfg.get("warmup", 60))
    repeat  = int(cfg.get("repeat", 600))
    ts = ROOT / "runs" / pruned_name / "structured.ts"
    ensure_exists(ts, "TorchScript artifact")

    # GPU
    sh([sys.executable, str(SCRIPTS / "bench_ts.py"),
        "--artifact", str(ts),
        "--device", "gpu", "--warmup", str(warmup), "--repeat", str(repeat), "--verbose", "1"])
    # CPU
    out_json = ROOT / "outputs" / f"bench_{pruned_name}_cpu.json"
    sh([sys.executable, str(SCRIPTS / "bench_ts.py"),
        "--artifact", str(ts),
        "--device", "cpu", "--warmup", str(warmup), "--repeat", str(repeat),
        "--out", str(out_json)])

@app.command()
def summarize():
    sh([sys.executable, str(SCRIPTS / "summarize.py")])

@app.command()
def plots():
    sh([sys.executable, str(SCRIPTS / "make_plots.py")])

@app.command()
def full(
    config: Path = typer.Option(None, help="YAML config"),
    # Optional quick overrides
    epochs: int | None = None,
    batch: int | None = None,
):
    # Run the whole pipeline
    env()
    data(config=config)
    train(config=config, epochs=epochs, batch=batch)
    bench_baseline(config=config)
    prune(config=config)
    bench_torchscript(config=config)
    summarize()
    plots()
    typer.secho("\n[DONE] Full pipeline complete ✅", fg=typer.colors.GREEN)

if __name__ == "__main__":
    app()
