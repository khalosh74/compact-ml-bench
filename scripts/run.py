#!/usr/bin/env python3
import json, subprocess, sys
from pathlib import Path
import typer
import yaml

app = typer.Typer(add_completion=False, no_args_is_help=False)

def sh(cmd: list[str]):
    print(">>", " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(r.returncode)

def require_file(p: Path, hint: str):
    if not p.exists() or (p.is_file() and p.stat().st_size==0):
        raise SystemExit(f"[ERROR] Missing artifact: {p} ({hint})")

def require_bench_keys(p: Path):
    with open(p, "r", encoding="utf-8-sig") as f:
        j = json.load(f)
    if j.get("schema") != "bench.v2":
        raise SystemExit(f"[ERROR] {p} not bench.v2")
    if "latency_ms" not in j or "throughput_img_s" not in j:
        raise SystemExit(f"[ERROR] {p} missing latency/throughput")
    if "b1" not in j["latency_ms"]:
        raise SystemExit(f"[ERROR] {p} missing b1 latency")

@app.command("full")
def full(config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c", help="YAML config")):
    cfg = yaml.safe_load(open(config, "r", encoding="utf-8"))

    # 0) Baseline eager (GPU): bench + eval
    ckpt = Path("runs/baseline/best.pt"); require_file(ckpt, "baseline ckpt")
    bench_out = Path("outputs/bench_resnet18_eager_gpu.json")
    sh([sys.executable, "-m", "scripts.bench",
        "--checkpoint", str(ckpt), "--arch", "resnet18", "--device", "gpu",
        "--warmup", str(cfg.get("warmup",60)), "--repeat", str(cfg.get("repeat",600)),
        "--out", str(bench_out), "--variant", "resnet18_eager_gpu"])
    require_bench_keys(bench_out)
    sh([sys.executable, "-m", "scripts.eval", "--variant", "resnet18_eager_gpu",
        "--checkpoint", str(ckpt), "--arch", "resnet18", "--device", "gpu"])

    # 1) Pruned TS (GPU): bench + eval
    ts = Path("runs/resnet18_struct30/structured.ts"); require_file(ts, "pruned TS")
    bench_ts_out = Path("outputs/bench_resnet18_struct30_ts_gpu.json")
    sh([sys.executable, "-m", "scripts.bench_ts",
        "--artifact", str(ts), "--device", "gpu",
        "--warmup", str(cfg.get("warmup",60)), "--repeat", str(cfg.get("repeat",600)),
        "--out", str(bench_ts_out), "--variant", "resnet18_struct30_ts_gpu"])
    require_bench_keys(bench_ts_out)
    sh([sys.executable, "-m", "scripts.eval", "--variant", "resnet18_struct30_ts_gpu",
        "--artifact", str(ts), "--device", "gpu"])

    # 2) KD (optional)
    if cfg.get("enable_kd", False):
        kd_cfg = cfg.get("kd", {}) or {}
        kd_outdir = Path(kd_cfg.get("outdir","runs/kd_mobilenetv2"))
        # Build subprocess args with optional teacher_ckpt
        distill_cmd = [sys.executable, "-m", "scripts.distill",
            "--device","gpu","--epochs", str(kd_cfg.get("epochs",2)),
            "--batch", str(kd_cfg.get("batch",512)),
            "--workers", str(kd_cfg.get("workers",8)),
            "--lr", str(kd_cfg.get("lr",0.2)),
            "--alpha", str(kd_cfg.get("alpha",0.5)),
            "--temperature", str(kd_cfg.get("temperature",4.0)),
            "--teacher-arch", str(kd_cfg.get("teacher","resnet34")),
            "--student-arch", str(kd_cfg.get("student","mobilenet_v2")),
            "--outdir", str(kd_outdir)
        ]
        if kd_cfg.get("teacher_ckpt"):
            distill_cmd.extend(["--teacher-ckpt", str(kd_cfg.get("teacher_ckpt"))])
        sh(distill_cmd)

        stu_ckpt = kd_outdir/"best.pt"; require_file(stu_ckpt, "KD student best.pt")
        kd_variant = "kd_mobilenetv2_gpu"
        kd_bench = Path("outputs")/f"bench_{kd_variant}.json"
        sh([sys.executable, "-m", "scripts.bench",
            "--checkpoint", str(stu_ckpt), "--arch", "mobilenet_v2", "--device", "gpu",
            "--warmup", str(cfg.get("warmup",60)), "--repeat", str(cfg.get("repeat",600)),
            "--out", str(kd_bench), "--variant", kd_variant])
        require_bench_keys(kd_bench)
        sh([sys.executable, "-m", "scripts.eval", "--variant", kd_variant,
            "--checkpoint", str(stu_ckpt), "--arch", "mobilenet_v2", "--device", "gpu"])

    # 3) PTQ (optional)
    if cfg.get("enable_quant_ptq", False):
        q_cfg = cfg.get("quant", {}) or {}
        quant_outdir = Path(q_cfg.get("outdir","runs/quantized"))
        calib_batches = str((q_cfg.get("ptq") or {}).get("calib_batches", 64))

        def ptq_one(arch, ckpt_path, variant):
            sh([sys.executable, "-m", "scripts.quantize",
                "--arch", arch, "--checkpoint", str(ckpt_path),
                "--calib-batches", calib_batches, "--variant", variant, "--outdir", str(quant_outdir)])
            ts_path = quant_outdir/f"{variant}.ts"; require_file(ts_path, "PTQ TS")
            # INT8 eval on CPU
            sh([sys.executable, "-m", "scripts.eval", "--variant", variant, "--artifact", str(ts_path), "--device", "cpu"])
            # INT8 bench on CPU via bench_ts (threads=1)
            bench_path = Path("outputs")/f"bench_{variant}.json"
            sh([sys.executable, "-m", "scripts.bench_ts", "--artifact", str(ts_path),
                "--device","cpu","--threads","1","--warmup", str(cfg.get("warmup",60)),
                "--repeat", str(cfg.get("repeat",600)),"--out", str(bench_path), "--variant", variant])
            require_bench_keys(bench_path)

        ptq_one("resnet18", Path("runs/baseline/best.pt"), "resnet18_int8_fx_ptq_cpu_t1")
        kd_ckpt = Path((cfg.get("kd") or {}).get("outdir","runs/kd_mobilenetv2"))/"best.pt"
        if kd_ckpt.exists():
            ptq_one("mobilenet_v2", kd_ckpt, "kd_mobilenetv2_int8_fx_ptq_cpu_t1")

    # 4) QAT (optional)
    if cfg.get("enable_quant_qat", False):
        q_cfg = cfg.get("quant", {}) or {}
        quant_outdir = Path(q_cfg.get("outdir","runs/quantized"))
        qat_epochs = str((q_cfg.get("qat") or {}).get("epochs", 2))

        def qat_one(arch, ckpt_path, variant):
            sh([sys.executable, "-m", "scripts.qat",
                "--arch", arch, "--checkpoint", str(ckpt_path),
                "--epochs", qat_epochs, "--variant", variant, "--outdir", str(quant_outdir)])
            ts_path = quant_outdir/f"{variant}.ts"; require_file(ts_path, "QAT TS")
            sh([sys.executable, "-m", "scripts.eval", "--variant", variant, "--artifact", str(ts_path), "--device", "cpu"])
            bench_path = Path("outputs")/f"bench_{variant}.json"
            sh([sys.executable, "-m", "scripts.bench_ts", "--artifact", str(ts_path),
                "--device","cpu","--threads","1","--warmup", str(cfg.get("warmup",60)),
                "--repeat", str(cfg.get("repeat",600)),"--out", str(bench_path), "--variant", variant])
            require_bench_keys(bench_path)

        qat_one("resnet18", Path("runs/baseline/best.pt"), "resnet18_int8_fx_qat_cpu_t1")

    # 5) Summarize + validate + plots
    sh([sys.executable, "scripts/summarize.py"])
    sh([sys.executable, "scripts/validate_results.py"])
    sh([sys.executable, "scripts/make_plots.py"])

@app.callback(invoke_without_command=True)
def _root(ctx: typer.Context,
          config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c", help="YAML config")):
    if ctx.invoked_subcommand is None:
        full(config=config)

if __name__ == "__main__":
    app()
