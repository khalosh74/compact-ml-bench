import os, json, csv, glob, re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
runs = sorted(ROOT.glob("runs/**/metrics.json"))
benches = sorted(ROOT.glob("outputs/*.json"))
out_dir = ROOT/"outputs"
out_dir.mkdir(parents=True, exist_ok=True)
csv_path = out_dir/"results.csv"

def safe_load(p):
    try:
        with open(p,"r") as f: return json.load(f)
    except Exception: return {}

def name_from_runpath(p: Path):
    # e.g., runs/baseline/metrics.json -> ("baseline","baseline")
    run = p.parent
    variant = run.name
    # try model name if present
    m = safe_load(p)
    model = m.get("model") or m.get("meta",{}).get("model_name") or ("resnet18" if "resnet" in variant else None)
    return model or "unknown", variant

def pick(v, *keys, default=None):
    for k in keys:
        if isinstance(k, tuple):
            for kk in k:
                if kk in v: return v[kk]
        else:
            if k in v: return v[k]
    return default

# index bench files (we expect keys: latency_ms_b1, device, maybe params/model_size)
bench_idx = {}
for b in benches:
    jb = safe_load(b)
    if not jb: continue
    dev = jb.get("device","")
    # infer variant key from filename: bench_<variant>_gpu.json or bench_latest.json beside runs
    key = b.stem.lower()
    bench_idx.setdefault(key, {}).update(jb)
    # also map by device for generic names
    bench_idx.setdefault(dev, {}).update(jb)

rows = []
for mj in runs:
    met = safe_load(mj)
    model, variant = name_from_runpath(mj)
    # core metrics (tolerant to schema differences)
    acc  = pick(met, "acc_top1", "best_acc_top1", default=None)
    params_m = pick(met, "params_millions", default=None)
    size_mb  = pick(met, "size_mb", "model_size_mb", default=None)
    notes = []
    # try to find sibling bench jsons in outputs by variant substring
    lat_cpu = lat_gpu = None
    for k,v in bench_idx.items():
        if variant.lower() in k:
            if v.get("device")=="cpu":  lat_cpu = v.get("latency_ms_b1", lat_cpu)
            if v.get("device") in ("cuda","gpu"): lat_gpu = v.get("latency_ms_b1", lat_gpu)
    # generic fallbacks
    if lat_cpu is None and "cpu" in bench_idx:  lat_cpu = bench_idx["cpu"].get("latency_ms_b1")
    if lat_gpu is None and "cuda" in bench_idx: lat_gpu = bench_idx["cuda"].get("latency_ms_b1")

    # collect metadata if present
    meta = met.get("meta",{})
    device = pick(met, "device", default=meta.get("device"))
    torch_v = pick(met, "torch", default=None)
    commit = pick(met, "git_commit", default=None)

    rows.append({
        "variant": variant,
        "model": model,
        "acc_top1": acc,
        "params_millions": params_m,
        "size_mb": size_mb,
        "latency_cpu_ms_b1": lat_cpu,
        "latency_gpu_ms_b1": lat_gpu,
        "device": device,
        "torch": torch_v,
        "git_commit": commit
    })

# write CSV (idempotent)
cols = ["variant","model","acc_top1","params_millions","size_mb","latency_cpu_ms_b1","latency_gpu_ms_b1","device","torch","git_commit"]
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in rows: w.writerow(r)

print(f"[SUMMARY] wrote {csv_path} with {len(rows)} rows")
