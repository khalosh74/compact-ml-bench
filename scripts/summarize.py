# scripts/summarize.py
import json, csv, glob, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
RUNS_DIR = ROOT / "runs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Unified schema expected by validator / plots
COLUMNS = [
    "model", "variant", "device", "threads",
    "acc_top1", "params_millions", "size_mb",
    "b1_ms", "ms_std_b1", "ms_p50_b1", "ms_p90_b1", "ms_p99_b1",
    "img_s_b1", "img_s_b8", "img_s_b32", "img_s_b128",
    "macs_g_flops", "energy_proxy_j",
    "matmul_precision", "warmup", "repeat", "batch_sizes"
]

# --- memory helper (used by bench_ts.py) ---
def psutil_rss_mb():
    """Resident Set Size (MB) for the current process, or None if psutil unavailable."""
    try:
        import os, psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return None


def load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def row_from_bench(path:str) -> dict:
    d = load_json(path)
    row = {k: None for k in COLUMNS}
    # identity / static
    row["model"] = d.get("model")
    row["variant"] = d.get("variant", "eager")
    row["device"] = d.get("device")
    row["threads"] = d.get("threads")
    row["params_millions"] = d.get("params_millions")
    row["size_mb"]  = d.get("size_mb")

    # latency / throughput
    row["b1_ms"]     = d.get("b1_ms")
    row["ms_std_b1"] = d.get("ms_std_b1")
    row["ms_p50_b1"] = d.get("ms_p50_b1")
    row["ms_p90_b1"] = d.get("ms_p90_b1")
    row["ms_p99_b1"] = d.get("ms_p99_b1")

    row["img_s_b1"]   = d.get("img_s_b1")
    row["img_s_b8"]   = d.get("img_s_b8")
    row["img_s_b32"]  = d.get("img_s_b32")
    row["img_s_b128"] = d.get("img_s_b128")

    # complexity / energy
    row["macs_g_flops"]  = d.get("macs_g_flops")
    row["energy_proxy_j"] = d.get("energy_proxy_j")

    # env / flags
    row["matmul_precision"] = d.get("matmul_precision")
    row["warmup"] = d.get("warmup")
    row["repeat"] = d.get("repeat")
    row["batch_sizes"] = d.get("batch_sizes")

    return row

def maybe_attach_accuracy(rows:list[dict]):
    """
    If we find runs/*/metrics.json, attach acc_top1 / params/model_size to matching rows (by model name).
    Fallback heuristic: same 'model' string.
    """
    # collect metrics by model name (last one wins)
    metrics = {}
    for mp in RUNS_DIR.glob("**/metrics.json"):
        m = load_json(mp)
        if not m: 
            continue
        key = str(m.get("model") or m.get("model_name") or m.get("meta",{}).get("model_name") or "").lower()
        if not key:
            # try to infer from checkpoint dir name
            key = mp.parent.name.lower()
        metrics[key] = {
            "acc_top1": m.get("acc_top1") or m.get("best_acc_top1"),
            "params_millions": m.get("params_millions"),
            "size_mb": m.get("size_mb") or m.get("size_mb")
        }

    for r in rows:
        k = (r.get("model") or "").lower()
        if k and k in metrics:
            r["acc_top1"] = r["acc_top1"] or metrics[k].get("acc_top1")
            r["params_millions"] = r["params_millions"] or metrics[k].get("params_millions")
            r["size_mb"] = r["size_mb"] or metrics[k].get("size_mb")

def main():
    # Gather all bench JSONs
    bench_paths = sorted(glob.glob(str(OUT_DIR / "bench_*.json")))
    rows = [row_from_bench(p) for p in bench_paths]

    # Attach accuracy if available
    maybe_attach_accuracy(rows)

    # Write CSV with consistent header
    out_csv = OUT_DIR / "results.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in COLUMNS})

    print(f"[SUMMARY] wrote {out_csv} with {len(rows)} rows")

if __name__ == "__main__":
    main()
