# scripts/summarize.py
import json, csv, glob
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

COLUMNS = [
    "model","variant","device","threads",
    "acc_top1","params_millions","size_mb",
    "b1_ms","ms_std_b1","ms_p50_b1","ms_p90_b1","ms_p99_b1",
    "img_s_b1","img_s_b8","img_s_b32","img_s_b128",
    "macs_g_flops","energy_proxy_j"
]

def _get(d, *keys, default=None):
    cur=d
    for k in keys:
        if cur is None: return default
        cur = cur.get(k)
        if cur is None: return default
    return cur

def row_from_bench(path: Path):
    j = json.load(open(path, "r", encoding="utf-8-sig"))
    if j.get("schema") != "bench.v2":
        return None
    lat = j.get("latency_ms") or {}
    b1 = lat.get("b1") or {}
    thr = j.get("throughput_img_s") or {}
    return {
        "model": j.get("model"),
        "variant": j.get("variant") or path.stem,
        "device": j.get("device"),
        "threads": j.get("threads"),
        "params_millions": j.get("params_millions"),
        "size_mb": j.get("size_mb"),
        "b1_ms": _get(b1,"mean"),
        "ms_std_b1": _get(b1,"std"),
        "ms_p50_b1": _get(b1,"p50"),
        "ms_p90_b1": _get(b1,"p90"),
        "ms_p99_b1": _get(b1,"p99"),
        "img_s_b1": thr.get("b1"),
        "img_s_b8": thr.get("b8"),
        "img_s_b32": thr.get("b32"),
        "img_s_b128": thr.get("b128"),
        "macs_g_flops": j.get("macs_g_flops"),
        "energy_proxy_j": j.get("energy_proxy_j"),
    }

def attach_accuracy(rows):
    acc_files = {}
    for p in glob.glob(str(OUT / "acc_*.json")):
        j = json.load(open(p, "r", encoding="utf-8-sig"))
        key = Path(p).stem.replace("acc_","")
        acc_files[key] = j.get("acc_top1")
    for r in rows:
        if r is None: continue
        var = r.get("variant")
        if var in acc_files:
            r["acc_top1"] = acc_files[var]

def main():
    benches = []
    for p in Path(OUT).glob("bench_*.json"):
        row = row_from_bench(p)
        if row: benches.append(row)
    attach_accuracy(benches)
    out_csv = OUT/"results.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in benches:
            w.writerow({k: r.get(k) for k in COLUMNS})
    print(f"[SUMMARY] wrote {out_csv} with {len(benches)} rows.")

if __name__ == "__main__":
    main()

