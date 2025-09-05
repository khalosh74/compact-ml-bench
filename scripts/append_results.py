import os, json, csv, sys

def row_from(tag, run_dir, gpu_json, cpu_json=None):
    m = json.load(open(os.path.join(run_dir,"metrics.json")))
    g = json.load(open(gpu_json))
    c = json.load(open(cpu_json)) if (cpu_json and os.path.exists(cpu_json)) else {}
    return {
        "variant": tag,
        "acc_top1": m.get("best_acc_top1"),
        "params_m": m.get("params_millions"),
        "size_mb": m.get("model_size_mb"),
        "sparsity_percent": m.get("post_prune_observed_sparsity_percent"),
        "latency_gpu_ms": g.get("latency_ms_b1"),
        "latency_cpu_ms": c.get("latency_ms_b1"),
    }

rows = []
# baseline
if os.path.exists("runs/baseline/metrics.json"):
    rows.append(row_from("baseline", "runs/baseline", "outputs/bench_latest.json"))
# pruned
if os.path.exists("runs/pruned/metrics.json"):
    gpu_json = "outputs/bench_latest.json"  # last run from bench.py (gpu)
    cpu_json = "outputs/bench_pruned_cpu.json"
    rows.append(row_from("pruned_l1_50", "runs/pruned", gpu_json, cpu_json))

os.makedirs("outputs", exist_ok=True)
fn = "outputs/results.csv"
write_header = not os.path.exists(fn)
with open(fn, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["variant","acc_top1","params_m","size_mb","sparsity_percent","latency_gpu_ms","latency_cpu_ms"])
    if write_header: w.writeheader()
    for r in rows: 
        if r: w.writerow(r)
print(f"Wrote/updated {fn}")
