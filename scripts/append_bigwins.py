import os, json, csv

rows=[]
def add(tag, run_dir, gpu_json=None, cpu_json=None):
    if not os.path.exists(os.path.join(run_dir,"metrics.json")): return
    m = json.load(open(os.path.join(run_dir,"metrics.json")))
    g = json.load(open(gpu_json)) if gpu_json and os.path.exists(gpu_json) else {}
    c = json.load(open(cpu_json)) if cpu_json and os.path.exists(cpu_json) else {}
    rows.append({
        "variant": tag,
        "acc_top1": m.get("best_acc_top1") or m.get("acc_top1"),
        "params_m": m.get("params_millions"),
        "size_mb": m.get("model_size_mb") or m.get("size_mb"),
        "latency_gpu_ms": g.get("latency_ms_b1"),
        "latency_cpu_ms": c.get("latency_ms_b1"),
    })

add("mnv2_baseline", "runs/mnv2_baseline", "outputs/bench_latest.json")
add("mnv2_distilled", "runs/mnv2_distilled", "outputs/bench_latest.json")
# PTQ INT8 (CPU-only) — size/acc in runs/mnv2_quant/metrics.json; bench in outputs/bench_mnv2_quant_cpu.json
if os.path.exists("runs/mnv2_quant/metrics.json"):
    qm = json.load(open("runs/mnv2_quant/metrics.json"))
    qb = json.load(open("outputs/bench_mnv2_quant_cpu.json"))
    rows.append({
        "variant":"mnv2_quant_int8", "acc_top1": qm.get("acc_top1"),
        "params_m": None, "size_mb": qm.get("size_mb"),
        "latency_gpu_ms": None, "latency_cpu_ms": qb.get("latency_ms_b1")
    })
add("mnv2_struct30", "runs/mnv2_struct30", "outputs/bench_latest.json", "outputs/bench_mnv2_struct30_cpu.json")

os.makedirs("outputs", exist_ok=True)
fn="outputs/results.csv"; write_header=not os.path.exists(fn)
with open(fn,"a",newline="") as f:
    w=csv.DictWriter(f, fieldnames=["variant","acc_top1","params_m","size_mb","latency_gpu_ms","latency_cpu_ms"])
    if write_header: w.writeheader()
    for r in rows: w.writerow(r)
print("Updated", fn)
