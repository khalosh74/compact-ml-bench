#!/usr/bin/env python3
import argparse, json, subprocess, threading, time
from pathlib import Path
import statistics as stats
import numpy as np

def file_size_mb(p: Path):
    try: return round(Path(p).stat().st_size/(1024*1024), 3)
    except: return None

class EnergySampler(threading.Thread):
    def __init__(self, interval_s=0.1):
        super().__init__(daemon=True); self.interval=interval_s; self.running=False; self.joules=0.0
    def run(self):
        self.running=True
        last = time.perf_counter()
        while self.running:
            try:
                res = subprocess.run(["nvidia-smi","--query-gpu=power.draw","--format=csv,noheader,nounits"], capture_output=True, text=True)
                power_w = float(res.stdout.strip().splitlines()[0])
                now = time.perf_counter(); dt = now-last; last=now
                self.joules += power_w * dt
            except Exception:
                pass
            time.sleep(self.interval)
    def stop(self): self.running=False

def percentile(xs, p):
    xs = sorted(xs); k = (len(xs)-1)*(p/100.0); f=int(k); c=min(f+1, len(xs)-1)
    return float(xs[f]) if f==c else float(xs[f] + (xs[c]-xs[f])*(k-f))

def run_bench(session, input_name, batches, warmup, repeat):
    lat, thr = {}, {}
    maxb = max(batches)
    example = np.random.randn(maxb,3,32,32).astype(np.float32)
    # warmup
    for _ in range(warmup):
        session.run(None, {input_name: example[:1]})
    # measure
    for b in batches:
        times=[]
        for _ in range(repeat):
            t0=time.perf_counter_ns()
            session.run(None, {input_name: example[:b]})
            dt=(time.perf_counter_ns()-t0)/1e6
            times.append(dt)
        mean = stats.mean(times); std = stats.pstdev(times) if len(times)>1 else 0.0
        lat[f"b{b}"] = {"mean":mean,"std":std,"p50":percentile(times,50),"p90":percentile(times,90),"p99":percentile(times,99),"samples":len(times)}
        thr[f"b{b}"] = (1000.0/mean)*b
    return lat, thr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--device", choices=["cpu","gpu","cuda"], default="cpu")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=60)
    ap.add_argument("--repeat", type=int, default=600)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--variant", type=str, required=True)
    args = ap.parse_args()

    try:
        import onnxruntime as ort
    except Exception as e:
        print(f"[ORT][SKIP] onnxruntime not available: {e}")
        return 0

    providers = ort.get_available_providers()
    want_gpu = args.device in ("gpu","cuda")
    use_gpu = (want_gpu and "CUDAExecutionProvider" in providers)
    device = "gpu" if use_gpu else "cpu"

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if not use_gpu:
        so.intra_op_num_threads = max(1, int(args.threads))
        so.inter_op_num_threads = 1

    sess = ort.InferenceSession(str(args.onnx), sess_options=so,
                                providers=(["CUDAExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]))
    input_name = sess.get_inputs()[0].name

    batches=[1,8,32,128]
    es=None
    if use_gpu:
        es=EnergySampler(0.1); es.start()
    lat, thr = run_bench(sess, input_name, batches, args.warmup, args.repeat)
    energy = es.joules if es else None
    if es: es.stop()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out,"w",encoding="utf-8") as f:
        json.dump({
            "schema":"bench.v2",
            "model":"unknown",
            "variant": args.variant,
            "device": device,
            "threads": None if device=="gpu" else int(args.threads),
            "params_millions": None,
            "size_mb": file_size_mb(args.onnx),
            "latency_ms": lat,
            "throughput_img_s": thr,
            "macs_g_flops": None,
            "energy_proxy_j": energy,
            "warmup": int(args.warmup),
            "repeat": int(args.repeat),
            "batch_sizes": [1,8,32,128]
        }, f)
    print(f"[ORT][BENCH] wrote {args.out} (device={device}, providers={providers})")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
