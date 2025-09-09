#!/usr/bin/env python3
import argparse, json, statistics as stats, time
from pathlib import Path
import numpy as np
import onnxruntime as ort

def percentile(xs, p):
    xs = sorted(xs); k = (len(xs)-1)*(p/100.0); f=int(k); c=min(f+1, len(xs)-1)
    return float(xs[f]) if f==c else float(xs[f] + (xs[c]-xs[f])*(k-f))

def get_static_batch(session) -> int | None:
    try:
        shp = session.get_inputs()[0].shape
        bs = shp[0]
        return int(bs) if isinstance(bs, int) and bs > 0 else None
    except Exception:
        return None

def run_once(session, input_name, arr):
    return session.run(None, {input_name: arr})

def run_bench(session, input_name, batches, warmup, repeat):
    lat = {}; thr = {}
    static_bs = get_static_batch(session)

    max_b = max(batches)
    example = np.random.randn(max_b, 3, 32, 32).astype(np.float32)

    # warmup (use b=1 or static_bs if defined)
    warm_b = 1 if (static_bs is None) else static_bs
    for _ in range(warmup):
        _ = run_once(session, input_name, example[:warm_b])

    for b in batches:
        times = []
        for _ in range(repeat):
            t0 = time.perf_counter_ns()
            if static_bs is None or static_bs == b:
                _ = run_once(session, input_name, example[:b])
                dt_ms = (time.perf_counter_ns() - t0) / 1e6
            elif static_bs == 1 and b > 1:
                # micro-batch fallback: accumulate b calls of batch-1
                for i in range(b):
                    _ = run_once(session, input_name, example[i:i+1])
                dt_ms = (time.perf_counter_ns() - t0) / 1e6
            else:
                # generic chunking by static_bs
                rem = b; start = 0
                while rem > 0:
                    chunk = min(static_bs, rem)
                    _ = run_once(session, input_name, example[start:start+chunk])
                    rem -= chunk; start += chunk
                dt_ms = (time.perf_counter_ns() - t0) / 1e6
            times.append(dt_ms)

        lat[f"b{b}"] = {
            "mean": float(stats.mean(times)),
            "std": float(stats.pstdev(times)) if len(times) > 1 else 0.0,
            "p50": percentile(times, 50),
            "p90": percentile(times, 90),
            "p99": percentile(times, 99),
            "samples": len(times),
        }
        thr[f"b{b}"] = (1000.0 / lat[f"b{b}"]["mean"]) * b
    return lat, thr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--device", choices=["cpu","gpu"], default="cpu")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=60)
    ap.add_argument("--repeat", type=int, default=600)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--variant", type=str, required=True)
    args = ap.parse_args()

    so = ort.SessionOptions()
    if args.device == "cpu":
        so.intra_op_num_threads = args.threads
        providers = ["CPUExecutionProvider"]
    else:
        providers = ["CUDAExecutionProvider"]

    sess = ort.InferenceSession(str(args.onnx), sess_options=so, providers=providers)
    input_name = sess.get_inputs()[0].name

    batches = [1,8,32,128]
    lat, thr = run_bench(sess, input_name, batches, args.warmup, args.repeat)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({
            "schema": "bench.v2",
            "model": "unknown",
            "variant": args.variant,
            "device": args.device,
            "threads": None if args.device=="gpu" else args.threads,
            "params_millions": None,
            "size_mb": round(Path(args.onnx).stat().st_size/(1024*1024), 3),
            "latency_ms": lat,
            "throughput_img_s": thr,
            "macs_g_flops": None,
            "energy_proxy_j": None,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "batch_sizes": batches
        }, f)
    print(f"[BENCH-ONNX] wrote {args.out}")

if __name__ == "__main__":
    main()
