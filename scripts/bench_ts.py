import argparse, json, os, time, torch

def measure_latency_ts(model, device="cpu", warmups=20, repeats=200):
    model.eval()
    x = torch.randn(1,3,32,32, device=device)
    with torch.inference_mode():
        if device == "cuda":
            starter = torch.cuda.Event(enable_timing=True)
            ender   = torch.cuda.Event(enable_timing=True)
            for _ in range(warmups): model(x)
            torch.cuda.synchronize()
            starter.record()
            for _ in range(repeats): model(x)
            ender.record()
            torch.cuda.synchronize()
            return float(starter.elapsed_time(ender) / repeats)
        else:
            for _ in range(warmups): model(x)
            t0 = time.perf_counter()
            for _ in range(repeats): model(x)
            return (time.perf_counter()-t0)*1000.0/repeats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True, help="TorchScript .ts file")
    ap.add_argument("--device", choices=["cpu","gpu"], default="cpu")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--repeat", type=int, default=200)
    ap.add_argument("--out", default="outputs/bench_ts.json")
    ap.add_argument("--verbose", type=int, default=1)
    args = ap.parse_args()

    dev = "cuda" if (args.device=="gpu" and torch.cuda.is_available()) else "cpu"
    if args.verbose: print(f"[BENCH-TS] device={dev}", flush=True)

    module = torch.jit.load(args.artifact, map_location=dev)
    if dev == "cuda":
        module.to("cuda")

    ms = measure_latency_ts(module, device=dev, warmups=args.warmup, repeats=args.repeat)
    out = {"device": dev, "latency_ms_b1": round(ms,3)}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f: json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2), flush=True)

if __name__ == "__main__":
    main()
