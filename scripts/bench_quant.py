import argparse, json, os, time, torch
def measure_latency(model, warmups=20, repeats=400):
    model.eval(); x=torch.randn(1,3,32,32)
    with torch.inference_mode():
        for _ in range(warmups): model(x)
        t0=time.perf_counter()
        for _ in range(repeats): model(x)
        dt=(time.perf_counter()-t0)/repeats
    return dt*1000.0
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--repeat", type=int, default=400)
    ap.add_argument("--out", default="outputs/bench_quant_cpu.json")
    args=ap.parse_args()
    quantized=torch.jit.load(args.artifact, map_location="cpu")
    ms=measure_latency(quantized, warmups=args.warmup, repeats=args.repeat)
    out={"device":"cpu","latency_ms_b1": round(ms,3)}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out,"w") as f: json.dump(out,f,indent=2)
    print(json.dumps(out, indent=2))
if __name__=="__main__":
    main()
