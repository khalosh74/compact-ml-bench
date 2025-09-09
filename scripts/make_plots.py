import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
CSV = OUT / "results.csv"

def load_results():
    if not CSV.exists():
        raise FileNotFoundError(f"{CSV} not found. Run summarize.py first.")
    df = pd.read_csv(CSV)
    # Normalize expected columns
    for col in [
        "model","variant","device","acc_top1","params_millions","size_mb",
        "b1_ms","ms_std_b1","ms_p50_b1","ms_p90_b1","ms_p99_b1",
        "img_s_b1","img_s_b8","img_s_b32","img_s_b128",
        "macs_g_flops","energy_proxy_j","matmul_precision"
    ]:
        if col not in df.columns:
            df[col] = pd.NA
    # Coerce types
    num_cols = [
        "acc_top1","params_millions","size_mb",
        "b1_ms","ms_std_b1","ms_p50_b1","ms_p90_b1","ms_p99_b1",
        "img_s_b1","img_s_b8","img_s_b32","img_s_b128",
        "macs_g_flops","energy_proxy_j"
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Derived convenience
    df["label"] = df[["model","variant","device"]].astype(str).agg("/".join, axis=1)
    return df

def plot_acc_vs_size(df: pd.DataFrame):
    fig = plt.figure(figsize=(7,5))
    sub = df.dropna(subset=["acc_top1","size_mb"])
    if sub.empty:
        print("[PLOTS] No rows with acc_top1 & size_mb; skipping acc_vs_size.")
        return
    # One point per (model,variant,device) taking the best acc if duplicates
    sub = sub.sort_values("acc_top1", ascending=False).drop_duplicates(["model","variant","device"])
    plt.scatter(sub["size_mb"], sub["acc_top1"])
    for _, r in sub.iterrows():
        plt.annotate(r["label"], (r["size_mb"], r["acc_top1"]), fontsize=8, xytext=(3,3), textcoords="offset points")
    plt.xlabel("Model size (MB)")
    plt.ylabel("Top-1 accuracy (%)")
    plt.title("Accuracy vs Size")
    plt.grid(True, alpha=0.3)
    fig.tight_layout()
    out = OUT / "plot_acc_vs_size.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[PLOTS] Wrote: {out}")

def plot_acc_vs_latency(df: pd.DataFrame):
    fig = plt.figure(figsize=(7,5))
    # Use b1_ms (batch=1) as the latency metric. Prefer GPU rows if any; else CPU.
    gpu = df[(df["device"] == "cuda") & df["b1_ms"].notna() & df["acc_top1"].notna()]
    cpu = df[(df["device"] == "cpu") & df["b1_ms"].notna() & df["acc_top1"].notna()]
    if not gpu.empty:
        sub = gpu
        dev_label = "GPU"
    elif not cpu.empty:
        sub = cpu
        dev_label = "CPU"
    else:
        print("[PLOTS] No rows with acc_top1 & b1_ms; skipping acc_vs_latency.")
        return
    # One point per (model,variant) picking best (lowest) latency
    sub = sub.sort_values("b1_ms", ascending=True).drop_duplicates(["model","variant"])
    plt.scatter(sub["b1_ms"], sub["acc_top1"])
    for _, r in sub.iterrows():
        plt.annotate(r["label"], (r["b1_ms"], r["acc_top1"]), fontsize=8, xytext=(3,3), textcoords="offset points")
    plt.xlabel(f"Batch-1 latency (ms) [{dev_label}] (lower is better)")
    plt.ylabel("Top-1 accuracy (%)")
    plt.title(f"Accuracy vs Latency ({dev_label})")
    plt.grid(True, alpha=0.3)
    plt.xscale("log")  # latency often benefits from log scale
    fig.tight_layout()
    out = OUT / "plot_acc_vs_latency.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[PLOTS] Wrote: {out}")

def main():
    df = load_results()
    OUT.mkdir(parents=True, exist_ok=True)
    plot_acc_vs_size(df)
    plot_acc_vs_latency(df)

if __name__ == "__main__":
    main()
