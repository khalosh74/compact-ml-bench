#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("outputs")
CSV = OUT / "results.csv"

def main():
    if not CSV.exists():
        print("[PLOTS] results.csv not found. Run summarize.py first.")
        return
    df = pd.read_csv(CSV)
    req = ["acc_top1","size_mb","b1_ms"]
    if df[req].isnull().any().any():
        print("[PLOTS] Some rows missing required fields; plotting only complete rows.")
    df = df.dropna(subset=req, how="any")
    if df.empty:
        note = OUT/"plot_note.txt"
        note.write_text("No complete rows with acc_top1, size_mb, b1_ms. Run eval + benches.", encoding="utf-8")
        print(f"[PLOTS] Skipped. See {note}")
        return

    # Accuracy vs Size
    plt.figure()
    for v in sorted(df["variant"].unique()):
        d = df[df["variant"]==v]
        plt.scatter(d["size_mb"], d["acc_top1"], label=v)
    plt.xlabel("Model size (MB)")
    plt.ylabel("Top-1 accuracy (%)")
    plt.title("Accuracy vs Size (CIFAR-10)")
    plt.legend(fontsize=8)
    p1 = OUT/"plot_acc_vs_size.png"
    plt.savefig(p1, bbox_inches="tight"); plt.close()
    print(f"[PLOTS] {p1}")

    # Accuracy vs Latency (b1)
    plt.figure()
    for v in sorted(df["variant"].unique()):
        d = df[df["variant"]==v]
        plt.scatter(d["b1_ms"], d["acc_top1"], label=v)
    plt.xlabel("Latency @ batch=1 (ms)")
    plt.ylabel("Top-1 accuracy (%)")
    plt.title("Accuracy vs Latency (b1)")
    plt.legend(fontsize=8)
    p2 = OUT/"plot_acc_vs_latency.png"
    plt.savefig(p2, bbox_inches="tight"); plt.close()
    print(f"[PLOTS] {p2}")

if __name__ == "__main__":
    main()
