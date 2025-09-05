import pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

out_dir = Path("outputs"); out_dir.mkdir(exist_ok=True)
df = pd.read_csv(out_dir/"results.csv")
df = df.sort_values(by=["model","variant"])

# 1) Accuracy vs Size
plt.figure()
for key,grp in df.groupby("model"):
    x = grp["size_mb"]; y = grp["acc_top1"]; lbls = grp["variant"]
    plt.scatter(x,y,label=key)
    for xi,yi,lb in zip(x,y,lbls):
        if pd.notna(xi) and pd.notna(yi):
            plt.annotate(lb, (xi,yi), textcoords="offset points", xytext=(4,4), fontsize=8)
plt.xlabel("Size (MB)"); plt.ylabel("Accuracy (Top-1 %)"); plt.title("Accuracy vs Size")
plt.legend(); plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout(); plt.savefig(out_dir/"plot_acc_vs_size.png", dpi=160)

# 2) Accuracy vs Latency (prefer GPU if present else CPU)
lat_col = "latency_gpu_ms_b1" if df["latency_gpu_ms_b1"].notna().any() else "latency_cpu_ms_b1"
plt.figure()
for key,grp in df.groupby("model"):
    x = grp[lat_col]; y = grp["acc_top1"]; lbls = grp["variant"]
    plt.scatter(x,y,label=key)
    for xi,yi,lb in zip(x,y,lbls):
        if pd.notna(xi) and pd.notna(yi):
            plt.annotate(lb, (xi,yi), textcoords="offset points", xytext=(4,4), fontsize=8)
plt.xlabel(f"Latency b=1 (ms) [{lat_col}]"); plt.ylabel("Accuracy (Top-1 %)"); plt.title("Accuracy vs Latency")
plt.legend(); plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout(); plt.savefig(out_dir/"plot_acc_vs_latency.png", dpi=160)

print("[PLOTS] Wrote:", out_dir/"plot_acc_vs_size.png", "and", out_dir/"plot_acc_vs_latency.png")
