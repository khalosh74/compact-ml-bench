# scripts/validate_results.py
import sys, json
from pathlib import Path
import pandas as pd

REQUIRED = [
    "model","variant","device",
    "acc_top1","size_mb","params_millions",
    "b1_ms","ms_p50_b1","ms_p99_b1","img_s_b32",
    "macs_g_flops","energy_proxy_j"
]

def main():
    csv = Path("outputs/results.csv")
    if not csv.exists():
        print("[VALIDATE] outputs/results.csv not found. Run summarize.py first.")
        sys.exit(2)
    df = pd.read_csv(csv)
    missing_cols = [c for c in REQUIRED if c not in df.columns]
    if missing_cols:
        print("[VALIDATE] Missing columns:", ", ".join(missing_cols))
        sys.exit(3)
    print("[VALIDATE] OK. Schema looks good.")
    print(df.tail(5).to_string(index=False))

if __name__ == "__main__":
    main()
