import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

lc_csv = Path("outputs/learning_curve.csv")
out1   = Path("outputs/plot_bias_variance_acc.png")
out2   = Path("outputs/plot_bias_variance_gap.png")
out1.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(lc_csv).sort_values("fraction")

# Plot 1: Train/Test accuracy vs fraction
plt.figure()
plt.plot(df["fraction"], df["train_acc"], marker="o", label="Train acc")
plt.plot(df["fraction"], df["test_acc"],  marker="o", label="Test acc")
plt.xlabel("Training fraction")
plt.ylabel("Accuracy (%)")
plt.title("Learning Curve: Accuracy vs Data Fraction")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(out1)
plt.close()

# Plot 2: Generalization gap vs fraction
plt.figure()
plt.plot(df["fraction"], df["gap_pp"], marker="o", label="Gap (pp)")
plt.xlabel("Training fraction")
plt.ylabel("Train − Test (percentage points)")
plt.title("Generalization Gap vs Data Fraction")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(out2)
plt.close()

print(f"[PLOTS] Wrote: {out1} and {out2}")
