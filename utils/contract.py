from pathlib import Path

# Project root = two levels up from this file: utils/ -> (root)
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR     = ROOT / "data"
CIFAR10_DIR  = DATA_DIR / "cifar-10-batches-py"
OUTPUTS_DIR  = ROOT / "outputs"

# Canonical artifact filenames (unchanged)
BASELINE_CKPT = OUTPUTS_DIR / "baseline.pt"
COMPRESSED_TS = OUTPUTS_DIR / "compressed.ts"
ACC_EAGER_JSON = OUTPUTS_DIR / "acc_resnet18_eager_cpu.json"
ACC_TS_JSON    = OUTPUTS_DIR / "acc_resnet18_ts_cpu.json"
BENCH_EAGER_JSON = OUTPUTS_DIR / "bench_resnet18_eager_cpu.json"
BENCH_TS_JSON    = OUTPUTS_DIR / "bench_resnet18_ts_cpu.json"
RESULTS_CSV = OUTPUTS_DIR / "results.csv"
PLOT_LAT    = OUTPUTS_DIR / "plot_acc_vs_latency.png"
PLOT_SIZE   = OUTPUTS_DIR / "plot_acc_vs_size.png"

DEFAULT_ARCH = "resnet18"
NUM_CLASSES  = 10
