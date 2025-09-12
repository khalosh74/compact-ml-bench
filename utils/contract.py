from pathlib import Path

# ---- Directories
DATA_DIR = Path("data")
CIFAR10_DIR = DATA_DIR / "cifar-10-batches-py"
OUTPUTS_DIR = Path("outputs")

# ---- Canonical artifact filenames (do not change later)
BASELINE_CKPT = OUTPUTS_DIR / "baseline.pt"               # PyTorch state_dict
COMPRESSED_TS = OUTPUTS_DIR / "compressed.ts"             # TorchScript int8

ACC_EAGER_JSON = OUTPUTS_DIR / "acc_resnet18_eager_cpu.json"
ACC_TS_JSON    = OUTPUTS_DIR / "acc_resnet18_ts_cpu.json"

BENCH_EAGER_JSON = OUTPUTS_DIR / "bench_resnet18_eager_cpu.json"
BENCH_TS_JSON    = OUTPUTS_DIR / "bench_resnet18_ts_cpu.json"

RESULTS_CSV = OUTPUTS_DIR / "results.csv"
PLOT_LAT    = OUTPUTS_DIR / "plot_acc_vs_latency.png"
PLOT_SIZE   = OUTPUTS_DIR / "plot_acc_vs_size.png"

# ---- Model defaults
DEFAULT_ARCH = "resnet18"
NUM_CLASSES  = 10
