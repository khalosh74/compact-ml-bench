# compact-ml-bench (Codespaces)
Day 1 setup for a compact-models demo (pruning · quantization · distillation). CPU-only here.

## Quickstart
source .venv/bin/activate
make status     # env info (PyTorch, CUDA flag)
make sanity     # tiny forward/backward + latency smoke test
python scripts/download_data.py --dataset cifar10 --out data/
