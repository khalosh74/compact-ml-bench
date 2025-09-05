# compact-ml-bench (Codespaces)
Day 1 setup for a compact-models demo (pruning · quantization · distillation). CPU-only here.

## Quickstart
source .venv/bin/activate
make status     # env info (PyTorch, CUDA flag)
make sanity     # tiny forward/backward + latency smoke test
python scripts/download_data.py --dataset cifar10 --out data/
## Results snapshot

Artifacts and plots are auto-generated from \uns/*/metrics.json\ and \outputs/*.json\.

![Accuracy vs Size](outputs/plot_acc_vs_size.png)
![Accuracy vs Latency](outputs/plot_acc_vs_latency.png)

See the full table in [outputs/results.csv](outputs/results.csv).
