# Compact ML Bench — experiments in model efficiency

A work-in-progress Python/PyTorch project exploring ways to make image-classification models smaller or faster, including pruning, quantization and distillation.

## What to inspect

- [Training](scripts/train.py), [structured pruning](scripts/structured_prune.py), [quantization](scripts/quantize.py), and [distillation](scripts/distill.py).
- [Benchmark code](scripts/bench.py), [metrics summary](scripts/summarize.py), and [plot generation](scripts/make_plots.py).
- [Makefile](Makefile) for environment checks and basic smoke tests.

## Quickstart (Linux/macOS or a Unix-like environment)

Create a virtual environment and install the dependencies before running the smoke test:

```bash
python3 -m venv .venv
source .venv/bin/activate
make setup
make status
make sanity
python scripts/download_data.py --dataset cifar10 --out data/
```

The smoke test does **not** run full training or establish benchmark performance.

## Running experiments

On Windows/PowerShell, [run_showcase.ps1](run_showcase.ps1) is an example of the full training/pruning/benchmark pipeline. It can be expensive to run; review its defaults before starting. Scripts in [scripts](scripts) can also be used individually.

Results are designed to be generated locally in `runs/` and `outputs/`. **This repository currently does not include the figures or `outputs/results.csv` referenced by the old README**, so no numerical accuracy/size/latency conclusions are claimed here.

To make the benchmark reviewable, the next step is to publish a representative, reproducible result set (configuration, environment, CSV, plots and limitations) after verifying it.

## Status

Experimental code, not a released or independently validated benchmark. Hardware, software versions, and training settings can substantially change the observed results.
