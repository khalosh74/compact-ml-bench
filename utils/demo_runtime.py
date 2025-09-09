from __future__ import annotations
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# Optional heavy deps guarded inside methods
import torch
import torch.nn.functional as F

from utils.data_transforms import test_transform_cifar10
from utils.models import build_model


CIFAR10_CLASSES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]


@dataclass
class TimingStats:
    mean: float
    std: float
    p50: float
    p90: float
    p99: float
    samples: int


def _percentiles(vals: List[float]) -> Tuple[float,float,float,float]:
    xs = sorted(vals)
    n = len(xs)
    def pct(p):
        k = (n - 1) * (p / 100.0)
        f = int(k)
        c = min(f + 1, n - 1)
        return float(xs[f]) if f == c else float(xs[f] + (xs[c] - xs[f]) * (k - f))
    return pct(50), pct(90), pct(99), float(np.std(xs, ddof=0) if n > 1 else 0.0)


def _to_tensor_batch(imgs: List[Image.Image]) -> torch.Tensor:
    tfm = test_transform_cifar10()
    xs = [tfm(im) for im in imgs]
    return torch.stack(xs, dim=0).contiguous()  # NCHW


class BaseRunner:
    runtime: str
    device: str

    def warmup(self, batch: int = 1, iters: int = 10):
        raise NotImplementedError

    def predict(self, imgs: List[Image.Image], repeats: int = 50) -> Tuple[np.ndarray, Dict[str, TimingStats]]:
        raise NotImplementedError

    @staticmethod
    def topk(logits: np.ndarray, k: int = 3) -> List[List[Tuple[str, float]]]:
        # logits: (N, 10)
        t = torch.from_numpy(logits)
        probs = F.softmax(t, dim=1)
        vals, idxs = probs.topk(k, dim=1)
        out = []
        for row_vals, row_idxs in zip(vals.tolist(), idxs.tolist()):
            out.append([(CIFAR10_CLASSES[i], float(v)) for i, v in zip(row_idxs, row_vals)])
        return out


class EagerRunner(BaseRunner):
    def __init__(self, arch: str, ckpt: Path, device: str = "auto"):
        self.runtime = "eager"
        if device == "gpu" and torch.cuda.is_available():
            self.dev = torch.device("cuda")
            self.device = "gpu"
        else:
            self.dev = torch.device("cpu")
            self.device = "cpu"
        self.model = build_model(arch, num_classes=10)
        # load weights (state_dict or raw)
        try:
            ck = torch.load(ckpt, map_location="cpu", weights_only=True)
        except TypeError:
            ck = torch.load(ckpt, map_location="cpu")
        sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
        self.model.load_state_dict(sd, strict=False)
        self.model.eval().to(self.dev)
        self.example_shape = (1, 3, 32, 32)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            return self.model(x)

    def warmup(self, batch: int = 1, iters: int = 10):
        x = torch.randn(batch, 3, 32, 32, device=self.dev)
        for _ in range(iters):
            _ = self._forward(x)

    def predict(self, imgs: List[Image.Image], repeats: int = 50):
        x = _to_tensor_batch(imgs).to(self.dev, non_blocking=True)
        times = []
        with torch.inference_mode():
            for _ in range(repeats):
                t0 = time.perf_counter_ns()
                y = self.model(x)
                dt_ms = (time.perf_counter_ns() - t0) / 1e6
                times.append(dt_ms)
        p50, p90, p99, std = _percentiles(times)
        stats = {"b{}".format(x.shape[0]): TimingStats(mean=float(np.mean(times)), std=std, p50=p50, p90=p90, p99=p99, samples=len(times))}
        return y.detach().cpu().numpy(), stats


class TSRunner(BaseRunner):
    def __init__(self, ts_path: Path, device: str = "auto"):
        self.runtime = "ts"
        if device == "gpu" and torch.cuda.is_available():
            self.dev = torch.device("cuda")
            self.device = "gpu"
        else:
            self.dev = torch.device("cpu")
            self.device = "cpu"
        self.model = torch.jit.load(str(ts_path), map_location=self.dev).eval()
        self.example_shape = (1, 3, 32, 32)

    def warmup(self, batch: int = 1, iters: int = 10):
        x = torch.randn(batch, 3, 32, 32, device=self.dev)
        with torch.inference_mode():
            for _ in range(iters):
                _ = self.model(x)

    def predict(self, imgs: List[Image.Image], repeats: int = 50):
        x = _to_tensor_batch(imgs).to(self.dev, non_blocking=True)
        times = []
        with torch.inference_mode():
            for _ in range(repeats):
                t0 = time.perf_counter_ns()
                y = self.model(x)
                dt_ms = (time.perf_counter_ns() - t0) / 1e6
                times.append(dt_ms)
        p50, p90, p99, std = _percentiles(times)
        stats = {"b{}".format(x.shape[0]): TimingStats(mean=float(np.mean(times)), std=std, p50=p50, p90=p90, p99=p99, samples=len(times))}
        return y.detach().cpu().numpy(), stats


class ONNXRunner(BaseRunner):
    def __init__(self, onnx_path: Path, device: str = "cpu", threads: int = 1):
        self.runtime = "onnx"
        self.device = "cpu"
        self.example_shape = (1, 3, 32, 32)
        import onnxruntime as ort  # lazy import
        so = ort.SessionOptions()
        if threads and threads > 0:
            so.intra_op_num_threads = int(threads)
        providers = ["CPUExecutionProvider"]
        if device == "gpu":
            # will still fall back if CUDA EP not installed
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self.device = "gpu"
        self.sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name

    def warmup(self, batch: int = 1, iters: int = 10):
        x = np.random.randn(batch, 3, 32, 32).astype(np.float32)
        for _ in range(iters):
            _ = self.sess.run(None, {self.input_name: x})

    def predict(self, imgs: List[Image.Image], repeats: int = 50):
        x_t = _to_tensor_batch(imgs)  # torch NCHW
        x = x_t.detach().cpu().numpy().astype(np.float32)
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter_ns()
            y = self.sess.run(None, {self.input_name: x})[0]
            dt_ms = (time.perf_counter_ns() - t0) / 1e6
            times.append(dt_ms)
        p50, p90, p99, std = _percentiles(times)
        stats = {"b{}".format(x.shape[0]): TimingStats(mean=float(np.mean(times)), std=std, p50=p50, p90=p90, p99=p99, samples=len(times))}
        return y, stats
