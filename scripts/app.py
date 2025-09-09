#!/usr/bin/env python3
from __future__ import annotations
import os, sys
# ensure repo root is on sys.path so 'utils' can be imported when launched via Streamlit
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
try:
    import sitecustomize  # noqa: F401
except Exception:
    pass

import os, sys
# ensure repo root is on sys.path so 'utils' can be imported when launched via Streamlit
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
# if the repo has sitecustomize.py, make sure it runs even if added late
try:
    import sitecustomize  # noqa: F401
except Exception:
    pass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from utils.demo_registry import load_results, build_registry
from utils.demo_runtime import EagerRunner, TSRunner, ONNXRunner, BaseRunner, CIFAR10_CLASSES

st.set_page_config(
    page_title="Compact ML Bench  Demo",
    page_icon="",
    layout="wide",
)

@st.cache_data(show_spinner=False)
def _load_df(csv_path: str = "outputs/results.csv") -> pd.DataFrame:
    return load_results(Path(csv_path))

@st.cache_data(show_spinner=False)
def _load_registry() -> Tuple[pd.DataFrame, list]:
    df = _load_df()
    reg = build_registry(df)
    return df, reg

def _device_choices():
    import torch
    have_cuda = torch.cuda.is_available()
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        have_ort_cuda = "CUDAExecutionProvider" in providers
    except Exception:
        have_ort_cuda = False
    return dict(
        torch_cuda=have_cuda,
        ort_cuda=have_ort_cuda,
    )

def _make_runner(entry, device_choice: str) -> BaseRunner:
    rt = entry.runtime
    art = entry.artifact
    v = entry.variant.lower()
    if rt == "eager":
        # crude arch inference
        arch = "mobilenet_v2" if "mobilenet" in (entry.model_name or v) else "resnet18"
        dev = "gpu" if device_choice == "GPU" else "cpu"
        return EagerRunner(arch=arch, ckpt=art, device=dev)
    if rt == "ts":
        dev = "gpu" if device_choice == "GPU" else "cpu"
        return TSRunner(ts_path=art, device=dev)
    # rt == "onnx"
    dev = "gpu" if device_choice == "GPU" else "cpu"
    threads = 1 if dev == "cpu" else None
    return ONNXRunner(onnx_path=art, device=dev, threads=threads or 1)

def _load_sample_images(n: int = 6) -> List[Image.Image]:
    # Try CIFAR-10 test set if present
    try:
        from torchvision import datasets
        from utils.data_transforms import test_transform_cifar10
        ds = datasets.CIFAR10(root="data", train=False, download=False)
        out = []
        for i in range(min(n, len(ds))):
            img, _ = ds[i]
            # ds returns PIL already
            out.append(img)
        return out
    except Exception:
        # Placeholder gradient squares
        out = []
        for i in range(n):
            arr = np.linspace(0, 255, 32*32*3, dtype=np.uint8).reshape(32,32,3)
            out.append(Image.fromarray(arr))
        return out

def _show_model_card(entry, col):
    row = entry.row
    with col:
        st.caption(entry.variant)
        st.write(f"**Runtime:** `{entry.runtime}`    **Device:** `{row.get('device','?')}`")
        m1, m2, m3 = st.columns(3)
        m1.metric("acc_top1", f"{row.get('acc_top1','?')}")
        m2.metric("size_mb", f"{row.get('size_mb','?')}")
        m3.metric("b1_ms", f"{row.get('b1_ms','?')}")
        st.progress(1.0 if entry.available else 0.0, text=("available" if entry.available else "artifact missing"))
        st.code(str(entry.artifact), language="bash")

def _show_topk(preds: List[List[tuple]], header: str = "Top-k"):
    st.write(f"**{header}**")
    for i, row in enumerate(preds):
        cols = st.columns(len(row))
        for (cls, prob), c in zip(row, cols):
            c.metric(cls, f"{prob*100:.1f}%")

def _show_timing(stats_dict):
    # stats_dict is {"bX": TimingStats}
    for b, s in stats_dict.items():
        st.write(f"**Batch {b[1:]}**  mean: {s.mean:.3f} ms  |  p50: {s.p50:.3f}  |  p90: {s.p90:.3f}  |  p99: {s.p99:.3f}  |  n={s.samples}")

st.title(" Compact ML Bench  Interactive Demo")

with st.sidebar:
    st.header("Controls")
    df, registry = _load_registry()
    if not registry:
        st.error("No rows in outputs/results.csv. Run the pipeline first.")
    variants_avail = [e.variant for e in registry if e.available]
    variants_all   = [e.variant for e in registry]
    default_variant = "resnet18_eager_gpu" if "resnet18_eager_gpu" in variants_all else (variants_avail[0] if variants_avail else None)
    variant = st.selectbox("Variant", options=variants_all, index=(variants_all.index(default_variant) if default_variant in variants_all else 0))
    entry = next((e for e in registry if e.variant == variant), None)

    caps = _device_choices()
    device_opts = ["CPU"]
    # GPU offered if eager/ts and torch CUDA, or if onnx and ORT CUDA present
    if entry and entry.runtime in ("eager","ts") and caps["torch_cuda"]:
        device_opts.append("GPU")
    if entry and entry.runtime == "onnx" and caps["ort_cuda"]:
        device_opts.append("GPU")

    device_choice = st.radio("Target device", options=device_opts, index=0, horizontal=True)
    batch = st.select_slider("Batch size", options=[1,8,32,128], value=1)
    repeats = st.slider("Repeats", min_value=10, max_value=200, value=50, step=10)
    topk = st.slider("Top-k", min_value=1, max_value=5, value=3)
    st.divider()
    st.caption("Environment snapshot")
    try:
        import torch
        st.code(f"Torch {torch.__version__}  CUDA avail={torch.cuda.is_available()}", language="text")
    except Exception:
        st.code("Torch not available", language="text")
    try:
        import onnxruntime as ort
        st.code(f"ONNXRuntime providers: {ort.get_available_providers()}", language="text")
    except Exception:
        st.code("ONNXRuntime not available", language="text")

tab_cards, tab_live, tab_compare, tab_explain = st.tabs([" Model cards", " Live inference", " Compare & plots", " Explainer"])

with tab_cards:
    st.subheader("Discovered models")
    if not registry:
        st.info("No models discovered.")
    else:
        ncols = 3
        cols = st.columns(ncols)
        for i, e in enumerate(registry):
            _show_model_card(e, cols[i % ncols])

with tab_live:
    st.subheader("Run predictions")
    if not entry:
        st.info("Select a variant in the sidebar.")
    else:
        if not entry.available:
            st.error(f"Artifact missing for '{entry.variant}'. Path: {entry.artifact}")
        else:
            runner = _make_runner(entry, device_choice=device_choice)
            c1, c2 = st.columns([2, 1])
            with c1:
                st.write("**Images**")
                up = st.file_uploader("Upload images", type=["png","jpg","jpeg"], accept_multiple_files=True)
                imgs: List[Image.Image] = []
                if up:
                    for f in up:
                        try:
                            imgs.append(Image.open(f).convert("RGB").resize((32,32)))
                        except Exception:
                            pass
                if not imgs:
                    if st.button("Load CIFAR-10 samples"):
                        imgs = _load_sample_images(n=batch)
                if imgs:
                    # Fit batch to chosen size
                    if len(imgs) >= batch:
                        imgs = imgs[:batch]
                    else:
                        # repeat to fill batch
                        k = batch - len(imgs)
                        imgs = imgs + imgs[:k]
                    st.image(imgs, caption=[f"img {i}" for i in range(len(imgs))], width=96)

            with c2:
                st.write("**Run**")
                if st.button("Warm up", use_container_width=True):
                    with st.spinner("Warming up..."):
                        runner.warmup(batch=batch, iters=10)
                    st.success("Warm up done.")
                if st.button("Predict", type="primary", use_container_width=True):
                    with st.spinner("Running..."):
                        logits, stats = runner.predict(imgs, repeats=repeats)
                    # show timing + topk
                    _show_timing(stats)
                    preds = BaseRunner.topk(logits, k=topk)
                    _show_topk(preds, header="Top-k predictions")

with tab_compare:
    st.subheader("Compare results")
    df = _load_df()
    if df is None or df.empty:
        st.info("No results yet.")
    else:
        # small selection
        show = df[["model","variant","device","threads","acc_top1","size_mb","b1_ms","img_s_b32","energy_proxy_j"]].copy()
        st.dataframe(show, use_container_width=True, hide_index=True)
        # plots (static) if present
        plot1 = Path("outputs/plot_acc_vs_size.png")
        plot2 = Path("outputs/plot_acc_vs_latency.png")
        c1, c2 = st.columns(2)
        if plot1.exists(): c1.image(str(plot1), caption="Accuracy vs Size", use_column_width=True)
        if plot2.exists(): c2.image(str(plot2), caption="Accuracy vs Latency", use_column_width=True)

with tab_explain:
    st.subheader("Why pruning + INT8 + KD?")
    st.markdown(
        """
- **Pruning** removes entire channels/blocks  lower params & model size  better memory bandwidth and cache friendliness.
- **INT8 quantization** shrinks weights/activations  faster vectorized kernels and smaller memory footprint.
- **Knowledge Distillation (KD)** trains a compact student to mimic a larger teacher  recovers most of the lost accuracy.
- Your runs (from `results.csv`) drive this demo: no hard-coded numbers.
        """
    )
    # quick pull of a few rows for narrative
    df = _load_df()
    if df is not None and not df.empty:
        picks = []
        for key in ["resnet18_eager_gpu","resnet18_struct30_ts_gpu","resnet18_int8_fx_ptq_cpu_t1","kd_mobilenetv2_gpu","kd_mobilenetv2_onnx_cpu_t1"]:
            if (df["variant"] == key).any():
                picks.append(df[df["variant"] == key].iloc[0][["variant","acc_top1","size_mb","b1_ms","img_s_b32"]])
        if picks:
            st.dataframe(pd.DataFrame(picks), hide_index=True, use_container_width=True)

