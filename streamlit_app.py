import time, json, os
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import models, datasets, transforms
from PIL import Image
import streamlit as st
import pandas as pd

# --- Config: where artifacts are expected ---
BASELINE_PT = None
for p in [
    "runs/baseline_40e/best.pt",
    "runs/baseline/best.pt",
]:
    if Path(p).exists():
        BASELINE_PT = p
        break

PRUNED_TS = None
for p in [
    "runs/resnet18_struct30/structured.ts",
    "runs/resnet_struct30/structured.ts",
]:
    if Path(p).exists():
        PRUNED_TS = p
        break

RESULTS_CSV = Path("outputs/results.csv")
PLOT_SIZE = Path("outputs/plot_acc_vs_size.png")
PLOT_LAT  = Path("outputs/plot_acc_vs_latency.png")

CIFAR_STATS = ((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
CIFAR_CLASSES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

def load_baseline(checkpoint):
    # safe load (weights_only=True if available)
    try:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(checkpoint, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        meta = ckpt.get("meta", {})
    else:
        state_dict = ckpt
        meta = {}

    model_name = (meta.get("model_name") or "resnet18").lower()
    num_classes = int(meta.get("num_classes", 10))
    if model_name == "resnet18":
        model = models.resnet18(num_classes=num_classes)
    elif model_name == "resnet34":
        model = models.resnet34(num_classes=num_classes)
    else:
        model = models.resnet18(num_classes=num_classes)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, meta

@torch.inference_mode()
def run_one(model, img_tensor, device):
    # img_tensor: [3,32,32]
    x = img_tensor.unsqueeze(0).to(device)
    logits = model(x)
    probs = F.softmax(logits, dim=1).squeeze(0)
    conf, idx = torch.max(probs, dim=0)
    return int(idx.item()), float(conf.item())

@torch.inference_mode()
def timed_latency(model, img_tensor, device, warmup=20, repeat=50):
    x = img_tensor.unsqueeze(0).to(device)
    # warmup
    for _ in range(warmup):
        _ = model(x)
        if device == "cuda":
            torch.cuda.synchronize()
    # timed
    if device == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeat):
            _ = model(x)
        end.record()
        torch.cuda.synchronize()
        total_ms = start.elapsed_time(end)
        return total_ms / repeat
    else:
        t0 = time.perf_counter()
        for _ in range(repeat):
            _ = model(x)
        t1 = time.perf_counter()
        return (t1 - t0) * 1000.0 / repeat

# ---------- UI ----------
st.set_page_config(page_title="Compact ML Demo", layout="wide")
st.title("Compact ML: Baseline vs Pruned (CIFAR-10)")

left, right = st.columns([2,1])

with right:
    st.subheader("Setup")
    device = "cuda" if (torch.cuda.is_available() and st.toggle("Use GPU (if available)", value=False)) else "cpu"
    st.write(f"**Device:** `{device}`")
    st.write(f"**Baseline checkpoint:** `{BASELINE_PT or 'not found'}`")
    st.write(f"**Pruned TorchScript:** `{PRUNED_TS or 'not found'}`")

    st.divider()
    st.subheader("Load an image")
    upl = st.file_uploader("Upload a CIFAR-like image (will be resized to 32×32), or use a sample:", type=["png","jpg","jpeg"])
    sample_btn = st.button("Use a random CIFAR-10 test image")

tf = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize(*CIFAR_STATS),
])

raw_img = None
if upl is not None:
    raw_img = Image.open(upl).convert("RGB")
elif sample_btn:
    # sample from local CIFAR-10 test (downloaded by your pipeline)
    try:
        test = datasets.CIFAR10(root="data", train=False, download=True)
        idx = torch.randint(low=0, high=len(test), size=(1,)).item()
        img, label = test[idx]
        raw_img = img
        # convert PIL.Image (raw dataset img already 32x32, no normalize yet)
    except Exception as e:
        st.warning(f"Could not load CIFAR-10 test set: {e}")

if raw_img is not None:
    left.subheader("Input")
    left.image(raw_img, caption="Input image (resized to 32×32)")

    img_tensor = tf(raw_img)

    # Load baseline model
    if not BASELINE_PT or not Path(BASELINE_PT).exists():
        st.error("Baseline checkpoint not found. Run the pipeline first.")
    else:
        base_model, meta = load_baseline(BASELINE_PT)
        base_model.to(device)

    # Load pruned TorchScript
    pruned_module = None
    if PRUNED_TS and Path(PRUNED_TS).exists():
        pruned_module = torch.jit.load(PRUNED_TS, map_location=device)
        pruned_module.eval().to(device)

    # Run both
    if BASELINE_PT and Path(BASELINE_PT).exists():
        idx_b, conf_b = run_one(base_model, img_tensor, device)
        lat_b = timed_latency(base_model, img_tensor, device)
    else:
        idx_b, conf_b, lat_b = None, None, None

    if pruned_module is not None:
        idx_p, conf_p = run_one(pruned_module, img_tensor, device)
        lat_p = timed_latency(pruned_module, img_tensor, device)
    else:
        idx_p, conf_p, lat_p = None, None, None

    # Display side-by-side
    c1, c2 = left.columns(2)
    with c1:
        st.markdown("### Baseline")
        if idx_b is not None:
            st.write(f"**Pred:** {CIFAR_CLASSES[idx_b]}  \n**Conf:** {conf_b:.2%}  \n**Latency (b1):** {lat_b:.3f} ms")
        else:
            st.write("_Not available_")

    with c2:
        st.markdown("### Pruned (TorchScript)")
        if idx_p is not None:
            st.write(f"**Pred:** {CIFAR_CLASSES[idx_p]}  \n**Conf:** {conf_p:.2%}  \n**Latency (b1):** {lat_p:.3f} ms")
        else:
            st.write("_Not available_")

# Results + plots section
st.divider()
st.subheader("Overall Results")
if RESULTS_CSV.exists():
    df = pd.read_csv(RESULTS_CSV)
    st.dataframe(df, use_container_width=True)
else:
    st.info("Run the pipeline to generate outputs/results.csv")

cols = st.columns(2)
with cols[0]:
    if PLOT_SIZE.exists():
        st.image(str(PLOT_SIZE), caption="Accuracy vs Size")
with cols[1]:
    if PLOT_LAT.exists():
        st.image(str(PLOT_LAT), caption="Accuracy vs Latency")
