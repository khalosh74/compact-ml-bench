from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


CSV_PATH = Path("outputs/results.csv")


@dataclass
class ModelEntry:
    variant: str
    model_name: str
    runtime: str          # "eager" | "ts" | "onnx"
    artifact: Path
    available: bool
    row: Dict


def _derive_runtime(variant: str) -> str:
    v = variant.lower()
    if "_onnx_" in v:
        return "onnx"
    if "_ts_" in v or v.endswith("_ts") or v.endswith(".ts"):
        return "ts"
    return "eager"


def _base_from_variant(variant: str) -> str:
    """
    'resnet18_onnx_cpu_t1'   -> 'resnet18'
    'kd_mobilenetv2_onnx'  -> 'kd_mobilenetv2'
    'resnet18_struct30_ts' -> 'resnet18_struct30'
    """
    v = variant
    for tok in ["_onnx_", "_ts_"]:
        if tok in v:
            return v.split(tok, 1)[0]
    # fallbacks
    v2 = re.sub(r"_eager_.*$", "", v)
    return v2


def _artifact_for_variant(variant: str, runtime: str, model_col: str) -> Path:
    v = variant.lower()
    base = _base_from_variant(v)

    if runtime == "eager":
        # primary heuristics
        if base.startswith("resnet18"):
            return Path("runs/baseline/best.pt")
        if base.startswith("kd_mobilenetv2") or "mobilenet_v2" in model_col:
            return Path("runs/kd_mobilenetv2/best.pt")
        # generic fallbacks
        cand = [
            Path(f"runs/{base}/best.pt"),
            Path(f"runs/{model_col}/best.pt"),
        ]
        for p in cand:
            if p.exists():
                return p
        return cand[0]

    if runtime == "ts":
        # known artifacts
        if "resnet18_struct30" in base:
            return Path("runs/resnet18_struct30/structured.ts")
        if "int8" in v:
            return Path(f"runs/quantized/{variant}.ts")
        # generic fallback: any .ts under runs/<base>
        for p in Path("runs").glob(f"{base}/**/*.ts"):
            return p
        return Path(f"runs/{base}/{variant}.ts")

    # runtime == "onnx"
    cand = [
        Path(f"runs/onnx/{base}.onnx"),
        Path(f"runs/onnx_quant/{variant}.onnx"),
    ]
    for p in cand:
        if p.exists():
            return p
    return cand[0]


def load_results(csv_path: Path = CSV_PATH) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame()
    # handle potential BOM
    try:
        df = pd.read_csv(csv_path)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    return df


def build_registry(df: pd.DataFrame) -> List[ModelEntry]:
    entries: List[ModelEntry] = []
    if df is None or df.empty:
        return entries
    for _, row in df.iterrows():
        variant = str(row.get("variant", "")).strip()
        model_col = str(row.get("model", "")).strip()
        if not variant:
            continue
        rt = _derive_runtime(variant)
        art = _artifact_for_variant(variant, rt, model_col)
        entries.append(
            ModelEntry(
                variant=variant,
                model_name=model_col or _base_from_variant(variant),
                runtime=rt,
                artifact=art,
                available=art.exists(),
                row=row.to_dict(),
            )
        )
    return entries
