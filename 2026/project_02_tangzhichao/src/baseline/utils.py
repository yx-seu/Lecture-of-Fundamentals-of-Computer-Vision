import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        log("CUDA unavailable, falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_name if device_name else ("cuda" if torch.cuda.is_available() else "cpu"))


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_csv(rows: List[Dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "non_trainable_params": int(total_params - trainable_params),
    }


def get_gpu_memory() -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {
            "gpu_memory_allocated_mb": 0.0,
            "gpu_memory_reserved_mb": 0.0,
            "max_gpu_memory_allocated_mb": 0.0,
        }
    return {
        "gpu_memory_allocated_mb": torch.cuda.memory_allocated() / (1024 ** 2),
        "gpu_memory_reserved_mb": torch.cuda.memory_reserved() / (1024 ** 2),
        "max_gpu_memory_allocated_mb": torch.cuda.max_memory_allocated() / (1024 ** 2),
    }


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def flatten_dict(data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_dict(value, full_key))
        else:
            flat[full_key] = value
    return flat


def to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.generic,)):
        return value.item()
    return value


def format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, sec = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"
