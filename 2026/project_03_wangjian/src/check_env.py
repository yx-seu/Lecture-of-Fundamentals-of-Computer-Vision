"""Check whether the project environment is ready."""

from __future__ import annotations

import importlib.util
import sys


PACKAGES = [
    ("numpy", "array operations"),
    ("pandas", "CSV result tables"),
    ("PIL", "image fallback I/O"),
    ("cv2", "OpenCV image I/O and GrabCut fallback"),
    ("ultralytics", "YOLOv11 detection"),
    ("torch", "GPU acceleration"),
    ("sam2", "SAM 2 segmentation"),
    ("labelme", "optional manual annotation tool"),
]


def has_package(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def main() -> None:
    print("Python:", sys.version.replace("\n", " "))
    print("Executable:", sys.executable)
    print()

    for module_name, purpose in PACKAGES:
        status = "OK" if has_package(module_name) else "MISSING"
        print(f"{module_name:12s} {status:8s} - {purpose}")

    if has_package("torch"):
        import torch

        print()
        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("CUDA device:", torch.cuda.get_device_name(0))


if __name__ == "__main__":
    main()
