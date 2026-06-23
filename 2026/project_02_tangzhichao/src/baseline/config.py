from pathlib import Path
from typing import Any, Dict

import yaml

from .utils import ensure_dir


REQUIRED_TOP_LEVEL_KEYS = ["project_name", "seed", "data", "model", "train", "output"]


def load_config(config_path: str) -> Dict[str, Any]:
    config_file = Path(config_path)
    with config_file.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    for key in REQUIRED_TOP_LEVEL_KEYS:
        if key not in config:
            raise KeyError(f"Missing required config key: {key}")

    output_root = ensure_dir(config["output"]["root"])
    config["output"]["root"] = str(output_root)
    return config
