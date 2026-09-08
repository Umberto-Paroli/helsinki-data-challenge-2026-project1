"""
python main_convex.py

Stage 1: convex-only shape reconstruction (see train_convex.py for the
reasoning -- classical convex lightcurve inversion is close-to-unique,
unlike the general non-convex problem). Deliberately a separate entry
point/config from main.py, so the convex and non-convex reconstructions
can be run and compared independently for now. The natural next step --
feeding this stage's output mesh into main.py/train.py's non-convex
refinement as a warm start -- is not wired up yet.
"""

import sys

import torch
import yaml

from train_convex import train_convex


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)

    context = {
        "torch": torch,
        "LMAX": cfg["model"]["LMAX"],
        "log_every": cfg["training"]["log_every"],
    }

    cfg["model"]["dtype"] = eval(cfg["model"]["dtype"], {"__builtins__": {}}, context)
    cfg["validation"]["every"] = eval(cfg["validation"]["every"], {"__builtins__": {}}, context)
    return cfg


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "./config_convex.yaml"
    cfg = load_config(config_path)
    assert cfg.get("optimizer", "ADAM").upper() == "ADAM", (
        f"optimizer={cfg.get('optimizer')!r} -- only ADAM is wired in right now"
    )
    train_convex(cfg, config_path)
