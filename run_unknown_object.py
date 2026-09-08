"""
run_unknown_object.py
======================

Runs the full 3-step pipeline (omega/period calibration -> convex ->
refine) on a NEW object you do NOT have ground truth for, using
config_calibrate_unknown.yaml / config_convex_unknown.yaml /
config_refine_unknown.yaml (edit those first -- see the comments in each,
most importantly filling in the "<-- FROM SEARCH" hyperparameters from
your winning search.py trial, and the data/cylinder_radius/camera_distance
placeholders for this specific object).

This is a plain SEQUENTIAL driver, not a search -- three real subprocess
runs, one after another, with output streamed live to your terminal
(unlike search.py's silent polling loop) since you'll want to watch a
real run rather than a search trial. Two values are threaded through
automatically so you don't have to copy them by hand:
  - physical.period: from step 1's calibrated omega -> written into both
    step 2's and step 3's config before they run.
  - model.warm_start_checkpoint: from step 2's own checkpoint.pt -> written
    into step 3's config before it runs.

Usage:
    python DIP_Danilo/run_unknown_object.py
"""
from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path
from typing import Optional

import torch
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent          # .../DIP_Danilo
PROJECT_ROOT = SCRIPT_DIR.parent                        # .../Helsinki Asteroid Challenge

CONFIG_CALIBRATE = SCRIPT_DIR / "config_calibrate_unknown.yaml"
CONFIG_CONVEX = SCRIPT_DIR / "config_convex_unknown.yaml"
CONFIG_REFINE = SCRIPT_DIR / "config_refine_unknown.yaml"

SCRATCH_DIR = SCRIPT_DIR / "unknown_object_scratch"    # generated (period/checkpoint-filled) configs go here


def _run(entry_script: str, config_path: Path) -> Path:
    """Runs `python <entry_script> <config_path>` as a subprocess (cwd=
    PROJECT_ROOT, matching how you run these by hand), streaming its
    output live line-by-line so you can watch progress, and returns the
    run_dir it reports. main.py/main_convex.py both print
    "run folder: <path>" early on -- same convention search.py's own
    subprocess runner relies on to chain steps together."""
    cmd = [sys.executable, str(SCRIPT_DIR / entry_script), str(config_path)]
    print(f"\n{'=' * 80}\n>>> running: {' '.join(cmd)}\n{'=' * 80}")
    proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
    run_dir: Optional[Path] = None
    for line in proc.stdout:
        print(line, end="")
        if run_dir is None and line.startswith("run folder: "):
            run_dir = Path(line[len("run folder: "):].strip())
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"{entry_script} exited with code {ret} -- see output above")
    if run_dir is None:
        raise RuntimeError(f"{entry_script} never printed a 'run folder: ' line -- "
                            f"can't chain to the next step")
    return run_dir


def _read_calibrated_period(checkpoint_path: Path) -> float:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    return -2 * math.pi / ckpt["omega"].item()


def _load(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _dump(cfg: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)


def main():
    SCRATCH_DIR.mkdir(exist_ok=True)

    print("\n### STEP 1/3: omega/period calibration ###")
    calib_run_dir = _run("main_convex.py", CONFIG_CALIBRATE)
    period = _read_calibrated_period(calib_run_dir / "checkpoint.pt")
    print(f"\n[run_unknown_object] calibrated period = {period:.4f}")

    print("\n### STEP 2/3: convex-stage reconstruction ###")
    convex_cfg = _load(CONFIG_CONVEX)
    convex_cfg["physical"]["period"] = period
    convex_cfg_path = SCRATCH_DIR / "convex_final.yaml"
    _dump(convex_cfg, convex_cfg_path)
    convex_run_dir = _run("main_convex.py", convex_cfg_path)
    convex_ckpt = convex_run_dir / "checkpoint.pt"
    print(f"\n[run_unknown_object] convex checkpoint = {convex_ckpt}")

    print("\n### STEP 3/3: non-convex refinement ###")
    refine_cfg = _load(CONFIG_REFINE)
    refine_cfg["physical"]["period"] = period
    refine_cfg["model"]["warm_start_checkpoint"] = str(convex_ckpt)
    refine_cfg_path = SCRATCH_DIR / "refine_final.yaml"
    _dump(refine_cfg, refine_cfg_path)
    refine_run_dir = _run("main.py", refine_cfg_path)

    print(f"\n{'=' * 80}\nDONE. Final reconstruction: {refine_run_dir / 'final_mesh.stl'}\n{'=' * 80}")


if __name__ == "__main__":
    main()
