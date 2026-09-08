"""
search.py
=========

Hyperparameter search across BOTH pipeline stages (train_convex.py /
train.py's warmstart_points refinement), evaluated on the 3 objects you
DO have ground truth for, so the winning config can be trusted to
generalize reasonably to future objects you won't. Not a grid search --
with ~10 free hyperparameters per stage and only a few hundred trainings
fitting in your time budget, a full grid is hopeless (even 3 values per
parameter would be 3^10). Uses Optuna's TPE sampler (a Bayesian-ish
search: models which regions of hyperparameter space look promising from
trials seen so far, samples the next trial accordingly) plus a median
pruner (kills a trial early if it's clearly worse than other trials were
at the same point, instead of always running the full step budget).

IMPORTANT, and specific to your situation: on a real unknown object there
will be no GT, so nothing like `best_val` (checkpoint picked by peeking at
a GT-based score during training) is available -- you can only run for a
fixed step budget or stop on a GT-independent signal (the loss-plateau /
lr-decay mechanism already in train.py/train_convex.py). So this script
optimizes and ranks configs by the FINAL checkpoint's score (what you'd
actually get on an unknown object), not best_val's -- best_val is still
computed and reported alongside, purely as a reference upper bound / to
gauge how much headroom GT-based early stopping would have bought you.

Two phases, run separately (see `--phase`):
  convex  -- searches train_convex.py's hyperparameters, using
             `main_convex.py` as a subprocess per (trial, object).
  refine  -- searches train.py's warmstart_points hyperparameters, warm-
             started from the winning convex trial's per-object
             checkpoints (run `--phase convex` first).

Usage:
  python ./search.py --phase convex --hours 40
  python ./search.py --phase refine --hours 40

Resumable: trials are persisted to a SQLite file (`--storage`, default
`./search.db`) via Optuna, so killing the script (Ctrl-C, SSH
drop, out of time) and re-running the SAME command later continues from
where it left off instead of losing progress -- essential over a 3.5-day
unattended run.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import optuna
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent          # .../
PROJECT_ROOT_DEFAULT = SCRIPT_DIR.parent               # .../Helsinki Asteroid Challenge

# ---------------------------------------------------------------------
# The 3 objects you have GT for -- same file-naming convention as
# config_adam.yaml, paths relative to PROJECT_ROOT except gt_path (given
# as absolute). cylinder_radius is a KNOWN PHYSICAL property of each
# object (not a free hyperparameter) -- fixed per object, never searched.
# ---------------------------------------------------------------------

def _object(num: int, cylinder_radius: float) -> dict:
    return {
        "num": num,
        "cylinder_radius": cylinder_radius,
        "intensity_file": f"./Data/AsteroidModel0{num}_shape_public/"
                           f"Asteroid{num}_lightcurve_data/Asteroid0{num}_lightcurve_intensity.txt",
        "binary_file": f"./Data/AsteroidModel0{num}_shape_public/"
                        f"Asteroid{num}_lightcurve_data/Asteroid0{num}_lightcurve_binary.txt",
        # Blender paths always recorded, but only actually used (data key
        # added to a trial's config, blender_loss_weight searched) when
        # `_blender_available` finds both files on disk for this object --
        # not every object necessarily has published Blender data yet.
        "intensity_file_blender": f"./Data/AsteroidModel0{num}_shape_public/"
                                   f"Asteroid{num}_lightcurve_data/Asteroid0{num}_lightcurve_intensity_blender.txt",
        "binary_file_blender": f"./Data/AsteroidModel0{num}_shape_public/"
                                f"Asteroid{num}_lightcurve_data/Asteroid0{num}_lightcurve_binary_blender.txt",
        "gt_path": f"./Data/AsteroidModel0{num}_shape_public/"
                    f"asteroid{num}.stl",
    }


OBJECTS = [
    _object(1, 1.12),
    _object(2, 1.42),
    _object(3, 0.88),
]

# ---------------------------------------------------------------------
# Fixed (not searched) settings -- keep runtime per trial predictable.
# nrows=12 chosen to match the ~10-15 min/run you already measured.
# Adjust STEPS if your actual measured per-trial time is very different
# from that estimate once a few real trials have run.
# ---------------------------------------------------------------------

NROWS = 12
LMAX = 9
STEPS_CONVEX = 5000
STEPS_REFINE = 5000
VALIDATION_EVERY = 20     # -> 20 validation checkpoints/run: granularity for pruning polling
VOXEL_SIZE = 0.02
SIDEVIEW_N_DIRECTIONS = 8
SIDEVIEW_RESOLUTION = 256

VAL_CFG = {
    "voxel_size": VOXEL_SIZE,
    "sideview_n_directions": SIDEVIEW_N_DIRECTIONS,
    "sideview_resolution": SIDEVIEW_RESOLUTION,
}

POLL_INTERVAL_SEC = 20      # how often to check on a running subprocess
SUBPROCESS_TIMEOUT_SEC = 45 * 60   # hard safety cap per (trial, object) run

# lr_decay_factor/lr_max_decays used to be searched (2 extra dimensions on
# top of use_lr_decay's on/off). Fixed instead now that blender_loss_weight
# needs a search dimension too, to keep total search-space size in check.
FIXED_LR_DECAY_FACTOR = 0.5
FIXED_LR_MAX_DECAYS = 3

BLENDER_LOSS_WEIGHT_RANGE = (0.1, 5.0)   # log-scale search range, only used for objects with Blender data

# One-off omega/period calibration (see calibrate_period_for_object) --
# cheap (a few minutes/object, shape frozen) and independent of every other
# hyperparameter, so it's done ONCE per object before any search trial
# runs, cached to CALIBRATED_PERIODS_PATH, and reused by BOTH phases
# (convex search trials, and later the refine search trials too) instead
# of being redone per trial or re-derived for refinement.
# Bumped from 100 -- object 1's period was confirmed still drifting (not
# converged) by step 100 via the omega/omega0 metrics.csv columns.
OMEGA_CALIBRATION_STEPS = 1000
OMEGA_CALIBRATION_OMEGA_LR = 1e-6
CALIBRATED_PERIODS_PATH = SCRIPT_DIR / "calibrated_periods.yaml"
# Narrowed from the general-purpose 0.5 default: a 50%-short recording is
# implausible for this data, and PDM's own p_min_frac=0.5 default was what
# let it wander down to object 2's suspiciously low ~423 in the first
# place. 0.85 still allows up to 15% clipping, which should be generous.
OMEGA_CALIBRATION_P_MIN_FRAC = 0.85
# Separate, WIDER bound applied to fit_omega's subsequent gradient drift
# (see train_convex.py's omega_period_min_frac/omega_period_max_frac) --
# PDM itself never proposes a period > T (capped at p_max_frac=1.0), but
# fit_omega's gradient descent has no such limit on its own; this caps how
# far it's allowed to push the period past what PDM even considered.
# Low bound matches OMEGA_CALIBRATION_P_MIN_FRAC (never below 0.85xT);
# high bound is looser (1.0x from PDM's own search) but still tight
# (1.05x, not unbounded) since a large upward correction is exactly the
# failure mode under suspicion for near-symmetric objects.
OMEGA_CALIBRATION_P_MAX_FRAC_DRIFT = 1.05

sys.path.insert(0, str(SCRIPT_DIR))
from validation import compute_combined_metric  # noqa: E402


# ---------------------------------------------------------------------
# subprocess-based training + polling/pruning
# ---------------------------------------------------------------------

def _write_config(cfg: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)


def _latest_recon_stl(run_dir: Path) -> Optional[Path]:
    val_dir = run_dir / "validation"
    if not val_dir.exists():
        return None
    stls = sorted(val_dir.glob("recon_step*.stl"))
    return stls[-1] if stls else None


def _find_run_dir(stdout_lines: list[str]) -> Optional[Path]:
    for line in stdout_lines:
        if line.startswith("run folder: "):
            return Path(line[len("run folder: "):].strip())
    return None


def run_training_subprocess(config_path: Path, entry_script: str, project_root: Path,
                             gt_path: str, trial: Optional[optuna.Trial] = None,
                             report_offset: int = 0,
                             ) -> dict:
    """Launches `python <entry_script> <config_path>` as a subprocess (cwd=
    project_root, matching how you already run these by hand), polls its
    run_dir for progress, optionally reports intermediate combined scores
    to `trial` for Optuna pruning (killing the subprocess if pruned), and
    returns a result dict once it's done (or was pruned/timed out/crashed
    -- this always returns rather than raising, except for the actual
    optuna.TrialPruned signal, so one bad object doesn't crash the whole
    multi-day search).

    `report_offset` lets phase 2 (3 objects x many steps) report into the
    SAME trial on a single increasing step axis across all 3 objects,
    since Optuna's pruner compares intermediate values at the same step
    across trials -- without an offset, object 2's step 20 would be
    compared against object 1's step 20 from other trials, which isn't
    a meaningful comparison.
    """
    cmd = [sys.executable, str(project_root / entry_script), str(config_path)]
    proc = subprocess.Popen(cmd, cwd=str(project_root), stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)

    stdout_lines: list[str] = []
    run_dir: Optional[Path] = None
    start = time.time()
    pruned = False
    crashed = False

    try:
        while True:
            ret = proc.poll()
            # drain whatever's available without blocking the poll loop
            line = proc.stdout.readline() if proc.stdout else ""
            if line:
                stdout_lines.append(line.rstrip("\n"))
                if run_dir is None:
                    run_dir = _find_run_dir(stdout_lines)

            if ret is not None:
                if ret != 0:
                    crashed = True
                break

            if time.time() - start > SUBPROCESS_TIMEOUT_SEC:
                proc.kill()
                crashed = True
                break

            if trial is not None and run_dir is not None:
                latest = _latest_recon_stl(run_dir)
                if latest is not None:
                    try:
                        score = compute_combined_metric(str(latest), gt_path, VAL_CFG)
                        step = int(latest.stem.split("recon_step")[-1])
                        trial.report(score, report_offset + step)
                        if trial.should_prune():
                            proc.kill()
                            pruned = True
                            break
                    except Exception:
                        pass  # a transient bad mesh mid-training shouldn't kill the search

            time.sleep(POLL_INTERVAL_SEC)
    finally:
        if proc.poll() is None:
            proc.kill()

    if pruned:
        raise optuna.TrialPruned()

    result = {"run_dir": run_dir, "crashed": crashed, "final_score": None,
              "best_val_score": None, "stdout_tail": "\n".join(stdout_lines[-30:])}
    if crashed or run_dir is None:
        result["final_score"] = 0.0
        result["best_val_score"] = 0.0
        return result

    for name in ("final_mesh", "best_val_mesh"):
        stl = run_dir / f"{name}.stl"
        key = "final_score" if name == "final_mesh" else "best_val_score"
        if stl.exists():
            try:
                result[key] = compute_combined_metric(str(stl), gt_path, VAL_CFG)
            except Exception:
                result[key] = 0.0
        else:
            result[key] = 0.0
    return result


def _cleanup_run_dir(run_dir: Optional[Path]):
    """Search runs generate a LOT of directories over a multi-day search
    (trials x objects x phases) -- the validation/ subfolder (repeated
    recon_stepXXXXXX.stl snapshots, needed only for this trial's live
    pruning) is the bulk of that disk use. Delete it once the trial is
    scored; keep final_mesh.stl / best_val_mesh.stl / checkpoint.pt /
    metrics.csv / config.yaml (small, and checkpoint.pt is needed by
    phase 2 for whichever trial ends up best)."""
    if run_dir is None:
        return
    val_dir = run_dir / "validation"
    if val_dir.exists():
        shutil.rmtree(val_dir, ignore_errors=True)


# ---------------------------------------------------------------------
# phase 1: convex stage
# ---------------------------------------------------------------------

def _suggest_generator(trial: optuna.Trial, prefix: str) -> tuple[str, dict]:
    kind = trial.suggest_categorical(f"{prefix}_generator", ["points", "shared_mlp"])
    kwargs = {}
    if kind == "shared_mlp":
        kwargs = {
            "noise_dim": trial.suggest_categorical(f"{prefix}_noise_dim", [4, 8, 16]),
            "hidden": trial.suggest_categorical(f"{prefix}_hidden", [32, 64, 128]),
            "n_layers": trial.suggest_categorical(f"{prefix}_n_layers", [2, 3, 4]),
        }
    return kind, kwargs


def _suggest_lr_schedule(trial: optuna.Trial, prefix: str, steps: int) -> dict:
    use_decay = trial.suggest_categorical(f"{prefix}_use_lr_decay", [True, False])
    tol_low = min(20, max(1, steps // 4))
    tol_high = max(tol_low, min(150, steps // 2))
    tr = {"tol": trial.suggest_int(f"{prefix}_tol", tol_low, tol_high)}
    if use_decay:
        # Fixed, not searched -- see FIXED_LR_DECAY_FACTOR/FIXED_LR_MAX_DECAYS.
        tr["lr_decay_factor"] = FIXED_LR_DECAY_FACTOR
        tr["lr_max_decays"] = FIXED_LR_MAX_DECAYS
    return tr


def _blender_available(obj: dict, project_root: Path) -> bool:
    return (project_root / obj["intensity_file_blender"]).exists() and \
           (project_root / obj["binary_file_blender"]).exists()


def build_convex_config(trial: optuna.Trial, obj: dict, run_name: str,
                         project_root: Path, period: float) -> dict:
    lr = trial.suggest_float("cx_lr", 1e-4, 3e-2, log=True)
    lambda_intensity = trial.suggest_float("cx_lambda_intensity", 1.0, 50.0, log=True)
    lambda_binary = trial.suggest_float("cx_lambda_binary", 0.1, 10.0, log=True)
    generator, gen_kwargs = _suggest_generator(trial, "cx")
    tr = _suggest_lr_schedule(trial, "cx", STEPS_CONVEX)

    # period comes from the one-off calibration (calibrate_period_for_object),
    # not "auto" -- every trial uses the SAME already-good period/omega
    # instead of each one re-running (or worse, re-degenerate-ing, see the
    # p_min_frac=p_max_frac=1.0 this used to have) its own PDM search.
    data_cfg = {"intensity_file": obj["intensity_file"], "binary_file": obj["binary_file"]}
    phys_cfg = {"cylinder_radius": obj["cylinder_radius"], "period": period, "fit_omega": False}
    if _blender_available(obj, project_root):
        data_cfg["intensity_file_blender"] = obj["intensity_file_blender"]
        data_cfg["binary_file_blender"] = obj["binary_file_blender"]
        phys_cfg["blender_loss_weight"] = trial.suggest_float(
            "cx_blender_loss_weight", *BLENDER_LOSS_WEIGHT_RANGE, log=True)

    return {
        "run_name": run_name, "optimizer": "ADAM", "tol": tr["tol"],
        "data": data_cfg,
        "validation": {"enabled": True, "every": "log_every", "gt_path": obj["gt_path"],
                        "voxel_size": VOXEL_SIZE},
        "physical": phys_cfg,
        "model": {"dtype": "torch.float64", "LMAX": LMAX, "nrows": NROWS,
                   "generator": generator, "generator_kwargs": gen_kwargs},
        "training": {"steps": STEPS_CONVEX, "lr": lr, "seed": 0,
                      "lambda_binary_start": lambda_binary, "lambda_binary_end": lambda_binary,
                      "lambda_intensity_start": lambda_intensity, "lambda_intensity_end": lambda_intensity,
                      "log_every": VALIDATION_EVERY,
                      **({"lr_decay_factor": tr["lr_decay_factor"], "lr_max_decays": tr["lr_max_decays"]}
                         if "lr_decay_factor" in tr else {})},
        "output": {"base_dir": "./runs"},
    }


def convex_objective(trial: optuna.Trial, project_root: Path, scratch_dir: Path,
                      periods: dict) -> float:
    per_object_final, per_object_best, run_dirs = [], [], {}
    for obj in OBJECTS:
        run_name = f"search_cx_t{trial.number}_obj{obj['num']}"
        cfg = build_convex_config(trial, obj, run_name, project_root, periods[obj["num"]])
        cfg_path = scratch_dir / f"{run_name}.yaml"
        _write_config(cfg, cfg_path)

        result = run_training_subprocess(cfg_path, "main_convex.py", project_root,
                                          obj["gt_path"], trial=trial,
                                          report_offset=obj["num"] * 100000)
        per_object_final.append(result["final_score"])
        per_object_best.append(result["best_val_score"])
        run_dirs[obj["num"]] = str(result["run_dir"]) if result["run_dir"] else None
        _cleanup_run_dir(result["run_dir"])

    trial.set_user_attr("run_dirs", run_dirs)
    trial.set_user_attr("per_object_final", per_object_final)
    trial.set_user_attr("per_object_best_val", per_object_best)
    return sum(per_object_final) / len(per_object_final)


# ---------------------------------------------------------------------
# phase 2: non-convex refinement, warm-started from phase 1's winner
# ---------------------------------------------------------------------

def build_refine_config(trial: optuna.Trial, obj: dict, run_name: str, checkpoint: str,
                         project_root: Path, period: float) -> dict:
    lr = trial.suggest_float("rf_lr", 1e-6, 5e-3, log=True)
    lambda_intensity = trial.suggest_float("rf_lambda_intensity", 1.0, 50.0, log=True)
    lambda_binary = trial.suggest_float("rf_lambda_binary", 0.1, 10.0, log=True)
    generator, gen_kwargs = _suggest_generator(trial, "rf")
    tr = _suggest_lr_schedule(trial, "rf", STEPS_REFINE)

    # Same calibrated period as the convex phase used -- NOT redone here,
    # see calibrate_period_for_object's docstring.
    data_cfg = {"intensity_file": obj["intensity_file"], "binary_file": obj["binary_file"]}
    phys_cfg = {"cylinder_radius": obj["cylinder_radius"], "period": period, "fit_omega": False}
    if _blender_available(obj, project_root):
        data_cfg["intensity_file_blender"] = obj["intensity_file_blender"]
        data_cfg["binary_file_blender"] = obj["binary_file_blender"]
        phys_cfg["blender_loss_weight"] = trial.suggest_float(
            "rf_blender_loss_weight", *BLENDER_LOSS_WEIGHT_RANGE, log=True)

    return {
        "run_name": run_name, "optimizer": "ADAM", "tol": tr["tol"],
        "data": data_cfg,
        "validation": {"enabled": True, "every": "log_every", "gt_path": obj["gt_path"],
                        "voxel_size": VOXEL_SIZE},
        "physical": phys_cfg,
        "model": {"dtype": "torch.float64", "LMAX": LMAX, "nrows": NROWS,
                   "hypernet": "warmstart_points", "warm_start_checkpoint": checkpoint,
                   "warm_start_generator": generator, "warm_start_generator_kwargs": gen_kwargs},
        "training": {"steps": STEPS_REFINE, "lr": lr, "seed": 0,
                      "lambda_binary_start": lambda_binary, "lambda_binary_end": lambda_binary,
                      "lambda_intensity_start": lambda_intensity, "lambda_intensity_end": lambda_intensity,
                      "log_every": VALIDATION_EVERY,
                      **({"lr_decay_factor": tr["lr_decay_factor"], "lr_max_decays": tr["lr_max_decays"]}
                         if "lr_decay_factor" in tr else {})},
        "output": {"base_dir": "./runs"},
    }


def refine_objective(trial: optuna.Trial, project_root: Path, scratch_dir: Path,
                      checkpoints: dict, periods: dict) -> float:
    per_object_final, per_object_best = [], []
    for obj in OBJECTS:
        ckpt = checkpoints.get(obj["num"])
        if ckpt is None:
            raise RuntimeError(f"no phase-1 checkpoint recorded for object {obj['num']} -- "
                                f"run --phase convex first")
        run_name = f"search_rf_t{trial.number}_obj{obj['num']}"
        cfg = build_refine_config(trial, obj, run_name, ckpt, project_root, periods[obj["num"]])
        cfg_path = scratch_dir / f"{run_name}.yaml"
        _write_config(cfg, cfg_path)

        result = run_training_subprocess(cfg_path, "main.py", project_root,
                                          obj["gt_path"], trial=trial,
                                          report_offset=obj["num"] * 100000)
        per_object_final.append(result["final_score"])
        per_object_best.append(result["best_val_score"])
        _cleanup_run_dir(result["run_dir"])

    trial.set_user_attr("per_object_final", per_object_final)
    trial.set_user_attr("per_object_best_val", per_object_best)
    return sum(per_object_final) / len(per_object_final)


# ---------------------------------------------------------------------
# one-off omega/period calibration (convex phase only, done once, reused
# by both phases -- see OMEGA_CALIBRATION_STEPS above)
# ---------------------------------------------------------------------

def _read_final_period(checkpoint_path: Path) -> float:
    import math
    import torch
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    return -2 * math.pi / ckpt["omega"].item()


def calibrate_period_for_object(obj: dict, project_root: Path, scratch_dir: Path) -> float:
    """~200 steps, shape frozen (training.lr=0), fit_omega=True, starting
    from a real PDM search (period: auto, p_min_frac/p_max_frac=
    OMEGA_CALIBRATION_P_MIN_FRAC/1.0 -- NOT the degenerate 1.0/1.0 that
    forces period=T, and narrower than the general-purpose 0.5 default,
    see OMEGA_CALIBRATION_P_MIN_FRAC). The subsequent fit_omega drift is
    ALSO bounded (omega_period_min_frac/max_frac, see train_convex.py) --
    PDM's own p_max_frac=1.0 stops IT from proposing a period > T, but
    fit_omega's gradient descent has no such limit by itself, so the
    bound is enforced there separately (and slightly WIDER on the upper
    side, OMEGA_CALIBRATION_P_MAX_FRAC_DRIFT, since PDM's own search
    already caps at 1.0x). Independent of every other hyperparameter
    being searched, so there's no reason to redo it per trial -- run once
    per object here, cached to CALIBRATED_PERIODS_PATH by
    `ensure_calibrated_periods`."""
    run_name = f"search_omega_calib_obj{obj['num']}"
    cfg = {
        "run_name": run_name, "optimizer": "ADAM",
        "tol": OMEGA_CALIBRATION_STEPS + 1,  # never let the plateau path cut this short
        "data": {"intensity_file": obj["intensity_file"], "binary_file": obj["binary_file"]},
        "validation": {"enabled": True, "every": "log_every", "gt_path": obj["gt_path"],
                        "voxel_size": VOXEL_SIZE},
        "physical": {"cylinder_radius": obj["cylinder_radius"], "period": "auto",
                     "period_search": {"p_min_frac": OMEGA_CALIBRATION_P_MIN_FRAC,
                                        "p_max_frac": 1.0, "n_bins": 12},
                     "fit_omega": True,
                     "omega_period_min_frac": OMEGA_CALIBRATION_P_MIN_FRAC,
                     "omega_period_max_frac": OMEGA_CALIBRATION_P_MAX_FRAC_DRIFT},
        "model": {"dtype": "torch.float64", "LMAX": LMAX, "nrows": NROWS,
                   "generator": "points", "generator_kwargs": {}},
        "training": {"steps": OMEGA_CALIBRATION_STEPS, "lr": 0.0, "seed": 0,
                      "omega_lr": OMEGA_CALIBRATION_OMEGA_LR,
                      "lambda_binary_start": 1.0, "lambda_binary_end": 1.0,
                      "lambda_intensity_start": 10.0, "lambda_intensity_end": 10.0,
                      "log_every": 20},
        "output": {"base_dir": "./runs"},
    }
    cfg_path = scratch_dir / f"{run_name}.yaml"
    _write_config(cfg, cfg_path)
    result = run_training_subprocess(cfg_path, "main_convex.py", project_root, obj["gt_path"])
    if result["run_dir"] is None:
        raise RuntimeError(f"omega calibration failed for object {obj['num']}:\n"
                            f"{result['stdout_tail']}")
    period = _read_final_period(result["run_dir"] / "checkpoint.pt")
    _cleanup_run_dir(result["run_dir"])
    print(f"[omega calib] object {obj['num']}: period={period:.4f}")
    return period


def _load_calibrated_periods() -> dict:
    with open(CALIBRATED_PERIODS_PATH) as f:
        raw = yaml.safe_load(f)
    return {int(k): float(v) for k, v in raw.items()}


def ensure_calibrated_periods(project_root: Path, scratch_dir: Path) -> dict:
    """Loads calibrated_periods.yaml if a PRIOR run already produced one
    (so this stays a true one-off across resumed --phase convex runs too),
    otherwise calibrates every object once and writes it."""
    if CALIBRATED_PERIODS_PATH.exists():
        periods = _load_calibrated_periods()
        if all(obj["num"] in periods for obj in OBJECTS):
            print(f"[omega calib] reusing {CALIBRATED_PERIODS_PATH}: {periods}")
            return periods
    periods = {obj["num"]: calibrate_period_for_object(obj, project_root, scratch_dir)
               for obj in OBJECTS}
    with open(CALIBRATED_PERIODS_PATH, "w") as f:
        yaml.safe_dump(periods, f)
    print(f"[omega calib] wrote {CALIBRATED_PERIODS_PATH}: {periods}")
    return periods


# ---------------------------------------------------------------------
# best-checkpoint lookup (phase 1 -> phase 2 bridge)
# ---------------------------------------------------------------------

def best_convex_checkpoints(study: optuna.Study) -> dict:
    """checkpoint.pt path per object, all from the SAME winning trial
    (the shared convex recipe you'd actually deploy), not each object's
    individually-best trial -- consistency with what you'll run on a
    real unknown object matters more than squeezing out per-object gains."""
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError("no completed phase-1 trials in this study -- run --phase convex first")
    best = max(completed, key=lambda t: t.value)
    run_dirs = best.user_attrs["run_dirs"]
    return {int(k): str(Path(v) / "checkpoint.pt") for k, v in run_dirs.items() if v}


# ---------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------

def print_leaderboard(study: optuna.Study, top_n: int = 10):
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda t: t.value, reverse=True)
    print(f"\n{'='*100}\nTop {min(top_n, len(completed))} of {len(completed)} completed trials "
          f"({len(study.trials) - len(completed)} pruned/failed)\n{'='*100}")
    print(f"{'trial':>6} {'combined(final)':>16} {'combined(best_val)':>20}  per-object final")
    for t in completed[:top_n]:
        pf = t.user_attrs.get("per_object_final", [])
        pb = t.user_attrs.get("per_object_best_val", [])
        print(f"{t.number:>6} {t.value:>16.4f} {sum(pb)/max(len(pb),1):>20.4f}  "
              f"{[round(x, 3) for x in pf]}  (best_val: {[round(x, 3) for x in pb]})")
    print()


def dump_leaderboard_csv(study: optuna.Study, path: Path):
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda t: t.value, reverse=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trial", "combined_final_avg", "combined_best_val_avg",
                    "per_object_final", "per_object_best_val", "params"])
        for t in completed:
            pf = t.user_attrs.get("per_object_final", [])
            pb = t.user_attrs.get("per_object_best_val", [])
            w.writerow([t.number, t.value, sum(pb) / max(len(pb), 1), pf, pb, t.params])


def write_best_config_yamls(study: optuna.Study, phase: str, project_root: Path,
                             checkpoints: Optional[dict] = None):
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        return
    best = max(completed, key=lambda t: t.value)
    out_dir = SCRIPT_DIR
    periods = _load_calibrated_periods()
    if phase == "convex":
        for obj in OBJECTS:
            cfg = build_convex_config(_FixedTrial(best.params), obj, f"BEST_convex_obj{obj['num']}",
                                       project_root, periods[obj["num"]])
            _write_config(cfg, out_dir / f"config_convex_BEST_obj{obj['num']}.yaml")
    else:
        for obj in OBJECTS:
            ckpt = checkpoints[obj["num"]]
            cfg = build_refine_config(_FixedTrial(best.params), obj,
                                       f"BEST_refine_obj{obj['num']}", ckpt,
                                       project_root, periods[obj["num"]])
            _write_config(cfg, out_dir / f"config_refine_BEST_obj{obj['num']}.yaml")
    print(f"[{phase}] wrote ready-to-use configs for the winning trial "
          f"(trial #{best.number}, combined score {best.value:.4f}) to {out_dir}")


class _FixedTrial:
    """Replays a completed trial's exact params instead of sampling new
    ones -- lets `build_convex_config`/`build_refine_config` be reused
    unchanged to materialize the winning trial's config to disk."""
    def __init__(self, params: dict):
        self.params = params

    def suggest_float(self, name, low, high, log=False):
        return self.params[name]

    def suggest_int(self, name, low, high):
        return self.params[name]

    def suggest_categorical(self, name, choices):
        return self.params[name]


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["convex", "refine"], required=True)
    ap.add_argument("--hours", type=float, default=20.0,
                     help="wall-clock budget for THIS phase (run the other phase separately)")
    ap.add_argument("--project-root", type=str, default=str(PROJECT_ROOT_DEFAULT))
    ap.add_argument("--storage", type=str, default=str(SCRIPT_DIR / "search.db"))
    ap.add_argument("--study-name", type=str, default=None,
                     help="default: 'convex_search' / 'refine_search'")
    args = ap.parse_args()

    project_root = Path(args.project_root)
    scratch_dir = SCRIPT_DIR / "search_scratch"
    scratch_dir.mkdir(exist_ok=True)
    storage = f"sqlite:///{args.storage}"
    study_name = args.study_name or f"{args.phase}_search"

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=STEPS_CONVEX // 4,
                                          interval_steps=VALIDATION_EVERY)
    study = optuna.create_study(study_name=study_name, storage=storage,
                                 load_if_exists=True, direction="maximize", pruner=pruner)

    print(f"[{args.phase}] study '{study_name}' -- {len(study.trials)} trials already recorded, "
          f"running for up to {args.hours}h more (project_root={project_root})")

    deadline = time.time() + args.hours * 3600

    if args.phase == "convex":
        periods = ensure_calibrated_periods(project_root, scratch_dir)

        def objective(trial):
            return convex_objective(trial, project_root, scratch_dir, periods)
    else:
        checkpoints = best_convex_checkpoints(
            optuna.load_study(study_name="convex_search", storage=storage))
        print(f"[refine] warm-starting from convex_search's winning trial's checkpoints: {checkpoints}")
        if not CALIBRATED_PERIODS_PATH.exists():
            raise RuntimeError(f"{CALIBRATED_PERIODS_PATH} not found -- run --phase convex first "
                                f"(it performs the one-off omega calibration this phase reuses).")
        periods = _load_calibrated_periods()
        print(f"[refine] reusing calibrated periods from --phase convex: {periods}")

        def objective(trial):
            return refine_objective(trial, project_root, scratch_dir, checkpoints, periods)

    csv_path = SCRIPT_DIR / f"search_leaderboard_{args.phase}.csv"

    def _checkpoint_progress():
        # Called after every trial, not just once at the end -- so you can
        # watch progress (`watch cat search_leaderboard_*.csv`, or just
        # re-open the file) DURING a multi-hour/overnight run instead of
        # only finding out how it went after the whole --hours budget (or
        # an interrupt) is used up.
        dump_leaderboard_csv(study, csv_path)
        if args.phase == "convex":
            write_best_config_yamls(study, "convex", project_root)
        else:
            write_best_config_yamls(study, "refine", project_root, checkpoints)

    try:
        while time.time() < deadline:
            remaining = deadline - time.time()
            if remaining < 60:
                break
            study.optimize(objective, n_trials=1, timeout=remaining, catch=(Exception,))
            _checkpoint_progress()
    except KeyboardInterrupt:
        print("\ninterrupted -- progress is saved, re-run the same command to resume.")

    print_leaderboard(study)
    _checkpoint_progress()


if __name__ == "__main__":
    main()
