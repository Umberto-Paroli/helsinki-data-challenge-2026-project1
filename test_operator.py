"""
test_operator.py
=================

Calibration script: given a mesh (a reconstruction or a public GT STL),
compare the forward operator's simulated lightcurves against BOTH the
real-lab lightcurves and the Blender-simulated ones, across a sweep of
`camera_distance` values, to see whether the camera_distance=30.0 you
found empirically for real data also explains the Blender curves, or
whether it needs its own value.

Rewritten from your original `test_operator.py`: same idea (RMSE +
correlation between simulated and observed curves, resampled onto the
observed timestamps), but ported onto the current consolidated
`geometry.py` (the old `forward_challenge_cristiano`/`operator_extensions`/
`save_lightcurves` modules it used are the ones now archived under
legacy/ -- everything they did is in `geometry.py`/`train.py` now) and
extended with:
  - a camera_distance SWEEP instead of one hardcoded value, run
    separately against real and Blender data
  - Blender-curve loading, with the period derived directly from the
    data (it's exactly one clean rotation by construction -- no PDM
    search needed there, unlike the real recordings)

Usage: edit CANDIDATE_STL / ASTEROID_NUM / DATA_DIR below to point at
your files, then:
    python ./test_operator.py
Prints a table per data source (real, blender) x camera_distance, plus
the usual comparison plot.
"""
import math
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from geometry import (
    load_stl_mesh, make_challenge_cameras_28, challenge_column_labels,
    light_direction_tensor, simulate_all_curves, ChallengeForward,
)
from train import estimate_period_from_data

DEVICE = "cpu"
torch.set_default_dtype(torch.float64)


# ---------------------------------------------------------------------
# config -- edit these
# ---------------------------------------------------------------------

ASTEROID_NUM = 3
DATA_DIR = Path.cwd() / "Data"

CANDIDATE_STL = [
    DATA_DIR / f"AsteroidModel0{ASTEROID_NUM}_shape_public/" / f"asteroid{ASTEROID_NUM}.stl",
    # or point this at one of your own reconstruction runs, e.g.:
    # Path("./runs/<some_run>/final_mesh.stl"),
]

CAMERA_DISTANCE_CANDIDATES = [None, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 100.0]
# None -> plain orthographic (no perspective mu correction); everything
# else -> finite-distance perspective at that many body-radii away.
SCATTERING = "mixed"
LS_WEIGHT = 0.03
N_EPOCHS_SIM = 360   # dense uniform grid over one rotation, resampled onto observed timestamps


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------

def _load_challenge_lc(path: Path):
    """Challenge lightcurve files are comma-delimited; skip a header row
    if there is a non-numeric one."""
    try:
        return np.loadtxt(path, delimiter=",")
    except ValueError:
        return np.loadtxt(path, delimiter=",", skiprows=1)


def resample_at_obs(sim_curves: np.ndarray, t_obs: np.ndarray, period: float) -> np.ndarray:
    """sim_curves (nC, N) over exactly one rotation -> values at each
    observed timestamp, via linear interpolation in phase."""
    N = sim_curves.shape[1]
    phase = ((t_obs - t_obs.min()) / period) % 1.0
    ii = phase * N
    i0 = np.floor(ii).astype(int) % N
    w = ii - np.floor(ii)
    i1 = (i0 + 1) % N
    return sim_curves[:, i0] * (1 - w) + sim_curves[:, i1] * w


def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def _normalize_columns(x: np.ndarray) -> np.ndarray:
    """Matches the challenge's own convention and train.py's
    normalize_columns: each curve divided by its own mean."""
    means = x.mean(axis=1, keepdims=True)
    means[means == 0] = 1.0
    return x / means


def load_lightcurve_pair(intensity_path: Path, binary_path: Path):
    data_i = _load_challenge_lc(intensity_path)
    data_b = _load_challenge_lc(binary_path)
    t_obs = data_i[:, 0]
    assert np.allclose(t_obs, data_b[:, 0]), \
        f"{intensity_path.name} and {binary_path.name} have different timestamps"
    obs_intensity = _normalize_columns(data_i[:, 1:].T)   # (nC, N)
    obs_binary = _normalize_columns(data_b[:, 1:].T)
    return t_obs, obs_intensity, obs_binary


def blender_period_from_data(t_obs: np.ndarray) -> float:
    """Blender curves are exactly one clean rotation, evenly sampled --
    unlike real recordings there's no PDM search needed, the period is
    directly implied by the timestamp spacing: with N samples spanning
    one full period, the observed span (first to last sample) is
    (N-1)/N * P."""
    n = len(t_obs)
    spacing = (t_obs[-1] - t_obs[0]) / (n - 1)
    return n * spacing


def sweep_camera_distance(tlist, vlist, E_cams, E0, t_obs, obs_intensity, obs_binary,
                           period: float, label: str):
    TIME_sim = torch.linspace(0.0, period, N_EPOCHS_SIM + 1, dtype=torch.float64)[:-1]
    omega = torch.tensor(-2.0 * math.pi / period, dtype=torch.float64)
    omega0 = torch.tensor(0.0, dtype=torch.float64)

    print(f"\n=== {label}: sweeping camera_distance (period={period:.3f}) ===")
    print(f"{'camera_distance':>16}  {'RMSE_I':>8}  {'RMSE_B':>8}  {'corr_I':>8}  {'corr_B':>8}")
    results = []
    for cd in CAMERA_DISTANCE_CANDIDATES:
        with torch.no_grad():
            curves = simulate_all_curves(
                tlist, vlist, E_cams, TIME_sim, omega=omega, omega0=omega0,
                E0_world=E0, mode="both", rel=True,
                camera_distance=cd, scattering=SCATTERING, ls_weight=LS_WEIGHT,
            )
        I_sim = resample_at_obs(curves[0].cpu().numpy(), t_obs, period)
        B_sim = resample_at_obs(curves[1].cpu().numpy(), t_obs, period)
        I_sim = _normalize_columns(I_sim)
        B_sim = _normalize_columns(B_sim)

        rmse_I = np.mean([_rmse(obs_intensity[k], I_sim[k]) for k in range(obs_intensity.shape[0])])
        rmse_B = np.mean([_rmse(obs_binary[k], B_sim[k]) for k in range(obs_binary.shape[0])])
        corr_I = np.mean([_corr(obs_intensity[k], I_sim[k]) for k in range(obs_intensity.shape[0])])
        corr_B = np.mean([_corr(obs_binary[k], B_sim[k]) for k in range(obs_binary.shape[0])])
        cd_label = "ortho" if cd is None else f"{cd:g}"
        print(f"{cd_label:>16}  {rmse_I:>8.4f}  {rmse_B:>8.4f}  {corr_I:>8.3f}  {corr_B:>8.3f}")
        results.append({"camera_distance": cd, "rmse_I": rmse_I, "rmse_B": rmse_B,
                         "corr_I": corr_I, "corr_B": corr_B})

    best = min(results, key=lambda r: r["rmse_I"] + r["rmse_B"])
    print(f"-> best by RMSE_I+RMSE_B: camera_distance="
          f"{'ortho' if best['camera_distance'] is None else best['camera_distance']}")
    return results


def plot_best_fit(tlist, vlist, E_cams, E0, t_obs, obs_intensity, obs_binary,
                   period: float, camera_distance, labels, title, savepath):
    TIME_sim = torch.linspace(0.0, period, N_EPOCHS_SIM + 1, dtype=torch.float64)[:-1]
    omega = torch.tensor(-2.0 * math.pi / period, dtype=torch.float64)
    omega0 = torch.tensor(0.0, dtype=torch.float64)
    with torch.no_grad():
        curves = simulate_all_curves(
            tlist, vlist, E_cams, TIME_sim, omega=omega, omega0=omega0,
            E0_world=E0, mode="both", rel=True,
            camera_distance=camera_distance, scattering=SCATTERING, ls_weight=LS_WEIGHT,
        )
    I_sim = _normalize_columns(resample_at_obs(curves[0].cpu().numpy(), t_obs, period))
    B_sim = _normalize_columns(resample_at_obs(curves[1].cpu().numpy(), t_obs, period))

    nC = obs_intensity.shape[0]
    fig, axes = plt.subplots(nC, 2, figsize=(11, 1.3 * nC), sharex=True)
    for k in range(nC):
        axes[k, 0].plot(t_obs, obs_intensity[k], "k-", lw=1.0, label="obs")
        axes[k, 0].plot(t_obs, I_sim[k], "b--", lw=1.0, label="sim")
        axes[k, 0].set_ylabel(labels[k], rotation=0, ha="right", va="center", fontsize=7)
        axes[k, 0].grid(True, alpha=0.3)
        axes[k, 1].plot(t_obs, obs_binary[k], color="gray", lw=1.0, label="obs")
        axes[k, 1].plot(t_obs, B_sim[k], color="tab:orange", ls="--", lw=1.0, label="sim")
        axes[k, 1].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8, loc="upper right")
    axes[0, 0].set_title(f"{title} -- Intensity")
    axes[0, 1].set_title(f"{title} -- Binary")
    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    print(f"saved {savepath}")


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

def main():
    stl_path = next((p for p in CANDIDATE_STL if p.exists()), None)
    if stl_path is None:
        raise FileNotFoundError(
            f"none of CANDIDATE_STL exist: {[str(p) for p in CANDIDATE_STL]} -- "
            f"edit the list at the top of this file")
    print(f"[STL] loading {stl_path}")
    tlist, vlist = load_stl_mesh(str(stl_path), normalize=True, max_faces=2000, device=DEVICE)
    print(f"mesh: nfac={tlist.shape[0]}, nvert={vlist.shape[0]}")

    E_cams = make_challenge_cameras_28()
    E0 = light_direction_tensor()
    labels = challenge_column_labels()

    n = ASTEROID_NUM
    real_intensity = DATA_DIR / f"AsteroidModel0{n}_shape_public" / f"Asteroid{n}_lightcurve_data" / f"Asteroid0{n}_lightcurve_intensity.txt"
    real_binary = DATA_DIR / f"AsteroidModel0{n}_shape_public" / f"Asteroid{n}_lightcurve_data" / f"Asteroid0{n}_lightcurve_binary.txt"
    blender_intensity = DATA_DIR / f"AsteroidModel0{n}_shape_public" / f"Asteroid{n}_lightcurve_data" / f"Asteroid0{n}_lightcurve_intensity_blender.txt"
    blender_binary = DATA_DIR / f"AsteroidModel0{n}_shape_public" / f"Asteroid{n}_lightcurve_data" / f"Asteroid0{n}_lightcurve_binary_blender.txt"

    if real_intensity.exists() and real_binary.exists():
        t_real, I_real, B_real = load_lightcurve_pair(real_intensity, real_binary)
        period_real, reliable = estimate_period_from_data(t_real, I_real.T, {})
        print(f"[real] period estimate: {period_real:.3f} frames (reliable={reliable})")
        real_results = sweep_camera_distance(tlist, vlist, E_cams, E0, t_real, I_real, B_real,
                                              period_real, label="REAL")
        best_real = min(real_results, key=lambda r: r["rmse_I"] + r["rmse_B"])
        plot_best_fit(tlist, vlist, E_cams, E0, t_real, I_real, B_real, period_real,
                      best_real["camera_distance"], labels, f"asteroid{n} REAL", f"fit_real_asteroid{n}.png")
    else:
        print(f"[real] no lightcurve files found at {real_intensity} -- skipping")

    if blender_intensity.exists() and blender_binary.exists():
        t_bl, I_bl, B_bl = load_lightcurve_pair(blender_intensity, blender_binary)
        period_bl = blender_period_from_data(t_bl)
        print(f"[blender] period from data (exact, no search needed): {period_bl:.3f}")
        bl_results = sweep_camera_distance(tlist, vlist, E_cams, E0, t_bl, I_bl, B_bl,
                                            period_bl, label="BLENDER")
        best_bl = min(bl_results, key=lambda r: r["rmse_I"] + r["rmse_B"])
        plot_best_fit(tlist, vlist, E_cams, E0, t_bl, I_bl, B_bl, period_bl,
                      best_bl["camera_distance"], labels, f"asteroid{n} BLENDER", f"fit_blender_asteroid{n}.png")
    else:
        print(f"[blender] no lightcurve files found at {blender_intensity} -- skipping "
              f"(check the _blender.txt naming, or that this asteroid's Blender data is published)")


if __name__ == "__main__":
    main()
