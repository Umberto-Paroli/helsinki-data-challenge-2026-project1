"""
train.py
========

ADAM training loop for lightcurve-inversion shape reconstruction.
Consolidated from the former `train_ADAM.py` + `period_search.py` (period
search is only ever called from `estimate_period_from_data` below, at the
very start of a run, so it lives here instead of its own module).

Two efficiency fixes applied vs. the original (approved -- same results,
less redundant work; see chat):

1. `model` (`ChallengeForward`) is never trained -- only `hyper.parameters()`
   is ever passed to the optimizer, `model`'s buffers (cameras, octantoid
   grid, light direction) are fixed for the whole run. The old code did
   `copy.deepcopy(model)` every time a validation step improved on the
   best score, alongside `copy.deepcopy(hyper)`, even though `model` is
   always identical to the live one. `best_val` now only snapshots `hyper`;
   `model` is reused directly wherever the "best" mesh is needed.

2. `hyper` used to be built TWICE: once with a placeholder r0 just to read
   its `.directions` grid for the ellipsoid warm-start estimate, then
   thrown away and rebuilt with the real r0. `.directions` only depends on
   (LMAX, nrows, dtype, device) -- not on r0 or on any trainable weight --
   so it's now read via `geometry.octantoid_directions(...)` directly,
   without constructing (and discarding) a full hypernet instance first.
"""
from __future__ import annotations
import csv
import shutil
from datetime import datetime
from pathlib import Path

import copy
import numpy as np
import torch

from geometry import ChallengeForward, make_challenge_cameras_28, challenge_column_labels, \
    simulate_all_curves, octantoid_directions, convex_hull_mesh, facets_over_horizon
from hypernets import (
    CoefficientHyperNet, SphereCoordHyperNet, GraphConvHyperNet,
    SphereNoiseGraphHyperNet, PlanarDiskHyperNet, ConvexPointCloud,
    ConvexOffsetNet, estimate_ellipsoid_r0,
)
from validation import validate_and_log, export_reconstruction, fit_z_range, soft_fit_z_range, \
    radial_clamp, soft_radial_clamp, omega_clamp, soft_omega_clamp


def make_run_dir(cfg: dict, config_path: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg["output"]["base_dir"]) / f"{cfg['run_name']}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy(config_path, run_dir / "config.yaml")  # exact config used, for the record
    return run_dir


# ---------------------------------------------------------------------
# hypernetwork factory
# ---------------------------------------------------------------------

def build_hypernet(cfg: dict, dtype=torch.float64, r0=None):
    m = cfg["model"]
    kwargs = dict(m.get("hypernet_kwargs", {}))
    if r0 is None:
        r0 = cfg["physical"]["cylinder_radius"]

    if m["hypernet"] == "flat":
        return CoefficientHyperNet(m["LMAX"], r0=r0, dtype=dtype,
                                    **{k: v for k, v in kwargs.items()
                                       if k in ("latent_dim", "hidden")})
    if m["hypernet"] == "sphere":
        return SphereCoordHyperNet(m["LMAX"], m["nrows"], r0=r0, dtype=dtype,
                                    **{k: v for k, v in kwargs.items()
                                       if k in ("hidden", "hidden_layers", "omega_0")})
    if m["hypernet"] == "graph":
        return GraphConvHyperNet(m["LMAX"], m["nrows"], r0=r0, dtype=dtype,
                                  **{k: v for k, v in kwargs.items()
                                     if k in ("hidden", "n_layers")})
    if m["hypernet"] == "variant_D1":
        return SphereNoiseGraphHyperNet(m["LMAX"], m["nrows"], r0=r0, dtype=dtype,
                                         **{k: v for k, v in kwargs.items()
                                            if k in ("hidden", "n_layers", "noise_dim", "k_hop")})
    if m["hypernet"] == "warmstart_points":
        # Continue from a finished train_convex.py run: load the ACTUAL
        # point positions it ended on (not a radius/ellipsoid reduction of
        # them -- see chat), and hand them to ConvexPointCloud as a direct
        # initial-position tensor (r0=1.0 trick, see its docstring).
        #
        # Topology: NOT `model`'s fixed octantoid triangulation (model.tlist)
        # -- that assumes point i stayed near its original grid direction,
        # which a freely-moved convex-stage point cloud doesn't respect (see
        # ConvexPointCloud's docstring). Instead, take the convex hull of
        # THESE points once, right now, and use that as the fixed topology
        # from here on: hull vertices get a triangulation that reflects
        # where they actually are, and points that never made the stage-1
        # hull just aren't referenced by any face (instead of being forced
        # into a wildly-stretched, near-degenerate one connecting them to
        # faraway surface points -- which is what was blowing up the
        # self-shadowing test's memory, ~100x+ candidate inflation, causing
        # CUDA OOM even at small nrows; see chat). Rendering still uses the
        # default assume_convex=False (full self-shadowing), so the shape is
        # free to develop concavities from here on -- only the CONNECTIVITY
        # is fixed at its warm-start value, same as every other
        # representation in this file, not the convexity.
        ckpt_path = m.get("warm_start_checkpoint")
        if not ckpt_path:
            raise ValueError(
                "model.hypernet == 'warmstart_points' requires "
                "model.warm_start_checkpoint: <path to a train_convex.py "
                "run's checkpoint.pt>"
            )
        ckpt = torch.load(ckpt_path, map_location="cpu")
        positions = ckpt["points"].to(dtype=dtype)
        expected_K = 4 * m["nrows"] ** 2 + 2
        if positions.shape[0] != expected_K:
            raise ValueError(
                f"warm_start_checkpoint has {positions.shape[0]} points but "
                f"model.nrows={m['nrows']} expects {expected_K} -- the convex "
                f"stage's config.yaml and this config must use the same nrows "
                f"(LMAX doesn't need to match: the convex stage doesn't fit "
                f"any SH coefficients, LMAX there only sized the direction grid; "
                f"nrows itself no longer drives the non-convex stage's topology "
                f"either now, but mismatched nrows here is still almost always "
                f"a sign the wrong checkpoint was pointed at)."
            )
        warmstart_tlist, _ = convex_hull_mesh(positions)
        n_on_hull = warmstart_tlist.unique().numel()
        n_dead = positions.shape[0] - n_on_hull
        print(f"[warm start] convex hull of {positions.shape[0]} checkpoint points -> "
              f"{n_on_hull} on the hull ({warmstart_tlist.shape[0]} faces) used as the "
              f"fixed topology for non-convex refinement; the other {n_dead} stay in "
              f"the parameter tensor as currently-unreferenced points (same as any "
              f"convex-stage 'dead' point).")

        # warm_start_generator: "points" (default) -> ConvexPointCloud, fully
        # independent per-point parameters, same as before. "shared_mlp" ->
        # ConvexOffsetNet: a small MLP shared across all points predicts each
        # one's offset FROM its loaded warm-start position (small final-layer
        # init -> starts ~exactly at the checkpoint, same as train_convex.py's
        # shared_mlp). Matters more here than in stage 1: with fixed_tlist now
        # locking the topology once and for all (the fix above), the `n_dead`
        # points above get EXACTLY zero gradient for the rest of this run under
        # ConvexPointCloud -- permanently frozen wherever the checkpoint left
        # them. Shared weights don't have that failure mode: every point's
        # predicted offset runs through the same weights, so gradient from the
        # on-hull points still updates those weights every step, which nudges
        # every point's predicted offset including currently-dead ones -- a
        # point can become geometrically relevant again later in training as
        # the (now non-convex) shape develops, instead of being stuck.
        gen_kind = m.get("warm_start_generator", "points")
        if gen_kind == "points":
            return ConvexPointCloud(positions, r0=1.0, dtype=dtype, fixed_tlist=warmstart_tlist)
        if gen_kind == "shared_mlp":
            kwargs = dict(m.get("warm_start_generator_kwargs", {}))
            return ConvexOffsetNet(positions, dtype=dtype, fixed_tlist=warmstart_tlist,
                                    **{k: v for k, v in kwargs.items()
                                       if k in ("noise_dim", "hidden", "n_layers")})
        raise ValueError(f"unknown model.warm_start_generator: {gen_kind!r} "
                          f"(expected 'points' or 'shared_mlp')")
    raise ValueError(f"unknown model.hypernet: {m['hypernet']!r}")


def get_mesh(hyper, model: ChallengeForward, beta=30.0, training=True, cylinder_radius=None):
    """Dispatch on hypernet type: flat/sphere return `a` (-> model.mesh(a)),
    graph returns (tlist, vlist) directly, warmstart_points (ConvexPointCloud/
    ConvexOffsetNet) returns free (K,3) positions paired with EITHER
    `hyper.fixed_tlist` (if set -- the convex hull of the checkpoint's own
    points, computed once at warm-start time, see build_hypernet) or, only
    if that's unset, `model`'s fixed octantoid triangulation (model.tlist).
    The fixed_tlist case is the one that actually matters in practice: it's
    what every warmstart_points run produces (build_hypernet always passes
    it), and using it instead of model.tlist is what avoids pairing
    interior/"dead" points with grid-index-based connectivity that assumes
    they're still near their original direction (see ConvexPointCloud's
    docstring -- that mismatch was inflating the self-shadowing test's
    memory use by 100x+ and causing CUDA OOM). No convexity constraint is
    enforced here either way (no hull RECOMPUTE step, unlike train_convex.py)
    -- self-shadowing (assume_convex=False, the default) applies and the
    shape is free to develop concavities. Either way, the result is passed
    through fit_z_range: mean-normalized lightcurve losses are exactly
    scale-invariant, so without this, absolute size is unconstrained by the
    data and can drift arbitrarily. Locking it to the known cylinder height
    on every forward pass means all gradient signal goes toward shape.

    If `cylinder_radius` is given, the xy-radius is likewise capped right
    after the z-fit (soft_radial_clamp/radial_clamp, matching training's
    soft/hard switch) -- without this, nothing bounds xy at all, and it
    was previously free to drift to several times cylinder_radius (see
    chat: recon_step000335.stl had xy up to 5.57 vs cylinder_radius=1.42,
    also the direct cause of frequent trimesh 'max_iter exceeded!' errors
    downstream). Applied AFTER fit_z_range since that rescale changes xy
    too (isotropic scale s), so clamping first could get undone."""
    model_device = model.E0_world.device
    if isinstance(hyper, GraphConvHyperNet) or isinstance(hyper, SphereNoiseGraphHyperNet):
        tlist, vlist = hyper()
        tlist = tlist.to(model_device)
        vlist = vlist.to(model_device)
    elif isinstance(hyper, (ConvexPointCloud, ConvexOffsetNet)):
        vlist = hyper().to(model_device)
        fixed_tlist = getattr(hyper, "fixed_tlist", None)
        tlist = fixed_tlist.to(model_device) if fixed_tlist is not None else model.tlist
    else:
        a = hyper()
        a = a.to(model_device)
        tlist, vlist = model.mesh(a)
    vlist = soft_fit_z_range(vlist, beta=beta) if training else fit_z_range(vlist)
    if cylinder_radius is not None:
        vlist = soft_radial_clamp(vlist, cylinder_radius, beta=beta) if training \
            else radial_clamp(vlist, cylinder_radius)
    return tlist, vlist


# ---------------------------------------------------------------------
# data
# ---------------------------------------------------------------------

def load_lightcurve_28(path: str):
    raw = np.loadtxt(path, delimiter=",")
    assert raw.shape[1] == 29, f"expected 29 columns, got {raw.shape[1]} in {path}"
    return raw[:, 0], raw[:, 1:]


def normalize_columns(x: torch.Tensor) -> torch.Tensor:
    return x / x.mean(dim=0, keepdim=True).clamp(min=1e-12)


def curriculum(step, total, w_start, w_end):
    f = min(1.0, step / max(1, total))
    return w_start + f * (w_end - w_start)


# ---------------------------------------------------------------------
# period search (Phase Dispersion Minimization, Stellingwerf 1978)
# ---------------------------------------------------------------------
#
# Appropriate specifically because the recording extends somewhat beyond
# one rotation by an amount that isn't a clean single-frame difference --
# so the true period must be found from the data's actual periodicity, not
# inferred from the total recorded time span directly. Searches trial
# periods up to the full recorded span T (an upper bound on the period),
# down to `p_min_frac * T`.

def pdm_theta(t: np.ndarray, L: np.ndarray, period: float, n_bins: int = 12) -> float:
    """
    t: (N,) timestamps
    L: (N, C) one or more channels, already roughly comparable in scale.
    Returns the combined Stellingwerf theta statistic for this trial period:
    ~1 for a wrong period, drops toward 0 near the correct one.
    """
    phase = (t / period) % 1.0
    bin_idx = np.clip((phase * n_bins).astype(int), 0, n_bins - 1)

    N, C = L.shape
    ss_within = 0.0
    n_within = 0
    for b in range(n_bins):
        mask = bin_idx == b
        n_b = mask.sum()
        if n_b > 1:
            seg = L[mask]  # (n_b, C)
            ss_within += ((seg - seg.mean(axis=0, keepdims=True)) ** 2).sum()
            n_within += n_b - 1

    ss_total = ((L - L.mean(axis=0, keepdims=True)) ** 2).sum()
    if n_within == 0 or ss_total <= 0:
        return 1.0
    return (ss_within / n_within) / (ss_total / (N - 1))


def search_period_pdm(t: np.ndarray, L: np.ndarray,
                       p_min_frac: float = 0.5, p_max_frac: float = 1.0,
                       n_coarse: int = 400, n_bins: int = 12,
                       refine: bool = True, n_perm: int = 100,
                       rng_seed: int = 0) -> dict:
    """
    Searches trial periods in [p_min_frac, p_max_frac] * T, T = t[-1]-t[0].

    Two diagnostics guard against the "recording shorter than true period"
    failure mode (theta decreases monotonically to the edge of the search
    range and gets picked with a deceptively good score):
      hit_boundary : best period lands within ~3% of either edge
      significant  : permutation test against `n_perm` row-shuffles of L
    `reliable = (not hit_boundary) and (not monotonic_to_edge) and significant`.
    """
    T = t[-1] - t[0]
    periods = np.linspace(p_min_frac * T, p_max_frac * T, n_coarse)
    thetas = np.array([pdm_theta(t, L, p, n_bins=n_bins) for p in periods])

    i_best = int(np.argmin(thetas))
    best_period = periods[i_best]
    best_theta = thetas[i_best]

    refined_period = best_period
    if refine and 0 < i_best < len(periods) - 1:
        p0, p1, p2 = periods[i_best - 1], periods[i_best], periods[i_best + 1]
        y0, y1, y2 = thetas[i_best - 1], thetas[i_best], thetas[i_best + 1]
        denom = (y0 - 2 * y1 + y2)
        if abs(denom) > 1e-12:
            refined_period = p1 + 0.5 * (y0 - y2) / denom * (p1 - p0)

    edge_tol = max(2, int(0.03 * n_coarse))  # within 3% of the search range from either edge
    hit_boundary = i_best <= edge_tol or i_best >= len(periods) - 1 - edge_tol

    window = max(5, int(0.2 * n_coarse))
    if i_best >= len(periods) - 1 - edge_tol:
        trail = thetas[-window:]
        monotonic_to_edge = np.all(np.diff(trail) <= 1e-9)
    elif i_best <= edge_tol:
        trail = thetas[:window]
        monotonic_to_edge = np.all(np.diff(trail) >= -1e-9)
    else:
        monotonic_to_edge = False

    rng = np.random.default_rng(rng_seed)
    null_best_thetas = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(L.shape[0])
        L_shuffled = L[perm]
        null_thetas = np.array([pdm_theta(t, L_shuffled, p, n_bins=n_bins) for p in periods])
        null_best_thetas[i] = null_thetas.min()
    p_value = float((null_best_thetas <= best_theta).mean())
    has_any_structure = p_value < 0.05

    reliable = (not hit_boundary) and (not monotonic_to_edge) and has_any_structure
    if not reliable:
        reasons = []
        if hit_boundary or monotonic_to_edge:
            reasons.append("theta(P) has no interior minimum and trends toward the edge "
                            "of the search range (classic 'recording shorter than true "
                            "period' signature)")
        if not has_any_structure:
            reasons.append(f"not even distinguishable from {n_perm} shuffled-data controls "
                            f"(p={p_value:.3f}) -- no periodicity detected at all")
        print(f"[period search / PDM] WARNING: result UNRELIABLE -- {'; '.join(reasons)}. "
              f"This likely means the recording does not contain a full rotation; "
              f"treat the period estimate below as a rough fallback, not ground truth.")

    return {
        "period": float(refined_period),
        "coarse_best_period": float(best_period),
        "theta_at_best": float(best_theta),
        "periods": periods,
        "thetas": thetas,
        "hit_boundary": bool(hit_boundary),
        "monotonic_to_edge": bool(monotonic_to_edge),
        "p_value": p_value,
        "has_any_structure": bool(has_any_structure),
        "reliable": bool(reliable),
    }


def estimate_period_from_data(time: np.ndarray, L_raw: np.ndarray, cfg_ps: dict) -> tuple[float, bool]:
    """PDM-based period search (see `search_period_pdm` above)."""
    L_norm = L_raw / L_raw.mean(axis=0, keepdims=True)  # match dataset's own preprocessing
    result = search_period_pdm(
        time, L_norm,
        p_min_frac=cfg_ps.get("p_min_frac", 0.5),
        p_max_frac=cfg_ps.get("p_max_frac", 1.0),
        n_bins=cfg_ps.get("n_bins", 12),
    )
    print(f"[period search / PDM] coarse best={result['coarse_best_period']:.4f}  "
          f"(theta={result['theta_at_best']:.4f})  refined={result['period']:.4f}  "
          f"reliable={result['reliable']}  "
          f"[searched {cfg_ps.get('p_min_frac', 0.5)}-{cfg_ps.get('p_max_frac', 1.0)} x T]")
    if not result["reliable"]:
        print("[period search / PDM] proceeding with this as a FALLBACK estimate only -- "
              "see warning above. Consider cross-checking against other lightcurve files "
              "of the same object if available, and inspect result['thetas'] vs "
              "result['periods'] directly before trusting downstream results from this run.")
    return result["period"], result["reliable"]


def _is_better(score, best_score) -> bool:
    """`validate_and_log` returns None on a failed validation (metric error,
    e.g. a transient bad mesh) -- the original code compared `score >
    best_val[1]` unconditionally, which raises TypeError the first time
    either side is None instead of just skipping the update. Fixed here:
    a None score never wins; a None best_score always loses to a real one."""
    if score is None:
        return False
    if best_score is None:
        return True
    return score > best_score


def plateau_lr_step(opt: torch.optim.Optimizer, tr: dict, n_decays: int, step: int) -> tuple[int, bool]:
    """Called when the patience counter hits cfg['tol'] (loss hasn't beaten
    its best value for that many steps in a row). Reuses that SAME plateau
    signal the training loop already tracks -- instead of stopping
    outright, cuts every param group's LR by `tr['lr_decay_factor']` and
    tells the caller to keep going (with patience reset to 0), on the
    theory that the plateau is often "this LR is too big to make further
    progress here", not "there is no further progress to make". Only
    actually stops once `tr.get('lr_max_decays', 3)` decays have already
    happened with no improvement in between -- at that point a smaller LR
    isn't helping either, so it really has converged.

    If `tr['lr_decay_factor']` isn't set at all, returns should_stop=True
    immediately -- the original hard-stop-on-first-plateau behavior, kept
    as the default so existing configs (no new fields added) run exactly
    as before.

    Returns (new_n_decays, should_stop).
    """
    decay_factor = tr.get("lr_decay_factor")
    if decay_factor is None:
        return n_decays, True
    max_decays = tr.get("lr_max_decays", 3)
    if n_decays >= max_decays:
        return n_decays, True
    for g in opt.param_groups:
        g["lr"] *= decay_factor
    print(f"[lr schedule] step {step}: loss plateaued -- reducing lr x{decay_factor} "
          f"(decay {n_decays + 1}/{max_decays}), new lr(s)={[g['lr'] for g in opt.param_groups]}")
    return n_decays + 1, False


def write_obj(path, vertices: torch.Tensor, faces: torch.Tensor):
    v = vertices.detach().cpu().numpy()
    f = faces.detach().cpu().numpy()
    with open(path, "w") as fh:
        for vv in v:
            fh.write(f"v {vv[0]:.6f} {vv[1]:.6f} {vv[2]:.6f}\n")
        for ff in f:
            fh.write(f"f {ff[0] + 1} {ff[1] + 1} {ff[2] + 1}\n")


# ---------------------------------------------------------------------
# training
# ---------------------------------------------------------------------

def train(cfg, config_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    run_dir = make_run_dir(cfg, config_path)
    print(f"run folder: {run_dir}")

    torch.manual_seed(cfg["training"]["seed"])
    np.random.seed(cfg["training"]["seed"])
    dtype = cfg["model"].get("dtype", "float32")

    model = ChallengeForward(LMAX=cfg["model"]["LMAX"], nrows=cfg["model"]["nrows"],
                              E_cams_world=make_challenge_cameras_28(dtype=dtype),
                              dtype=dtype, device=device).to(device)

    t_i, ch_i = load_lightcurve_28(cfg["data"]["intensity_file"])
    t_b, ch_b = load_lightcurve_28(cfg["data"]["binary_file"])
    assert np.allclose(t_i, t_b), "intensity/binary files must share timestamps"
    TIME = torch.tensor(t_i, dtype=dtype).to(device)
    L_intensity = normalize_columns(torch.tensor(ch_i, dtype=dtype)).to(device)
    L_binary = normalize_columns(torch.tensor(ch_b, dtype=dtype)).to(device)

    # Optional second data source: Blender-simulated lightcurves for the
    # SAME object (same 28-column format, same physical rotation -- same
    # omega/omega0 below -- just a different TIME grid: an exact single
    # rotation, 360 evenly-spaced synthetic frames per the challenge page,
    # vs. the real recording's 800+ frames that run slightly past one
    # rotation). Both share the same t=0 convention (challenge page: "at
    # the initial time of any lightcurve, that [fixed] point faces the
    # light source"), so no extra phase alignment is needed -- omega0=0
    # applies to both as-is. May also need its own camera_distance
    # (Blender's virtual camera isn't necessarily calibrated the same as
    # the real optical setup) -- calibrate with test_operator.py rather
    # than assuming the real data's value transfers.
    blender_intensity_file = cfg["data"].get("intensity_file_blender")
    blender_binary_file = cfg["data"].get("binary_file_blender")
    use_blender = bool(blender_intensity_file and blender_binary_file)
    if use_blender:
        t_i_bl, ch_i_bl = load_lightcurve_28(blender_intensity_file)
        t_b_bl, ch_b_bl = load_lightcurve_28(blender_binary_file)
        assert np.allclose(t_i_bl, t_b_bl), "blender intensity/binary files must share timestamps"
        TIME_blender = torch.tensor(t_i_bl, dtype=dtype).to(device)
        L_intensity_blender = normalize_columns(torch.tensor(ch_i_bl, dtype=dtype)).to(device)
        L_binary_blender = normalize_columns(torch.tensor(ch_b_bl, dtype=dtype)).to(device)
        print(f"[blender] joint training enabled: {len(t_i_bl)} frames")

    # Ellipsoid warm-start r0: only needs the octantoid grid's own vertex
    # directions (fixed by LMAX/nrows), not a full hypernet instance --
    # avoids building (and immediately discarding) the real hypernet twice.
    directions = octantoid_directions(cfg["model"]["LMAX"], cfg["model"]["nrows"],
                                       dtype=dtype, device=device)
    r0_ellipsoid = estimate_ellipsoid_r0(directions, L_intensity,
                                          cylinder_radius=cfg["physical"]["cylinder_radius"],
                                          damping=0.5)
    hyper = build_hypernet(cfg, dtype=dtype, r0=r0_ellipsoid).to(device)

    phys = cfg["physical"]
    tr = cfg["training"]
    val_cfg = cfg.get("validation", {})

    # Every internal get_mesh call in this run must apply the SAME
    # cylinder_radius -- bind it once here rather than threading a literal
    # through every call site below (easy to miss one otherwise).
    cylinder_radius = phys["cylinder_radius"]

    def _get_mesh(h, m, **kw):
        return get_mesh(h, m, cylinder_radius=cylinder_radius, **kw)

    # camera_distance used to be a bare literal (30.0) in the
    # simulate_all_curves call below -- now config-driven (still defaults
    # to 30.0, so an unconfigured run behaves identically) so a value
    # found via test_operator.py's sweep can actually be used without
    # editing this file. blender_camera_distance defaults to the SAME
    # value as camera_distance if not set separately, since it's entirely
    # possible the two turn out to match -- only set it if your own sweep
    # against the Blender curves says otherwise.
    camera_distance = phys.get("camera_distance", 30.0)
    blender_camera_distance = phys.get("blender_camera_distance", camera_distance)
    blender_loss_weight = phys.get("blender_loss_weight", 1.0)

    if phys.get("period", "auto") == "auto":
        period, period_reliable = estimate_period_from_data(t_i, ch_i, phys.get("period_search", {}))
    else:
        period, period_reliable = float(phys["period"]), True
    fit_omega = phys.get("fit_omega", False)
    # Built directly on `device` (not torch.tensor(...).to(device)) -- if
    # requires_grad=True is set BEFORE a device move, the move itself is
    # tracked by autograd and the result is a non-leaf tensor, which the
    # optimizer then rejects ("can't optimize a non-leaf Tensor"). Only
    # bites when device != cpu (a cpu->cpu .to() is a no-op returning the
    # same leaf tensor, which is why this only shows up on GPU runs).
    omega = torch.tensor(-2 * np.pi / period, dtype=dtype, device=device, requires_grad=fit_omega)
    omega0 = torch.tensor(0.0, dtype=dtype, device=device, requires_grad=fit_omega)

    # Optional drift constraint on omega itself -- see train_convex.py's
    # identical block for the full reasoning. Soft during every training
    # forward pass, hard once for the final exported/checkpointed value.
    omega_bounds = None
    if phys.get("omega_period_min_frac") is not None and phys.get("omega_period_max_frac") is not None:
        T_obs = float(t_i[-1] - t_i[0])
        period_lo = phys["omega_period_min_frac"] * T_obs
        period_hi = phys["omega_period_max_frac"] * T_obs
        omega_bounds = (-2 * np.pi / period_lo, -2 * np.pi / period_hi)
        print(f"[refine] omega constrained to period in [{period_lo:.3f}, {period_hi:.3f}] "
              f"(omega in [{omega_bounds[0]:.6f}, {omega_bounds[1]:.6f}])")

    def _omega_eff():
        return soft_omega_clamp(omega, *omega_bounds) if omega_bounds is not None else omega

    params = list(hyper.parameters())
    if fit_omega:
        # separate, much smaller LR for omega/omega0 -- a single rotation's
        # worth of data makes joint fitting far less dangerous than the
        # multi-apparition case, but the scale of omega vs. shape-network
        # weights is very different, so don't share the shape LR with it.
        opt = torch.optim.Adam([
            {"params": params, "lr": tr["lr"]},
            {"params": [omega, omega0], "lr": tr.get("omega_lr", tr["lr"] * 0.1)},
        ])
    else:
        opt = torch.optim.Adam(params, lr=tr["lr"])

    labels = challenge_column_labels()
    print(f"hypernet={cfg['model']['hypernet']}  columns={len(labels)}  "
          f"e.g. {labels[:4]} ...")

    metrics_path = run_dir / "metrics.csv"
    loss_old = float("inf")
    patience = 0
    n_lr_decays = 0

    # `model` never trains (only hyper.parameters() are in the optimizer),
    # so "best_val" only ever needs to snapshot `hyper` -- `model` is
    # reused directly below instead of being deep-copied alongside it.
    score = validate_and_log(hyper, model, _get_mesh, _omega_eff(), omega0, TIME[0],
                              gt_path=val_cfg["gt_path"], run_dir=run_dir, step=0, val_cfg=val_cfg,
                              cylinder_radius=cylinder_radius)
    best_val = (0, score, copy.deepcopy(hyper))  # (step, score, hyper)
    tlist, vlist = _get_mesh(best_val[2], model)
    write_obj(run_dir / "initial_mesh.obj", vlist, tlist)
    export_reconstruction(vlist, tlist, _omega_eff(), omega0, TIME[0],
                           run_dir / "initial_mesh.stl", cylinder_radius)
    with open(metrics_path, "w", newline="") as mf:
        writer = csv.writer(mf)
        writer.writerow(["step", "loss", "loss_intensity", "loss_binary",
                          "loss_intensity_blender", "loss_binary_blender",
                          "w_intensity", "w_binary", "lr", "score", "omega", "omega0"])

        for step in range(tr["steps"]):
            opt.zero_grad()
            tlist, vlist = _get_mesh(hyper, model)

            # Horizon candidates depend only on (tlist, vlist), not on
            # TIME/omega -- computed once here and reused for both the
            # real and (if enabled) Blender simulate_all_curves calls
            # below, instead of paying for facets_over_horizon's O(nfac^2)
            # candidate generation twice for the identical mesh.
            cached_horizon = facets_over_horizon(tlist, vlist.detach())

            omega_eff = _omega_eff()
            curves = simulate_all_curves(tlist, vlist, model.E_cams_world, TIME,
                                          omega=omega_eff, omega0=omega0,
                                          mode="both", rel=True,
                                          camera_distance=camera_distance, scattering='mixed', ls_weight=0.03,
                                          cached_horizon=cached_horizon)
            intensity_sim_n = normalize_columns(curves[0].T)
            binary_sim_n = normalize_columns(curves[1].T)

            loss_int = ((intensity_sim_n - L_intensity) ** 2).mean()
            loss_bin = ((binary_sim_n - L_binary) ** 2).mean()

            w_bin = curriculum(step, tr["steps"], tr["lambda_binary_start"], tr["lambda_binary_end"])
            w_int = curriculum(step, tr["steps"], tr["lambda_intensity_start"], tr["lambda_intensity_end"])
            loss = w_bin * loss_bin + w_int * loss_int

            loss_int_bl = loss_bin_bl = None
            if use_blender:
                curves_bl = simulate_all_curves(tlist, vlist, model.E_cams_world, TIME_blender,
                                                 omega=omega_eff, omega0=omega0,
                                                 mode="both", rel=True,
                                                 camera_distance=blender_camera_distance, scattering='mixed', ls_weight=0.03,
                                                 cached_horizon=cached_horizon)
                intensity_sim_n_bl = normalize_columns(curves_bl[0].T)
                binary_sim_n_bl = normalize_columns(curves_bl[1].T)
                loss_int_bl = ((intensity_sim_n_bl - L_intensity_blender) ** 2).mean()
                loss_bin_bl = ((binary_sim_n_bl - L_binary_blender) ** 2).mean()
                loss = loss + blender_loss_weight * (w_bin * loss_bin_bl + w_int * loss_int_bl)

            loss.backward()
            opt.step()
            current_lr = opt.param_groups[0]["lr"]

            # loss_int_bl/loss_bin_bl are None when use_blender is False --
            # logged as blank rather than 0.0 so they're not mistaken for
            # an actually-computed zero loss.
            bl_row = [loss_int_bl.item() if loss_int_bl is not None else "",
                      loss_bin_bl.item() if loss_bin_bl is not None else ""]
            # EFFECTIVE (soft-clamped, if omega_bounds is set) omega as of
            # THIS step's update (post opt.step() above), matching what's
            # actually used in the forward pass -- not the raw underlying
            # parameter, which can wander outside the bounds while the
            # effective value stays pinned near them. Logged every row
            # regardless of fit_omega so you can watch it move (or confirm
            # it's frozen) directly instead of inferring it from the loss.
            om_row = [_omega_eff().item(), omega0.item()]

            if loss < loss_old:
                patience = 0
                loss_old = loss.item()
            else:
                patience += 1
                if patience >= cfg['tol']:
                    n_lr_decays, should_stop = plateau_lr_step(opt, tr, n_lr_decays, step)
                    if not should_stop:
                        patience = 0
                    else:
                        mf.flush()
                        print(f"Converged at step {step} with loss {loss.item():.6f} and patience {patience}")
                        if val_cfg.get("enabled", False) and step % val_cfg.get("every", 500) == 0:
                            score = validate_and_log(hyper, model, _get_mesh, _omega_eff(), omega0, TIME[0],
                                                      gt_path=val_cfg["gt_path"], run_dir=run_dir, step=step, val_cfg=val_cfg,
                                                      cylinder_radius=cylinder_radius)
                            if _is_better(score, best_val[1]):
                                best_val = (step, score, copy.deepcopy(hyper))
                        writer.writerow([step, loss.item(), loss_int.item(), loss_bin.item(), *bl_row,
                                          w_int, w_bin, current_lr, score, *om_row])
                        break

            if step % tr["log_every"] == 0:
                mf.flush()
                bl_msg = f"  intensity_bl {loss_int_bl.item():.6f}  binary_bl {loss_bin_bl.item():.6f}" \
                    if use_blender else ""
                print(f"step {step:5d}  loss {loss.item():.6f}  "
                      f"intensity {loss_int.item():.6f}  binary {loss_bin.item():.6f}{bl_msg}  lr {current_lr:.3g}")
            if step == 0 or step % tr["log_every"] == 0:
                writer.writerow([step, loss.item(), loss_int.item(), loss_bin.item(), *bl_row,
                                  w_int, w_bin, current_lr, score, *om_row])
            else:
                writer.writerow([step, loss.item(), loss_int.item(), loss_bin.item(), *bl_row,
                                  w_int, w_bin, current_lr, "", *om_row])

            if val_cfg.get("enabled", False) and (step + 1) % val_cfg.get("every", 500) == 0:
                score = validate_and_log(hyper, model, _get_mesh, _omega_eff(), omega0, TIME[0],
                                          gt_path=val_cfg["gt_path"], run_dir=run_dir, step=step + 1, val_cfg=val_cfg,
                                          cylinder_radius=cylinder_radius)
                if _is_better(score, best_val[1]):
                    best_val = (step + 1, score, copy.deepcopy(hyper))

    # Hard clamp (not soft) for everything exported/checkpointed below --
    # this is the final, "official" value, mirroring radial_clamp being
    # used at export time vs soft_radial_clamp during training.
    omega_final = omega_clamp(omega.detach(), *omega_bounds) if omega_bounds is not None else omega.detach()

    tlist, vlist = _get_mesh(hyper, model)
    write_obj(run_dir / "final_mesh.obj", vlist, tlist)
    export_reconstruction(vlist, tlist, omega_final, omega0, TIME[0],
                           run_dir / "final_mesh.stl", cylinder_radius)     # challenge-frame, for the metric
    tlist, vlist = _get_mesh(best_val[2], model)
    write_obj(run_dir / "best_val_mesh.obj", vlist, tlist)
    export_reconstruction(vlist, tlist, omega_final, omega0, TIME[0],
                           run_dir / "best_val_mesh.stl", cylinder_radius)  # challenge-frame, for the metric
    torch.save({"hyper_state_dict": hyper.state_dict(),
                "omega": omega_final, "omega0": omega0.detach(),
                "period_used": period, "period_reliable": period_reliable,
                "config": cfg}, run_dir / "checkpoint.pt")
    print(f"done. mesh + checkpoint + metrics.csv + config.yaml saved in {run_dir}")
    print(f"best val score: {best_val[1]}; best val step: {best_val[0]}")
