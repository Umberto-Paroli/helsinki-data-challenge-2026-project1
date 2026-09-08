"""
train_convex.py
================

Stage 1 of a two-stage reconstruction: fit a genuinely CONVEX shape to the
lightcurves, with no self-shadowing ambiguity and (unlike the general
non-convex problem) a well-posed, close-to-unique solution -- the same
reasoning behind classical convex lightcurve inversion (Kaasalainen &
Torppa 2001): a convex body is determined by its surface curvature/area
function (Minkowski problem), so enough lightcurve geometries pin it down
almost uniquely, while concavities are only weakly/ambiguously constrained
by disk-integrated photometry alone.

Shape representation: K freely-moving points in R^3 (not radius-along-a-
fixed-direction, so it can represent genuine flat faces / sharp edges,
e.g. a cube, which no spherical-harmonics or radial-function
representation can) -- passed through `geometry.convex_hull_mesh` before
every render, which is convex by construction no matter where the points
sit. Two interchangeable generators (config: `model.generator`):
  "points"     -> `hypernets.ConvexPointCloud`, fully independent
                   per-point parameters, zero hyperparameters.
  "shared_mlp" -> `hypernets.ConvexOffsetNet`, a small MLP shared across
                   all points (fixed per-point noise -> offset), no
                   neighbor-mixing -- meant to keep "dead" (hull-interior)
                   points reachable by gradient via the shared weights.
See both classes' docstrings in hypernets.py for the full trade-off; try
both and compare, there's no a priori winner. Rendering uses
`simulate_all_curves(..., assume_convex=True)`, which skips the O(nfac^2)
self-shadowing test entirely: this is exact (not an approximation) for a
body that really is convex, and is the single biggest cost in the forward
model, so this stage is also considerably cheaper per step than the
non-convex one.

Deliberately kept as its own file/entry point (`main_convex.py`), separate
from `train.py`/`main.py`, so the two stages can be run and inspected
independently for now. Reuses everything reusable from `train.py` (data
loading, period search, run-dir/logging plumbing, the loss formulation,
the plateau-triggered LR decay) rather than duplicating it. Once this
stage is trusted, wiring its output into the non-convex stage as a warm
start is the natural next step -- see `train.py`'s
`hypernet == "warmstart_points"` branch, which does exactly that from the
`checkpoint.pt` this file saves.
"""
from __future__ import annotations
import csv
import copy

import numpy as np
import torch

from geometry import (
    ChallengeForward, make_challenge_cameras_28, challenge_column_labels,
    simulate_all_curves, octantoid_directions, convex_hull_mesh,
)
from hypernets import ConvexPointCloud, ConvexOffsetNet, estimate_ellipsoid_r0
from validation import validate_and_log, export_reconstruction, fit_z_range, soft_fit_z_range, \
    radial_clamp, soft_radial_clamp, omega_clamp, soft_omega_clamp
from train import (
    make_run_dir, load_lightcurve_28, normalize_columns, curriculum,
    estimate_period_from_data, write_obj, _is_better, plateau_lr_step,
)


def build_convex_generator(cfg: dict, directions: torch.Tensor, r0, dtype=torch.float64):
    """Dispatch on cfg["model"].get("generator", "points"): "points" ->
    ConvexPointCloud (independent per-point parameters), "shared_mlp" ->
    ConvexOffsetNet (weight-shared MLP, no neighbor-mixing). See both
    classes' docstrings in hypernets.py for the trade-off."""
    kind = cfg["model"].get("generator", "points")
    if kind == "points":
        return ConvexPointCloud(directions, r0, dtype=dtype)
    if kind == "shared_mlp":
        kwargs = dict(cfg["model"].get("generator_kwargs", {}))
        init_positions = directions.to(dtype=dtype) * (
            r0.to(dtype=dtype) if isinstance(r0, torch.Tensor)
            else torch.full((directions.shape[0],), float(r0), dtype=dtype, device=directions.device)
        ).unsqueeze(-1)
        return ConvexOffsetNet(init_positions, dtype=dtype,
                                **{k: v for k, v in kwargs.items()
                                   if k in ("noise_dim", "hidden", "n_layers")})
    raise ValueError(f"unknown model.generator: {kind!r} (expected 'points' or 'shared_mlp')")


def get_mesh_convex(gen, model: ChallengeForward, beta: float = 30.0, training: bool = True,
                     cylinder_radius=None):
    """Points -> [radial clamp] -> convex hull -> z-range fit. Mirrors
    `train.get_mesh`'s shape (same fit_z_range/soft_fit_z_range
    convention, same training-vs-export switch), just with the hull step
    standing in for `model.mesh(a)` / the graph-based generators' own
    (tlist, vlist). Works identically for either generator kind -- both
    just return a (K,3) position tensor from `gen()`.

    If `cylinder_radius` is given, the radial cap is applied to the raw
    points BEFORE the hull, not to the hull's output vertices after (the
    order matters here, unlike in train.get_mesh): `convex_hull_mesh` is
    convex by construction for ANY input points, so clamping pre-hull
    keeps that guarantee exactly. fit_z_range is fine post-hull because
    it's an affine map (uniform scale + shift) -- affine maps preserve
    convexity. Radial clamping is NOT affine (only out-of-bounds points
    move), so clamping the hull's own output vertices post-hoc could pull
    some of them back inside the hull and leave a mesh whose face
    connectivity no longer matches a true convex hull of those positions."""
    points = gen()
    if cylinder_radius is not None:
        points = soft_radial_clamp(points, cylinder_radius, beta=beta) if training \
            else radial_clamp(points, cylinder_radius)
    tlist, vlist = convex_hull_mesh(points)
    vlist = soft_fit_z_range(vlist, beta=beta) if training else fit_z_range(vlist)
    return tlist, vlist


def train_convex(cfg, config_path):
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

    # Optional joint training on Blender-simulated lightcurves for the
    # SAME object -- see train.py's identical block for the full
    # reasoning (same camera/column format, same physical omega/omega0,
    # different TIME grid and possibly different camera_distance).
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

    # Same ellipsoid warm-start estimate the non-convex stage uses (see
    # train.py) -- here it ALSO doubles as the initial point positions:
    # every point starts on this convex surface, so ~all of them start
    # "active" on the hull (see ConvexPointCloud's docstring for why that
    # matters for this particular parametrization).
    directions = octantoid_directions(cfg["model"]["LMAX"], cfg["model"]["nrows"],
                                       dtype=dtype, device=device)
    r0_ellipsoid = estimate_ellipsoid_r0(directions, L_intensity,
                                          cylinder_radius=cfg["physical"]["cylinder_radius"],
                                          damping=0.5)
    gen = build_convex_generator(cfg, directions, r0_ellipsoid, dtype=dtype).to(device)

    phys = cfg["physical"]
    tr = cfg["training"]
    val_cfg = cfg.get("validation", {})

    # Same reasoning as train.py's _get_mesh: bind cylinder_radius once so
    # every internal get_mesh_convex call below applies it consistently.
    cylinder_radius = phys["cylinder_radius"]

    def _get_mesh_convex(g, m, **kw):
        return get_mesh_convex(g, m, cylinder_radius=cylinder_radius, **kw)

    # See train.py's identical block -- config-driven now instead of a
    # bare 30.0 literal, defaults preserve prior behavior exactly.
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

    # Optional drift constraint on omega itself (separate from period_search's
    # bounds, which only shape PDM's STARTING estimate -- fit_omega's
    # subsequent gradient descent is otherwise free to wander arbitrarily far
    # from it, in either direction, with nothing stopping it. See chat: one
    # object's calibration kept drifting the period down with no floor, and a
    # different object's near-symmetric shape risked the same happening
    # upward via aliasing). Soft (smooth, nonzero gradient beyond the bound)
    # during every training forward pass; exact/hard only applied once, to
    # the value actually exported/checkpointed -- mirrors soft_radial_clamp/
    # radial_clamp exactly. Opt-in: only active when BOTH fractions are set.
    omega_bounds = None
    if phys.get("omega_period_min_frac") is not None and phys.get("omega_period_max_frac") is not None:
        T_obs = float(t_i[-1] - t_i[0])
        period_lo = phys["omega_period_min_frac"] * T_obs
        period_hi = phys["omega_period_max_frac"] * T_obs
        omega_bounds = (-2 * np.pi / period_lo, -2 * np.pi / period_hi)
        print(f"[convex] omega constrained to period in [{period_lo:.3f}, {period_hi:.3f}] "
              f"(omega in [{omega_bounds[0]:.6f}, {omega_bounds[1]:.6f}])")

    def _omega_eff():
        return soft_omega_clamp(omega, *omega_bounds) if omega_bounds is not None else omega

    params = list(gen.parameters())
    if fit_omega:
        opt = torch.optim.Adam([
            {"params": params, "lr": tr["lr"]},
            {"params": [omega, omega0], "lr": tr.get("omega_lr", tr["lr"] * 0.1)},
        ])
    else:
        opt = torch.optim.Adam(params, lr=tr["lr"])

    labels = challenge_column_labels()
    print(f"[convex] generator={type(gen).__name__}  K={gen().shape[0]} points  "
          f"columns={len(labels)}  e.g. {labels[:4]} ...")

    metrics_path = run_dir / "metrics.csv"
    loss_old = float("inf")
    patience = 0
    n_lr_decays = 0

    # `model` never trains here either (only gen.parameters() are in the
    # optimizer) -- same reasoning as train.py: best_val only snapshots
    # `gen`, `model` is reused directly.
    score = validate_and_log(gen, model, _get_mesh_convex, _omega_eff(), omega0, TIME[0],
                              gt_path=val_cfg["gt_path"], run_dir=run_dir, step=0, val_cfg=val_cfg,
                              cylinder_radius=cylinder_radius)
    best_val = (0, score, copy.deepcopy(gen))  # (step, score, gen)
    tlist, vlist = _get_mesh_convex(best_val[2], model)
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
            tlist, vlist = _get_mesh_convex(gen, model)

            omega_eff = _omega_eff()
            curves = simulate_all_curves(tlist, vlist, model.E_cams_world, TIME,
                                          omega=omega_eff, omega0=omega0,
                                          mode="both", rel=True,
                                          camera_distance=camera_distance, scattering='mixed', ls_weight=0.03,
                                          assume_convex=True)
            intensity_sim_n = normalize_columns(curves[0].T)
            binary_sim_n = normalize_columns(curves[1].T)

            loss_int = ((intensity_sim_n - L_intensity) ** 2).mean()
            loss_bin = ((binary_sim_n - L_binary) ** 2).mean()

            w_bin = curriculum(step, tr["steps"], tr["lambda_binary_start"], tr["lambda_binary_end"])
            w_int = curriculum(step, tr["steps"], tr["lambda_intensity_start"], tr["lambda_intensity_end"])
            loss = w_bin * loss_bin + w_int * loss_int

            # No cached_horizon reuse to wire in here (unlike train.py):
            # assume_convex=True already skips facets_over_horizon
            # entirely for BOTH calls, so there's nothing shared to reuse.
            loss_int_bl = loss_bin_bl = None
            if use_blender:
                curves_bl = simulate_all_curves(tlist, vlist, model.E_cams_world, TIME_blender,
                                                 omega=omega_eff, omega0=omega0,
                                                 mode="both", rel=True,
                                                 camera_distance=blender_camera_distance, scattering='mixed', ls_weight=0.03,
                                                 assume_convex=True)
                intensity_sim_n_bl = normalize_columns(curves_bl[0].T)
                binary_sim_n_bl = normalize_columns(curves_bl[1].T)
                loss_int_bl = ((intensity_sim_n_bl - L_intensity_blender) ** 2).mean()
                loss_bin_bl = ((binary_sim_n_bl - L_binary_blender) ** 2).mean()
                loss = loss + blender_loss_weight * (w_bin * loss_bin_bl + w_int * loss_int_bl)

            loss.backward()
            opt.step()
            current_lr = opt.param_groups[0]["lr"]

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
                        print(f"[convex] Converged at step {step} with loss {loss.item():.6f} and patience {patience}")
                        if val_cfg.get("enabled", False) and step % val_cfg.get("every", 500) == 0:
                            score = validate_and_log(gen, model, _get_mesh_convex, _omega_eff(), omega0, TIME[0],
                                                      gt_path=val_cfg["gt_path"], run_dir=run_dir, step=step, val_cfg=val_cfg,
                                                      cylinder_radius=cylinder_radius)
                            if _is_better(score, best_val[1]):
                                best_val = (step, score, copy.deepcopy(gen))
                        writer.writerow([step, loss.item(), loss_int.item(), loss_bin.item(), *bl_row,
                                          w_int, w_bin, current_lr, score, *om_row])
                        break

            if step % tr["log_every"] == 0:
                mf.flush()
                bl_msg = f"  intensity_bl {loss_int_bl.item():.6f}  binary_bl {loss_bin_bl.item():.6f}" \
                    if use_blender else ""
                print(f"[convex] step {step:5d}  loss {loss.item():.6f}  "
                      f"intensity {loss_int.item():.6f}  binary {loss_bin.item():.6f}{bl_msg}  lr {current_lr:.3g}")
            if step == 0 or step % tr["log_every"] == 0:
                writer.writerow([step, loss.item(), loss_int.item(), loss_bin.item(), *bl_row,
                                  w_int, w_bin, current_lr, score, *om_row])
            else:
                writer.writerow([step, loss.item(), loss_int.item(), loss_bin.item(), *bl_row,
                                  w_int, w_bin, current_lr, "", *om_row])

            if val_cfg.get("enabled", False) and (step + 1) % val_cfg.get("every", 500) == 0:
                score = validate_and_log(gen, model, _get_mesh_convex, _omega_eff(), omega0, TIME[0],
                                          gt_path=val_cfg["gt_path"], run_dir=run_dir, step=step + 1, val_cfg=val_cfg,
                                          cylinder_radius=cylinder_radius)
                if _is_better(score, best_val[1]):
                    best_val = (step + 1, score, copy.deepcopy(gen))

    # Hard clamp (not soft) for everything exported/checkpointed below --
    # this is the final, "official" value, mirroring radial_clamp being
    # used at export time vs soft_radial_clamp during training.
    omega_final = omega_clamp(omega.detach(), *omega_bounds) if omega_bounds is not None else omega.detach()

    tlist, vlist = _get_mesh_convex(gen, model)
    write_obj(run_dir / "final_mesh.obj", vlist, tlist)
    export_reconstruction(vlist, tlist, omega_final, omega0, TIME[0],
                           run_dir / "final_mesh.stl", cylinder_radius)
    tlist, vlist = _get_mesh_convex(best_val[2], model)
    write_obj(run_dir / "best_val_mesh.obj", vlist, tlist)
    export_reconstruction(vlist, tlist, omega_final, omega0, TIME[0],
                           run_dir / "best_val_mesh.stl", cylinder_radius)
    # Saved as raw (K,3) positions regardless of which generator produced
    # them ("points" or "shared_mlp") -- this is what train.py's
    # hypernet="warmstart_points" loads to continue into the non-convex
    # refinement stage.
    torch.save({"points": gen().detach(),
                "generator_state_dict": gen.state_dict(),
                "generator_kind": type(gen).__name__,
                "omega": omega_final, "omega0": omega0.detach(),
                "period_used": period, "period_reliable": period_reliable,
                "config": cfg}, run_dir / "checkpoint.pt")
    print(f"[convex] done. mesh + checkpoint + metrics.csv + config.yaml saved in {run_dir}")
    print(f"[convex] best val score: {best_val[1]}; best val step: {best_val[0]}")
