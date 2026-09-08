"""
validation.py
==============

- Binary STL export.
- Normalizing a reconstructed mesh into the required challenge frame
  (rotation axis = z, top/bottom touch z=+-1, pose matches the first
  lightcurve frame).
- Voxel-IoU (Dice) shape-similarity metric against ground truth.
- A pluggable validation hook called periodically from the training loop.

Consolidated from the former `validation.py` + `voxel_measure.py` (the
latter was only ever called from here, as `compute_voxel_metric`'s
backend).

Efficiency fix vs. the original: `validate_and_log` is called every
`val_cfg["every"]` steps -- with a typical config that's every few steps,
so ~100-200 times per run -- and every call used to reload AND re-voxelize
the ground-truth STL from scratch, even though it never changes during a
run. The ground-truth voxelization is now cached (keyed on path + voxel
size), so repeated validation calls only redo the (cheap) reconstruction
side. This is the main real memory/CPU cost `validate_and_log` was adding
to every run, independent of file layout.
"""
from __future__ import annotations
import functools
import struct
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import trimesh

from geometry import rotate_z
from sideview import sideview_score


# ---------------------------------------------------------------------
# STL export
# ---------------------------------------------------------------------

def write_stl_binary(path, vertices: torch.Tensor, faces: torch.Tensor):
    """Standard binary STL: 80-byte header, uint32 triangle count, then
    50 bytes/triangle (12 bytes normal + 3x12 bytes vertices + 2 bytes
    attribute)."""
    v = vertices.detach().cpu().numpy().astype(np.float32)
    f = faces.detach().cpu().numpy()

    v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norm_len = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, norm_len, out=np.zeros_like(normals),
                         where=norm_len > 1e-20)

    with open(path, "wb") as fh:
        fh.write(b"\0" * 80)
        fh.write(struct.pack("<I", len(f)))
        for i in range(len(f)):
            fh.write(struct.pack("<3f", *normals[i]))
            fh.write(struct.pack("<3f", *v0[i]))
            fh.write(struct.pack("<3f", *v1[i]))
            fh.write(struct.pack("<3f", *v2[i]))
            fh.write(struct.pack("<H", 0))


# ---------------------------------------------------------------------
# Orientation / scale normalization
# ---------------------------------------------------------------------

def soft_fit_z_range(vlist: torch.Tensor, beta: float = 30.0) -> torch.Tensor:
    """Same intent as fit_z_range (rescale+shift so the mesh touches
    z~=-1,+1), but using smooth soft-min/soft-max (LogSumExp) so gradient
    is distributed across all vertices instead of concentrated on the
    extremes. Use this during training."""
    z = vlist[:, 2]
    soft_max = torch.logsumexp(beta * z, dim=0) / beta
    soft_min = -torch.logsumexp(-beta * z, dim=0) / beta
    s = 2.0 / (soft_max - soft_min).clamp(min=1e-12)
    dz = -1.0 - s * soft_min
    v = vlist * s
    v = torch.cat([v[:, :2], (v[:, 2] + dz).unsqueeze(-1)], dim=-1)
    return v


def fit_z_range(vlist: torch.Tensor) -> torch.Tensor:
    """Isotropic rescale + z-only shift so the mesh EXACTLY touches z=-1
    and z=+1. Mean-normalized lightcurve losses are exactly scale-invariant,
    so this turns "touch z=+-1" into a hard constraint every forward pass
    instead of a weak soft-penalty suggestion."""
    z = vlist[:, 2]
    z_min, z_max = z.min(), z.max()
    s = 2.0 / (z_max - z_min).clamp(min=1e-12)
    dz = -1.0 - s * z_min
    v = vlist * s
    v = torch.cat([v[:, :2], (v[:, 2] + dz).unsqueeze(-1)], dim=-1)
    return v


def soft_radial_clamp(vlist: torch.Tensor, cylinder_radius: float, beta: float = 30.0) -> torch.Tensor:
    """Smoothly caps the xy (radial, around the z-axis) distance of every
    vertex at `cylinder_radius`, leaving z untouched -- same LogSumExp
    soft-min trick `soft_fit_z_range` uses for z, applied to the radius
    instead. Deliberately NOT a loss penalty: a penalty only discourages
    exceeding cylinder_radius (nothing stops gradient descent from doing
    it anyway if the photometric loss wants to badly enough -- which is
    exactly what was happening before this existed, see chat), this
    actually caps the value used in every forward pass.

    No ReLU-style dead zone either: for a vertex sitting well outside the
    cylinder, the output is (near-)cylinder_radius * (x,y)/r, whose
    gradient is well-defined and NONZERO -- it keeps pushing the vertex
    around the cylinder wall (its angle) rather than the gradient just
    vanishing the way a flat ReLU region would. The only place this smooth
    version differs from a true hard clamp is a small "give" right around
    r == cylinder_radius, controlled by `beta` (same knob, same meaning,
    as `soft_fit_z_range`'s).

    Use this (soft) during training; use `radial_clamp` (hard) at export
    time -- exactly mirroring `soft_fit_z_range`/`fit_z_range`.
    """
    r = torch.linalg.norm(vlist[:, :2], dim=-1)
    R = torch.as_tensor(cylinder_radius, dtype=vlist.dtype, device=vlist.device).expand_as(r)
    r_capped = -torch.logsumexp(-beta * torch.stack([r, R], dim=0), dim=0) / beta  # smooth min(r, R)
    scale = r_capped / r.clamp(min=1e-12)
    xy = vlist[:, :2] * scale.unsqueeze(-1)
    return torch.cat([xy, vlist[:, 2:3]], dim=-1)


def radial_clamp(vlist: torch.Tensor, cylinder_radius: float) -> torch.Tensor:
    """Hard version of `soft_radial_clamp`: every vertex with xy-distance
    > cylinder_radius is projected radially back onto the cylinder wall --
    angle (in the xy-plane) and z preserved exactly, radius capped exactly
    at cylinder_radius; vertices already inside are left untouched. This
    is a genuine geometric guarantee (the exported mesh really does fit
    inside the cylinder, not just "was discouraged from not fitting"),
    exact like `fit_z_range`'s touching guarantee, not a loss penalty.

    Differentiable almost everywhere: same reasoning as `soft_radial_clamp`
    (the projection itself, cylinder_radius*(x,y)/r, has a well-defined
    nonzero gradient for clamped points) -- the only non-smooth point is
    the single measure-zero boundary r == cylinder_radius, not a whole
    dead half-space the way a ReLU penalty's flat region would be.
    """
    r = torch.linalg.norm(vlist[:, :2], dim=-1)
    r_capped = r.clamp(max=cylinder_radius)
    scale = r_capped / r.clamp(min=1e-12)
    xy = vlist[:, :2] * scale.unsqueeze(-1)
    return torch.cat([xy, vlist[:, 2:3]], dim=-1)


def soft_omega_clamp(omega: torch.Tensor, omega_lo: float, omega_hi: float,
                      softness: float = 30.0) -> torch.Tensor:
    """Same idea as `soft_radial_clamp`, applied to the rotation-rate
    scalar `omega` instead of per-vertex xy radius -- smoothly caps omega
    into [omega_lo, omega_hi] (omega_lo < omega_hi; recall omega is
    negative here, so omega_lo is the MORE negative bound, corresponding
    to the SHORTER allowed period) via the same LogSumExp soft-min/max
    trick, so a value pushed past either bound still gets a well-defined
    nonzero gradient pulling it back, rather than a dead flat region.

    `beta` (the softness knob) can't reuse `soft_radial_clamp`'s literal
    default of 30 -- that value is calibrated for xy coordinates of order
    ~1, while omega is of order 1e-3 (2*pi / a period of hundreds of
    frames), so a fixed beta=30 would be almost perfectly flat over
    omega's entire realistic range, not a clamp at all. Instead `beta` is
    derived from the bound spacing itself: beta = softness / (omega_hi -
    omega_lo), so the "give" near either edge is always about
    1/softness of the allowed span, regardless of omega's absolute scale
    -- matching the RELATIVE softness soft_radial_clamp's beta=30 gives
    relative to cylinder_radius (itself of order ~1).

    Use this (soft) during training; use `omega_clamp` (hard) once, when
    the final value is exported/checkpointed -- exactly mirroring
    `soft_radial_clamp`/`radial_clamp`.
    """
    lo = torch.as_tensor(omega_lo, dtype=omega.dtype, device=omega.device)
    hi = torch.as_tensor(omega_hi, dtype=omega.dtype, device=omega.device)
    beta = softness / max(float(hi - lo), 1e-12)
    capped_hi = -torch.logsumexp(-beta * torch.stack([omega, hi]), dim=0) / beta   # smooth min(omega, hi)
    capped = torch.logsumexp(beta * torch.stack([capped_hi, lo]), dim=0) / beta    # smooth max(., lo)
    return capped


def omega_clamp(omega: torch.Tensor, omega_lo: float, omega_hi: float) -> torch.Tensor:
    """Hard version of `soft_omega_clamp`: exact clamp into [omega_lo, omega_hi]."""
    return omega.clamp(min=omega_lo, max=omega_hi)


def orient_to_first_frame(vlist: torch.Tensor, omega: torch.Tensor,
                           omega0: torch.Tensor, t0: torch.Tensor) -> torch.Tensor:
    """Pure rotation (about z) to pose the mesh as it appeared at the
    lightcurve's first frame. Export-time only."""
    dtype, device = vlist.dtype, vlist.device
    t0 = torch.as_tensor(t0, dtype=dtype, device=device).reshape(1)
    M = rotate_z(omega, omega0, t0)[0]
    return vlist @ M


def normalize_to_challenge_frame(vlist: torch.Tensor, omega: torch.Tensor,
                                  omega0: torch.Tensor, t0: torch.Tensor,
                                  cylinder_radius: float = None
                                  ) -> torch.Tensor:
    """Returns (K,3) vertices in the required frame: z-axis = rotation axis
    (automatic), oriented to match the lightcurve's first frame,
    isotropically rescaled + z-shifted so the shape exactly touches
    z=-1 and z=+1.

    If `cylinder_radius` is given, also applies the hard `radial_clamp`
    as a final step (must come after the z-fit's isotropic rescale, since
    that rescale changes xy-radius too). This is the exact/hard version --
    every exported STL will genuinely fit inside the cylinder, not just
    have been discouraged from not fitting during training."""
    v = fit_z_range(orient_to_first_frame(vlist, omega, omega0, t0))
    if cylinder_radius is not None:
        v = radial_clamp(v, cylinder_radius)
    return v


def export_reconstruction(vlist: torch.Tensor, faces: torch.Tensor,
                           omega: torch.Tensor, omega0: torch.Tensor,
                           t0: torch.Tensor, path, cylinder_radius: float = None):
    v_final = normalize_to_challenge_frame(vlist, omega, omega0, t0, cylinder_radius)
    write_stl_binary(path, v_final, faces)
    return v_final


# ---------------------------------------------------------------------
# Voxel-IoU (Dice) shape-similarity metric
# ---------------------------------------------------------------------
#
#     score = 1 - (#(A\B) + #(B\A)) / (#(A) + #(B))  ==  2*#(A∩B)/(#A+#B)
#
# A = voxels of the true shape, B = voxels of the reconstruction, both on
# the SAME regular voxel grid. Solid voxelization via trimesh's
# `voxelized(pitch).fill()` (rasterize + flood-fill), much more
# memory-stable than point-sampling with `mesh.contains()`.
#
# Grid alignment: trimesh snaps each mesh's voxel grid origin to the
# nearest lower multiple of `pitch` relative to the global origin, so two
# meshes voxelized at the same pitch land on the same infinite lattice and
# only need an integer-voxel shift to compare directly.

def _load_and_repair(path):
    """Load an STL as a single mesh and attempt basic watertight repair
    (small holes, degenerate/duplicate faces, bad normals) -- solid
    voxelization needs a closed, consistently-oriented surface."""
    mesh = trimesh.load(path, force='mesh')

    if not mesh.is_watertight:
        print(f"Warning: mesh {path} is not watertight, attempting repair...")
        try:
            mesh.process(validate=True)
        except Exception:
            pass
        try:
            trimesh.repair.fill_holes(mesh)
        except Exception:
            pass
        try:
            mesh.remove_degenerate_faces()
        except AttributeError:
            try:
                mesh.update_faces(mesh.nondegenerate_faces())
            except Exception:
                pass
        except Exception:
            pass
        try:
            mesh.remove_duplicate_faces()
        except AttributeError:
            try:
                mesh.update_faces(mesh.unique_faces())
            except Exception:
                pass
        except Exception:
            pass
        try:
            trimesh.repair.fix_normals(mesh)
        except Exception:
            pass

    return mesh


def _voxelize_solid(mesh, pitch):
    """Return (bool 3D array, integer grid-origin index) for a solid mesh."""
    vg = mesh.voxelized(pitch=pitch).fill()
    matrix = np.asarray(vg.matrix, dtype=bool)
    origin_idx = np.round(vg.translation / pitch).astype(np.int64)
    return matrix, origin_idx


@functools.lru_cache(maxsize=8)
def _voxelize_gt_cached(path: str, voxel_size: float):
    """Ground truth never changes within a run, but `validate_and_log`
    calls this metric ~every `val_cfg['every']` steps -- without caching,
    that means reloading and re-voxelizing the (potentially large) GT mesh
    from disk every single validation, purely wasted repeated work. Cache
    key is (path, voxel_size); recon is intentionally NOT cached since it
    changes every call."""
    mesh = _load_and_repair(path)
    return _voxelize_solid(mesh, voxel_size)


def voxel_shape_metric(true_stl_path, recon_stl_path, voxel_size,
                        return_counts=False):
    """
    Compute the voxel-based similarity metric between a true asteroid
    shape and its lightcurve-inversion reconstruction. Both STL files are
    assumed already expressed in the shared canonical frame; this performs
    no additional registration/rotation/rescaling.
    """
    mat_true, origin_true = _voxelize_gt_cached(str(true_stl_path), float(voxel_size))
    recon_mesh = _load_and_repair(recon_stl_path)
    mat_recon, origin_recon = _voxelize_solid(recon_mesh, voxel_size)

    end_true = origin_true + np.array(mat_true.shape)
    end_recon = origin_recon + np.array(mat_recon.shape)

    combined_origin = np.minimum(origin_true, origin_recon)
    combined_end = np.maximum(end_true, end_recon)
    combined_shape = tuple((combined_end - combined_origin).astype(int))

    A = np.zeros(combined_shape, dtype=bool)
    B = np.zeros(combined_shape, dtype=bool)

    off_true = (origin_true - combined_origin).astype(int)
    off_recon = (origin_recon - combined_origin).astype(int)

    A[off_true[0]:off_true[0] + mat_true.shape[0],
      off_true[1]:off_true[1] + mat_true.shape[1],
      off_true[2]:off_true[2] + mat_true.shape[2]] = mat_true

    B[off_recon[0]:off_recon[0] + mat_recon.shape[0],
      off_recon[1]:off_recon[1] + mat_recon.shape[1],
      off_recon[2]:off_recon[2] + mat_recon.shape[2]] = mat_recon

    n_A = int(A.sum())
    n_B = int(B.sum())
    n_A_and_B = int(np.logical_and(A, B).sum())
    n_A_minus_B = int(np.logical_and(A, ~B).sum())
    n_B_minus_A = int(np.logical_and(~A, B).sum())

    denom = n_A + n_B
    if denom == 0:
        score = 1.0
    else:
        score = 1.0 - (n_A_minus_B + n_B_minus_A) / denom

    if return_counts:
        dice_check = (2 * n_A_and_B / denom) if denom > 0 else 1.0
        counts = {
            'A': n_A,
            'B': n_B,
            'A_and_B': n_A_and_B,
            'A_minus_B': n_A_minus_B,
            'B_minus_A': n_B_minus_A,
            'score_dice_check': dice_check,
        }
        return score, counts

    return score


# ---------------------------------------------------------------------
# Validation hook
# ---------------------------------------------------------------------

def compute_voxel_metric(recon_stl_path: str, gt_stl_path: str, val_cfg: dict) -> float:
    return voxel_shape_metric(gt_stl_path, recon_stl_path, val_cfg.get("voxel_size", 0.05))


def compute_sideview_metric(recon_stl_path: str, gt_stl_path: str, val_cfg: dict) -> float:
    """Same (recon, gt, val_cfg) -> float-in-[0,1] signature as
    `compute_voxel_metric`, so either can be dropped into
    `validate_and_log`'s `metric_fn`. See `sideview.py` for the actual
    implementation and the assumptions it documents (fixed viewing
    directions, our own distance-to-score mapping -- the organizers'
    exact side-view formula isn't published)."""
    return sideview_score(
        recon_stl_path, gt_stl_path,
        n_directions=val_cfg.get("sideview_n_directions", 8),
        resolution=val_cfg.get("sideview_resolution", 256),
        window_half_size=val_cfg.get("sideview_window_half_size", 2.0),
        length_scale=val_cfg.get("sideview_length_scale", 1.0),
    )


def compute_combined_metric(recon_stl_path: str, gt_stl_path: str, val_cfg: dict) -> float:
    """voxel_score + sideview_score, each in [0, 1] -> combined in [0, 2].
    This is what `search.py` optimizes: a single number per (config,
    object) averaging how well the reconstruction matches GT under both
    the 3D-overlap view (voxel IoU) and the 2D-silhouette view (side-view),
    so a config can't win purely by exploiting one metric's blind spot."""
    v = compute_voxel_metric(recon_stl_path, gt_stl_path, val_cfg)
    s = compute_sideview_metric(recon_stl_path, gt_stl_path, val_cfg)
    return v + s


def maybe_convert_msh_to_stl(msh_path: str, stl_path: str):
    """One-off ground-truth prep: converts a Gmsh .msh file to .stl (needs
    `pip install trimesh meshio --break-system-packages`). Not run
    automatically -- a one-time step per ground-truth file, not per step."""
    mesh = trimesh.load(msh_path)
    mesh.export(stl_path)


def validate_and_log(hyper, model, get_mesh_fn, omega, omega0, t0,
                      gt_path: str, run_dir: Path, step: int,
                      val_cfg: dict = {},
                      metric_fn: Callable[[str, str], float] = compute_voxel_metric,
                      cylinder_radius: float = None,
                      ) -> Optional[float]:
    """Call periodically from the training loop. Exports the CURRENT mesh
    in the normalized challenge frame, evaluates metric_fn against
    gt_path, appends to run_dir/validation.csv. Never raises -- a metric
    failure is logged and training continues.

    `cylinder_radius`, if given, is passed to `export_reconstruction` so
    the exported STL gets the hard radial clamp (on top of whatever
    get_mesh_fn already applied internally, typically the soft version)."""
    val_dir = run_dir / "validation"
    val_dir.mkdir(exist_ok=True)
    recon_path = val_dir / f"recon_step{step:06d}.stl"

    try:
        tlist, vlist = get_mesh_fn(hyper, model)
        export_reconstruction(vlist, tlist, omega, omega0, t0, recon_path, cylinder_radius)
        score = metric_fn(str(recon_path), gt_path, val_cfg)
    except Exception as e:
        print(f"[validation] step {step}: FAILED ({e!r}) -- continuing training")
        score = None

    log_path = run_dir / "validation.csv"
    write_header = not log_path.exists()
    with open(log_path, "a") as f:
        if write_header:
            f.write("step,score,recon_path\n")
        f.write(f"{step},{'' if score is None else score},{recon_path}\n")

    if score is not None:
        print(f"[validation] step {step}: score={score:.4f}")
    return score
