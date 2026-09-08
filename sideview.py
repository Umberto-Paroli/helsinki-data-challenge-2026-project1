"""
sideview.py
===========

"Side-view" shape-similarity metric: project both the reconstruction and
the ground truth onto 2D planes along a FIXED set of viewing directions,
extract each projection's boundary (silhouette outline), and measure the
distance between the two boundary curves -- this is our own implementation
of the metric described for the challenge ("Look at 2D projections along
unspecified directions. We calculate the distance between two boundary
curves."). The organizers' exact viewing directions and exact distance
formula aren't published, so both are choices made here, clearly isolated
in `DEFAULT_DIRECTIONS` and `_distance_to_score` below so they're easy to
revisit once/if more detail becomes available.

Direction choice: "side-view" (as opposed to a pole-on view) most
naturally means viewing roughly perpendicular to the rotation (z) axis --
matching how ground-based lightcurve photometry actually observes these
objects. `DEFAULT_DIRECTIONS` is `n_directions` equatorial directions
(elevation ~0, i.e. lying in the xy-plane), evenly spaced in azimuth.

Distance choice: for each direction, rasterize both meshes' silhouettes
onto a shared pixel grid (same window/resolution, so the two grids are
directly comparable pixel-for-pixel), extract each silhouette's boundary
pixels, and compute the symmetric mean nearest-boundary-pixel distance
(a 2D boundary/Chamfer-style distance, via `cv2.distanceTransform` for
speed -- no shapely/skimage dependency needed). Averaged over all
directions, then mapped through `_distance_to_score` (monotonic decay,
1.0 for a perfect match) to land in [0, 1] -- summable with
`validation.compute_voxel_metric`'s own [0, 1] score for a combined
[0, 2] number (see `validation.compute_combined_metric`).
"""
from __future__ import annotations
import functools
from typing import Optional, Sequence, Tuple

import numpy as np
import cv2
import trimesh


# ---------------------------------------------------------------------
# viewing directions
# ---------------------------------------------------------------------

def default_sideview_directions(n_directions: int = 8) -> np.ndarray:
    """n_directions unit vectors evenly spaced in azimuth, lying in the
    xy-plane (elevation 0 -- true "side" views, perpendicular to the
    rotation/z axis). Fixed/deterministic: same directions every call,
    since neither side (recon or GT) knows the organizers' real ones."""
    az = np.linspace(0.0, 2 * np.pi, n_directions, endpoint=False)
    return np.stack([np.cos(az), np.sin(az), np.zeros_like(az)], axis=1)


def _orthonormal_basis(direction: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (u, v): unit vectors spanning the plane perpendicular to
    `direction`. For an equatorial `direction` (in the xy-plane), this
    picks u = world z-axis and v = direction x u, i.e. the 2D projection's
    u-axis is literally the rotation axis and v-axis is the in-plane
    "depth-perpendicular" horizontal -- keeps the projected frame aligned
    with the physically meaningful axes instead of an arbitrary rotation."""
    d = direction / np.linalg.norm(direction)
    world_z = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(d, world_z)) > 0.999:  # direction ~parallel to z: fall back
        u = np.array([1.0, 0.0, 0.0])
    else:
        u = world_z - np.dot(world_z, d) * d
        u = u / np.linalg.norm(u)
    v = np.cross(d, u)
    v = v / np.linalg.norm(v)
    return u, v


def _project_2d(vertices: np.ndarray, direction: np.ndarray) -> np.ndarray:
    u, v = _orthonormal_basis(direction)
    return np.stack([vertices @ u, vertices @ v], axis=1)


# ---------------------------------------------------------------------
# rasterization + boundary distance
# ---------------------------------------------------------------------

def _rasterize_silhouette(verts2d: np.ndarray, faces: np.ndarray,
                           window_half_size: float, resolution: int) -> np.ndarray:
    """Fills every projected triangle into a shared `resolution x
    resolution` pixel grid covering [-window_half_size, window_half_size]^2
    -- a FIXED window (not each mesh's own bounding box), so two masks
    built from the same window/resolution are directly comparable
    pixel-for-pixel. Overlapping/occluding triangles are fine: we only
    want the union's outer boundary, not per-triangle visibility."""
    scale = resolution / (2.0 * window_half_size)
    px = ((verts2d + window_half_size) * scale).astype(np.int32)
    mask = np.zeros((resolution, resolution), dtype=np.uint8)
    tris = px[faces]  # (nfac, 3, 2)
    cv2.fillPoly(mask, tris, 1)
    return mask


def _boundary_pixels(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(mask, kernel)
    return (mask.astype(bool)) & (~eroded.astype(bool))


def _symmetric_boundary_distance_px(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Mean symmetric boundary-to-boundary distance, in PIXELS. Empty
    silhouette on either side (degenerate mesh) is treated as maximally
    wrong (a large fixed penalty) rather than raising, since this needs
    to run unattended for hundreds of search trials."""
    bnd_a = _boundary_pixels(mask_a)
    bnd_b = _boundary_pixels(mask_b)
    if not bnd_a.any() or not bnd_b.any():
        return float(mask_a.shape[0])  # ~ full image span: maximally bad

    # distance_to_b[y, x] = distance from pixel (y, x) to the nearest
    # foreground pixel of mask_b (cv2.distanceTransform measures distance
    # to the nearest ZERO pixel, so invert mask_b first).
    dist_to_b = cv2.distanceTransform((1 - mask_b).astype(np.uint8), cv2.DIST_L2, 5)
    dist_to_a = cv2.distanceTransform((1 - mask_a).astype(np.uint8), cv2.DIST_L2, 5)

    d_a_to_b = dist_to_b[bnd_a].mean()
    d_b_to_a = dist_to_a[bnd_b].mean()
    return float(0.5 * (d_a_to_b + d_b_to_a))


def _distance_to_score(distance: float, length_scale: float = 1.0) -> float:
    """Monotonic decay, 1.0 at distance=0, ~0.37 at distance=length_scale
    -- exponential rather than a hard linear clip so trials that are
    "very wrong" in different degrees still produce distinguishable
    (if all low) scores instead of all bottoming out at exactly 0."""
    return float(np.exp(-distance / max(length_scale, 1e-9)))


# ---------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------

@functools.lru_cache(maxsize=8)
def _load_gt_cached(path: str) -> trimesh.Trimesh:
    """GT never changes within a run/search -- avoid re-reading a
    potentially large STL from disk on every single trial's evaluation
    (search.py evaluates this hundreds of times over a multi-day run)."""
    return trimesh.load(path, force='mesh')


def _load_mesh(path_or_mesh, is_gt: bool = False):
    if isinstance(path_or_mesh, trimesh.Trimesh):
        return path_or_mesh
    return _load_gt_cached(str(path_or_mesh)) if is_gt else trimesh.load(path_or_mesh, force='mesh')


def sideview_score(recon_path, gt_path,
                    directions: Optional[Sequence[np.ndarray]] = None,
                    n_directions: int = 8,
                    window_half_size: float = 2.0,
                    resolution: int = 512,
                    length_scale: float = 1.0,
                    return_per_direction: bool = False):
    """Side-view silhouette-boundary-distance score in [0, 1], averaged
    over `directions` (default: `default_sideview_directions(n_directions)`,
    fixed equatorial views -- see module docstring).

    Both meshes are assumed already in the shared canonical frame (z in
    [-1, 1], same convention `validation.compute_voxel_metric` assumes) --
    no additional registration is performed here. `window_half_size`
    should comfortably cover both meshes' projected extent for every
    direction (default 2.0: generous margin over any of this project's
    cylinder_radius values plus the z half-height of 1.0).
    """
    recon = _load_mesh(recon_path)
    gt = _load_mesh(gt_path, is_gt=True)
    if directions is None:
        directions = default_sideview_directions(n_directions)

    scores = []
    for d in directions:
        v_r = _project_2d(np.asarray(recon.vertices), d)
        v_g = _project_2d(np.asarray(gt.vertices), d)
        mask_r = _rasterize_silhouette(v_r, recon.faces, window_half_size, resolution)
        mask_g = _rasterize_silhouette(v_g, gt.faces, window_half_size, resolution)
        dist_px = _symmetric_boundary_distance_px(mask_r, mask_g)
        dist_units = dist_px * (2.0 * window_half_size / resolution)  # px -> physical units
        scores.append(_distance_to_score(dist_units, length_scale))

    scores = np.array(scores)
    if return_per_direction:
        return float(scores.mean()), scores
    return float(scores.mean())
