"""
hypernets.py
============

DIP-style reparametrizations of the shape produced for `geometry.py`'s
forward operator. Nothing about the forward model is touched -- every
variant here only changes WHERE the mesh comes from (a network's output
instead of a bare `nn.Parameter`).

Consolidated from the former `dip_wrapper.py` + `grid_hypernets.py` (kept
as one file: both are just a family of interchangeable generators selected
by `train.build_hypernet` via `cfg["model"]["hypernet"]`).

  CoefficientHyperNet     ("flat")   -- flat MLP -> octantoid SH coefficients
  SphereCoordHyperNet     ("sphere") -- coordinate net on the octantoid's own
                                        grid, projected onto the SH basis
  GraphConvHyperNet       ("graph")  -- graph-conv on the octantoid's mesh
                                        connectivity, isotropic radius only
  SphereNoiseGraphHyperNet("variant_D1") -- same graph-conv, but literal
                                        fixed-noise DIP input instead of
                                        vertex coordinates
  PlanarDiskHyperNet                 -- polar-disk / dual-height-field
                                        generator (not currently wired into
                                        build_hypernet's dispatch, kept
                                        available for experimentation)

`octantoid_directions()` factors out the per-vertex unit-direction
computation that all three grid-based variants (SphereCoordHyperNet,
GraphConvHyperNet, SphereNoiseGraphHyperNet) used to repeat identically --
also used by `train.py` to read a hypernet's `.directions` grid for the
r0-ellipsoid warm start WITHOUT building a full (throwaway) hypernet
instance just to get it.
"""
from __future__ import annotations
import math
from typing import Optional

import torch
import torch.nn as nn

from geometry import _OctantoidGrid, K, octantoid_directions


# ---------------------------------------------------------------------
# Variant A ("flat"): flat MLP -> coefficient vector directly
# ---------------------------------------------------------------------

class CoefficientHyperNet(nn.Module):
    """
    z (fixed, frozen) -> MLP -> a  (3*(LMAX+1)^2,)

    Degree-decaying output scale: coefficient (l, m) for axis-channel c is
    scaled by ~ 1/(l+1) at initialization, so the network starts closer to
    "low order dominates" without hard-forcing it.
    """

    def __init__(self, LMAX: int, latent_dim: int = 16, hidden: int = 64,
                 r0: float = 1.0, dtype=torch.float64):
        super().__init__()
        self.LMAX = LMAX
        self.n_coeff = (LMAX + 1) ** 2
        self.out_dim = 3 * self.n_coeff

        self.z = nn.Parameter(torch.randn(latent_dim, dtype=dtype), requires_grad=False)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden, dtype=dtype), nn.Tanh(),
            nn.Linear(hidden, hidden, dtype=dtype), nn.Tanh(),
            nn.Linear(hidden, self.out_dim, dtype=dtype),
        )
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)  # start close to the a=0 (sphere of radius r0) state
            self.net[-1].bias.zero_()

        degree_scale = torch.tensor(
            [1.0 / (l + 1.0) for l in range(LMAX + 1) for _m in range(-l, l + 1)],
            dtype=dtype)
        self.register_buffer("decay", degree_scale.repeat(3))  # (3*n_coeff,)

        self.register_buffer(
            "bias0",
            torch.zeros(self.out_dim, dtype=dtype).index_fill_(
                0, torch.tensor([0, self.n_coeff, 2 * self.n_coeff]),
                math.log(r0) / K(0, 0)))

    def forward(self) -> torch.Tensor:
        raw = self.net(self.z)
        return self.bias0 + self.decay * raw


# ---------------------------------------------------------------------
# Variant B ("sphere"): coordinate network over the octantoid's own
# (theta,phi) grid, projected onto the SH basis via a fixed forward
# transform
# ---------------------------------------------------------------------

class _Sine(nn.Module):
    def __init__(self, omega_0):
        super().__init__()
        self.omega_0 = omega_0

    def forward(self, x):
        return torch.sin(self.omega_0 * x)


class SphereCoordHyperNet(nn.Module):
    """
    directions_k (fixed, = grid vertices) -> shared MLP -> raw_k (3,)
    then  a_axis = pinv(B) @ raw[:, axis]   for axis in {x,y,z}
    """

    def __init__(self, LMAX: int, nrows: int, hidden: int = 64,
                 hidden_layers: int = 3, omega_0: float = 10.0,
                 r0: float = 1.0, dtype=torch.float64, device=None):
        super().__init__()
        grid = _OctantoidGrid(LMAX, nrows, dtype=dtype, device=device)
        self.LMAX = LMAX
        self.n_coeff = (LMAX + 1) ** 2
        self.register_buffer("theta", grid.theta)
        self.register_buffer("phi", grid.phi)
        self.register_buffer("B_pinv", torch.linalg.pinv(grid.B))  # (n_coeff, nvert)
        self.register_buffer("directions",
                              octantoid_directions(LMAX, nrows, dtype=dtype, device=device))

        layers = []
        in_f = 3
        for i in range(hidden_layers):
            layers.append(nn.Linear(in_f, hidden, dtype=dtype))
            layers.append(_Sine(omega_0))
            in_f = hidden
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, 3, dtype=dtype)  # 3 raw channels: x,y,z log-scale
        with torch.no_grad():
            self.head.weight.mul_(0.01)
            self.head.bias.zero_()

        self.register_buffer(
            "bias0",
            torch.zeros(3 * self.n_coeff, dtype=dtype).index_fill_(
                0, torch.tensor([0, self.n_coeff, 2 * self.n_coeff]),
                math.log(r0) / K(0, 0)))

    def forward(self) -> torch.Tensor:
        raw = self.head(self.trunk(self.directions))          # (nvert, 3)
        a_x = self.B_pinv @ raw[:, 0]                          # (n_coeff,)
        a_y = self.B_pinv @ raw[:, 1]
        a_z = self.B_pinv @ raw[:, 2]
        return self.bias0 + torch.cat([a_x, a_y, a_z], dim=0)


# ---------------------------------------------------------------------
# graph-conv building blocks, shared by GraphConvHyperNet and
# SphereNoiseGraphHyperNet below
# ---------------------------------------------------------------------

def _build_mesh_adjacency(tlist: torch.Tensor, nvert: int, dtype, device) -> torch.Tensor:
    """(nvert, nvert) row-normalized adjacency (incl. self-loops), built from
    triangle edges. Dense here since these grids are small (~100s of
    vertices)."""
    A = torch.zeros(nvert, nvert, dtype=dtype, device=device)
    edges = torch.cat([
        tlist[:, [0, 1]], tlist[:, [1, 2]], tlist[:, [2, 0]],
        tlist[:, [1, 0]], tlist[:, [2, 1]], tlist[:, [0, 2]],
    ], dim=0)
    A[edges[:, 0], edges[:, 1]] = 1.0
    A[torch.arange(nvert), torch.arange(nvert)] = 1.0  # self-loop
    A = A / A.sum(dim=-1, keepdim=True).clamp(min=1.0)
    return A


def _build_multihop_adjacency(tlist: torch.Tensor, nvert: int, dtype, device,
                               k_hop: int = 3, weights=None) -> torch.Tensor:
    """Atilde = sum_{k=0}^{k_hop} w_k * A^k. Widens the receptive field in
    one mixing step; re-normalized so rows still sum to 1."""
    A1 = _build_mesh_adjacency(tlist, nvert, dtype, device)
    if weights is None:
        weights = [1.0 / (k + 1) for k in range(k_hop + 1)]

    Ak = torch.eye(nvert, dtype=dtype, device=device)  # A^0
    A_multi = weights[0] * Ak
    for k in range(1, k_hop + 1):
        Ak = Ak @ A1
        A_multi = A_multi + weights[k] * Ak

    return A_multi / A_multi.sum(dim=-1, keepdim=True).clamp(min=1e-12)


# ---------------------------------------------------------------------
# ellipsoid warm-start estimation (used by train.py to pick r0 for the
# chosen hypernet before the real optimization starts)
# ---------------------------------------------------------------------

def estimate_equatorial_axis_ratio(L_intensity: torch.Tensor) -> float:
    """L_intensity: (F, C) mean-normalized intensity curves (all channels).
    Rough estimate of equatorial elongation a/b from peak-to-peak flux
    ratio. Consistently OVERSHOOTS the true ratio -- use
    `estimate_ellipsoid_r0`'s `damping` knob to correct for it."""
    ratios = L_intensity.max(dim=0).values / L_intensity.min(dim=0).values.clamp(min=1e-6)
    return ratios.max().item()


def ellipsoid_radius(directions: torch.Tensor, a: float, b: float, c: float) -> torch.Tensor:
    """r(direction) for a triaxial ellipsoid with semi-axes a (x), b (y), c
    (z), evaluated at each row of `directions` (K,3), unit vectors."""
    dx, dy, dz = directions[:, 0], directions[:, 1], directions[:, 2]
    inv_r2 = (dx / a) ** 2 + (dy / b) ** 2 + (dz / c) ** 2
    return 1.0 / torch.sqrt(inv_r2.clamp(min=1e-12))


def estimate_ellipsoid_r0(directions: torch.Tensor, L_intensity: torch.Tensor,
                           cylinder_radius: float, damping: float = 0.5) -> torch.Tensor:
    """Per-vertex ellipsoid baseline radius (K,), used as `r0` for the
    graph-based hypernets instead of a scalar -- a warm start closer to the
    true shape's gross elongation than a sphere. `damping` in [0,1] shrinks
    the estimated log-ratio to compensate for the estimator's measured
    overshoot."""
    raw_ratio = estimate_equatorial_axis_ratio(L_intensity)
    corrected_ratio = math.exp(damping * math.log(raw_ratio))  # damp in log-space
    a, b, c = cylinder_radius, cylinder_radius / corrected_ratio, 1.0
    return ellipsoid_radius(directions, a, b, c)


# ---------------------------------------------------------------------
# Variant C ("graph"): graph convolution directly on the octantoid's own
# mesh connectivity -- isotropic radius only, bypasses SH coefficients
# entirely
# ---------------------------------------------------------------------

class GraphConvHyperNet(nn.Module):
    """
    directions_k (fixed grid vertices) -> [Linear -> mesh-adjacency mixing
    -> nonlinearity] x n_layers -> radius_k > 0

    Returns (tlist, vlist) directly; does not go through octantoid_to_trimesh
    / SH coefficients at all, so smoothness comes entirely from the graph
    structure and depth.
    """

    def __init__(self, LMAX_grid_only: int, nrows: int, hidden: int = 64,
                 n_layers: int = 3, r0: float = 1.0, k_hop: int = 1,
                 dtype=torch.float64, device=None):
        super().__init__()
        grid = _OctantoidGrid(LMAX_grid_only, nrows, dtype=dtype, device=device)
        self.register_buffer("tlist", grid.tlist)
        directions = octantoid_directions(LMAX_grid_only, nrows, dtype=dtype, device=device)
        self.register_buffer("directions", directions)  # (nvert, 3)
        if k_hop > 1:
            A = _build_multihop_adjacency(grid.tlist, directions.shape[0], dtype, device, k_hop=k_hop)
        else:
            A = _build_mesh_adjacency(grid.tlist, directions.shape[0], dtype, device)
        self.register_buffer("A", A)
        if isinstance(r0, torch.Tensor):
            self.register_buffer("r0", r0.to(dtype=dtype, device=device))
        else:
            self.r0 = r0

        dims = [3] + [hidden] * (n_layers - 1) + [1]
        self.layers = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1], dtype=dtype) for i in range(len(dims) - 1)
        ])
        with torch.no_grad():
            self.layers[-1].weight.mul_(0.01)
            self.layers[-1].bias.zero_()

    def forward(self):
        h = self.directions
        for i, layer in enumerate(self.layers):
            h = layer(h)
            h = self.A @ h  # mix with immediate mesh-neighbors (the "conv" step)
            if i < len(self.layers) - 1:
                h = torch.tanh(h)
        delta = h.squeeze(-1)
        # radius = r0 * exp(delta): delta=0 -> radius=r0 exactly
        radius = self.r0 * torch.exp(delta)
        vlist = radius.unsqueeze(-1) * self.directions
        return self.tlist, vlist


# ---------------------------------------------------------------------
# Variant D1 ("variant_D1"): same graph-conv machinery, literal fixed-noise
# DIP input instead of vertex coordinates
# ---------------------------------------------------------------------

class SphereNoiseGraphHyperNet(nn.Module):
    """
    z_k (FIXED random noise, shape (K, noise_dim), sampled once and frozen --
    the network never sees the vertex coordinates at all) -> shared
    [Linear -> adjacency mixing -> nonlinearity] x n_layers -> radius_k > 0
    -> vlist_k = radius_k * direction_k
    """

    def __init__(self, LMAX_grid_only: int, nrows: int, noise_dim: int = 8,
                 hidden: int = 64, n_layers: int = 3, r0: float = 1.0, k_hop: int = 1,
                 dtype=torch.float64, device=None):
        super().__init__()
        grid = _OctantoidGrid(LMAX_grid_only, nrows, dtype=dtype, device=device)
        self.register_buffer("tlist", grid.tlist)
        directions = octantoid_directions(LMAX_grid_only, nrows, dtype=dtype, device=device)
        self.register_buffer("directions", directions)  # (K,3), used only for output placement
        Kv = directions.shape[0]
        self.register_buffer("noise", torch.randn(Kv, noise_dim, dtype=dtype, device=device))
        if k_hop > 1:
            A = _build_multihop_adjacency(grid.tlist, directions.shape[0], dtype, device, k_hop=k_hop)
        else:
            A = _build_mesh_adjacency(grid.tlist, directions.shape[0], dtype, device)
        self.register_buffer("A", A)
        self.r0 = r0

        dims = [noise_dim] + [hidden] * (n_layers - 1) + [1]
        self.layers = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1], dtype=dtype) for i in range(len(dims) - 1)
        ])
        with torch.no_grad():
            self.layers[-1].weight.mul_(0.1)
            self.layers[-1].bias.zero_()

    def forward(self):
        h = self.noise
        for i, layer in enumerate(self.layers):
            h = layer(h)
            h = self.A @ h
            if i < len(self.layers) - 1:
                h = torch.tanh(h)
        delta = h.squeeze(-1)
        radius = self.r0 * torch.exp(delta)  # strictly positive, no need for softplus
        vlist = radius.unsqueeze(-1) * self.directions
        return self.tlist, vlist


# ---------------------------------------------------------------------
# Convex-hull stage generator: a free point cloud, NOT a hypernetwork
# ---------------------------------------------------------------------

class ConvexPointCloud(nn.Module):
    """
    Direct (x, y, z) parametrization of K points in R^3 -- deliberately NOT
    a neural network: every point is its own free `nn.Parameter` entry, no
    shared weights, no smoothness prior at all. Meant to be passed through
    `geometry.convex_hull_mesh` before rendering, which always returns a
    convex polytope no matter where the points sit -- convexity comes
    entirely from that hull step, not from anything about this class.

    This is deliberately more general than the radius-only generators
    above (GraphConvHyperNet / SphereNoiseGraphHyperNet): those move each
    vertex only along ITS OWN fixed direction from the origin (radius is
    the only free number per vertex), so a vertex can never leave its
    assigned angular slot. A flat face meeting at a sharp corner (e.g. a
    cube) needs vertices free to reposition off their initial direction to
    become exactly coplanar/collinear with their neighbors on the hull --
    something a purely radial function can never do, no matter how it's
    parametrized (spherical harmonics included), and something free point
    coordinates can.

    Points not on the hull at a given step get no gradient that step (a
    convex hull is locally constant under small moves of an interior
    point) -- and unlike the weight-sharing hypernets above, there's no
    indirect path for gradient to reach an interior point either, since
    every point is fully independent here. A point that becomes interior
    early can, in principle, stay permanently "dead" for the rest of
    training. Initializing every point ON the warm-start convex surface
    (`r0 * directions` -- a convex surface, so ~all points start active)
    is the mitigation used by `train_convex.py`; if in practice too many
    points still go dead too early, see `ConvexOffsetNet` below for a
    version with weight-sharing (but still no neighbor-smoothing) instead.

    Dual use: this same class is also how `train.py` warm-starts the
    non-convex refinement stage FROM a finished convex-stage checkpoint --
    pass the checkpoint's saved point positions directly as
    `directions_init` with `r0=1.0` (so `init = directions_init * 1.0`
    reproduces those positions exactly), rather than `octantoid_directions(
    ...) * ellipsoid_r0`. The class doesn't care either way; `points * 1.0`
    from arbitrary positions and `directions * r0` from a unit sphere are
    the same operation.

    `fixed_tlist`, if given, is a triangle-index tensor stored as a buffer
    and reported back to callers (see `train.get_mesh`) instead of the
    octantoid grid's own `model.tlist`. This matters specifically for the
    warm-start case above: `model.tlist` assumes point i stays close to
    its ORIGINAL octantoid grid direction (that's how the grid's face
    indices were built), which free points from a finished convex stage
    do not respect -- many end up far from their starting direction, and
    some (the ones that never made it onto the stage-1 hull, see
    `geometry.convex_hull_mesh`'s docstring) collapse toward the interior.
    Pairing THOSE with `model.tlist`'s fixed connectivity produces wildly
    stretched, near-degenerate triangles connecting interior points to
    faraway surface ones -- harmless-looking but numerically disastrous:
    it inflates the self-shadowing test's candidate count by ~100x+ (see
    chat), which is what was causing CUDA OOM on the non-convex refinement
    even at small nrows. Passing `fixed_tlist=geometry.convex_hull_mesh(
    checkpoint_points)[0]` instead gives a triangulation that reflects
    where the points ACTUALLY are: hull vertices get real, sane faces;
    points that never made the hull simply aren't referenced by any face
    (same "dead point" outcome they already had in stage 1) instead of
    being forced into a bogus one. Computed ONCE at warm-start time, not
    every step -- from then on it's a fixed topology like every other
    representation in this file, vertices still fully free to move
    (including developing concavities, since it's never recomputed).
    """

    def __init__(self, directions_init: torch.Tensor, r0, dtype=torch.float64,
                 fixed_tlist: Optional[torch.Tensor] = None):
        super().__init__()
        device = directions_init.device
        if isinstance(r0, torch.Tensor):
            r0v = r0.to(dtype=dtype, device=device)
        else:
            r0v = torch.full((directions_init.shape[0],), float(r0), dtype=dtype, device=device)
        init = directions_init.to(dtype=dtype) * r0v.unsqueeze(-1)
        self.points = nn.Parameter(init.clone())
        self.register_buffer(
            "fixed_tlist",
            None if fixed_tlist is None else fixed_tlist.to(dtype=torch.long, device=device))

    def forward(self) -> torch.Tensor:
        return self.points


class ConvexOffsetNet(nn.Module):
    """
    Same role as `ConvexPointCloud` (K free 3D points -> pass through
    `geometry.convex_hull_mesh`), but every point's OFFSET from its initial
    position comes from a small MLP SHARED across all K points (fixed
    per-point noise -> shared weights -> (dx,dy,dz)), the same
    weight-sharing pattern as `SphereNoiseGraphHyperNet` -- but with NO
    adjacency/neighbor-mixing step at all. That's deliberate: mixing a
    point's signal with its neighbors is explicitly a smoothing operation,
    which directly fights forming a flat face / sharp edge (the entire
    reason to use a point cloud here instead of a radius field -- see
    `ConvexPointCloud`'s docstring). Weight-sharing and neighbor-smoothing
    are independent ideas even though the older hypernets bundle them
    together; this class keeps only the first one.

    Why bother over plain `ConvexPointCloud`: a point that ends up INSIDE
    the hull (not on it) gets zero gradient that step from the render loss
    -- with `ConvexPointCloud`'s fully independent parameters, that point
    is then permanently stuck wherever it was. Here every point's output
    passes through the same shared weights, so gradient from the ACTIVE
    (on-hull) points still updates those weights every step, which changes
    EVERY point's predicted offset, including currently-inactive ones --
    a point can "come back to life" later in training as the hull
    reshapes, instead of being frozen at whatever it was doing when it
    first went inactive.

    Trade-off: sharing weights also couples points together somewhat (an
    update pulls on all of them a little, not just the active ones) and
    adds a couple of network hyperparameters (`hidden`, `n_layers`,
    `noise_dim`) to pick, versus `ConvexPointCloud`'s zero hyperparameters.
    Try both (`config_convex.yaml`'s `model.generator: points` vs
    `shared_mlp`) and compare -- there's no a priori answer for which wins
    on a given shape/dataset.
    """

    def __init__(self, init_positions: torch.Tensor, noise_dim: int = 8,
                 hidden: int = 64, n_layers: int = 3, dtype=torch.float64,
                 fixed_tlist: Optional[torch.Tensor] = None):
        super().__init__()
        device = init_positions.device
        K = init_positions.shape[0]
        self.register_buffer("init_positions", init_positions.to(dtype=dtype))
        self.register_buffer("noise", torch.randn(K, noise_dim, dtype=dtype, device=device))
        # see ConvexPointCloud's docstring -- same warm-start topology fix,
        # not currently used by any wired-up config (only ConvexPointCloud
        # is used for warmstart_points today) but supported for symmetry.
        self.register_buffer(
            "fixed_tlist",
            None if fixed_tlist is None else fixed_tlist.to(dtype=torch.long, device=device))

        dims = [noise_dim] + [hidden] * (n_layers - 1) + [3]
        layers = []
        in_f = dims[0]
        for out_f in dims[1:-1]:
            layers += [nn.Linear(in_f, out_f, dtype=dtype), nn.Tanh()]
            in_f = out_f
        layers.append(nn.Linear(in_f, dims[-1], dtype=dtype))
        self.net = nn.Sequential(*layers)
        with torch.no_grad():
            # start close to offset=0 -- i.e. right on the initial (warm-start) surface
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.zero_()

    def forward(self) -> torch.Tensor:
        offset = self.net(self.noise)
        return self.init_positions + offset


# ---------------------------------------------------------------------
# Planar disk / dual height-field generator (not wired into
# train.build_hypernet's dispatch by default -- kept available)
# ---------------------------------------------------------------------

def _build_polar_grid(n_rings: int, n_theta: int, radius: float, dtype, device):
    """
    Returns (interior_xy, rim_xy). Ring i sits at radius*(i+1)/n_rings;
    ring n_rings-1 IS the rim. Rings 0..n_rings-2 plus the center point are
    "interior".
    """
    center = torch.zeros(1, 2, dtype=dtype, device=device)
    rings_xy = []
    for i in range(n_rings):
        r = radius * (i + 1) / n_rings
        angles = torch.linspace(0, 2 * math.pi, n_theta + 1, dtype=dtype, device=device)[:-1]
        ring = torch.stack([r * torch.cos(angles), r * torch.sin(angles)], dim=-1)
        rings_xy.append(ring)

    interior_xy = torch.cat([center] + rings_xy[:-1], dim=0)  # center + rings 0..n_rings-2
    rim_xy = rings_xy[-1]                                      # ring n_rings-1
    return interior_xy, rim_xy


def _polar_topology(n_rings: int, n_theta: int):
    """
    Index layout (ring-major): center -> 0; ring i, angle j -> 1 + i*n_theta + j
    (i = 0..n_rings-2). Rim indices supplied separately by the caller.
    """
    def ring_start(i):  # i = 0 .. n_rings-2 (interior rings only)
        return 1 + i * n_theta

    n_interior_rings = n_rings - 1  # rings 0..n_rings-2
    n_interior = 1 + n_interior_rings * n_theta
    rim_start = n_interior  # rim indices are n_interior .. n_interior+n_theta-1 (local)

    faces = []
    if n_interior_rings > 0:
        r0 = ring_start(0)
    else:
        r0 = rim_start
    for j in range(n_theta):
        a, b = r0 + j, r0 + (j + 1) % n_theta
        faces.append((0, a, b))

    def add_strip(A, B):
        for j in range(n_theta):
            a0, a1 = A + j, A + (j + 1) % n_theta
            b0, b1 = B + j, B + (j + 1) % n_theta
            faces.append((a1, a0, b0))   # ring A backward
            faces.append((a1, b0, b1))   # ring B forward

    for i in range(n_interior_rings - 1):
        add_strip(ring_start(i), ring_start(i + 1))

    if n_interior_rings > 0:
        add_strip(ring_start(n_interior_rings - 1), rim_start)

    return faces, n_interior, rim_start


class PlanarDiskHyperNet(nn.Module):
    """
    Fixed polar (x,y) grid over a disk of radius `cylinder_radius`. Two
    height values per INTERIOR point (top, bottom); RIM points get a
    single shared, fixed height (0) -- so top and bottom surfaces meet
    exactly at the rim by construction, guaranteed watertight.
    """

    def __init__(self, n_rings: int = 8, n_theta: int = 16,
                 cylinder_radius: float = 1.0, input_mode: str = "noise",
                 noise_dim: int = 8, hidden: int = 64, n_layers: int = 3,
                 dtype=torch.float64, device=None):
        super().__init__()
        assert input_mode in ("noise", "coordinate")
        self.input_mode = input_mode
        self.n_theta = n_theta

        interior_xy, rim_xy = _build_polar_grid(n_rings, n_theta, cylinder_radius, dtype, device)
        n_interior = interior_xy.shape[0]
        self.register_buffer("interior_xy", interior_xy)
        self.register_buffer("rim_xy", rim_xy)

        faces_local, n_interior_check, rim_start = _polar_topology(n_rings, n_theta)
        assert n_interior_check == n_interior

        def top_idx(local):
            return local if local < rim_start else 2 * n_interior + (local - rim_start)

        def bottom_idx(local):
            return (n_interior + local) if local < rim_start else 2 * n_interior + (local - rim_start)

        top_faces = [(top_idx(a), top_idx(b), top_idx(c)) for a, b, c in faces_local]
        bottom_faces = [(bottom_idx(a), bottom_idx(c), bottom_idx(b)) for a, b, c in faces_local]
        all_faces = top_faces + bottom_faces
        self.register_buffer("tlist", torch.tensor(all_faces, dtype=torch.long, device=device))

        self.n_interior = n_interior
        input_dim = noise_dim if input_mode == "noise" else 2
        if input_mode == "noise":
            self.register_buffer("net_input",
                                  torch.randn(n_interior, noise_dim, dtype=dtype, device=device))
        else:
            self.register_buffer("net_input", interior_xy)  # (n_interior, 2)

        self.register_buffer(
            "A", _build_mesh_adjacency(
                torch.tensor([(a, b, c) for a, b, c in faces_local
                              if a < n_interior and b < n_interior and c < n_interior]
                             or [(0, 0, 0)], dtype=torch.long),
                n_interior, dtype, device))

        dims = [input_dim] + [hidden] * (n_layers - 1) + [2]  # 2 outputs: top, bottom offset
        self.layers = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1], dtype=dtype) for i in range(len(dims) - 1)
        ])
        with torch.no_grad():
            self.layers[-1].weight.mul_(0.01)
            self.layers[-1].bias.zero_()

        r_interior = interior_xy.norm(dim=-1)
        taper = (1.0 - r_interior / cylinder_radius).clamp(min=0.0)
        self.register_buffer("taper", taper)

    def forward(self):
        h = self.net_input
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.A @ h
                h = torch.tanh(h)
        top_raw, bot_raw = h[:, 0], h[:, 1]

        top_h = torch.nn.functional.softplus(top_raw) * self.taper
        bot_h = -torch.nn.functional.softplus(bot_raw) * self.taper

        top_v = torch.cat([self.interior_xy, top_h.unsqueeze(-1)], dim=-1)
        bot_v = torch.cat([self.interior_xy, bot_h.unsqueeze(-1)], dim=-1)
        rim_v = torch.cat([self.rim_xy, torch.zeros(self.rim_xy.shape[0], 1,
                                                      dtype=self.rim_xy.dtype,
                                                      device=self.rim_xy.device)], dim=-1)
        vlist = torch.cat([top_v, bot_v, rim_v], dim=0)
        return self.tlist, vlist
