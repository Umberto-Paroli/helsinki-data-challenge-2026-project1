"""
geometry.py
===========

Forward model for the Helsinki Asteroid Challenge 2026: octantoid shape
representation, self-shadowing visibility, camera geometry, and Lambert /
Lommel-Seeliger lightcurve simulation. Also mesh I/O (STL load/decimate)
used by validation to compare against ground truth.

Consolidated from the former `forward_challenge_cristiano.py` +
`operator_extensions.py`. The two were previously separate files because
`operator_extensions` started as an additive, non-invasive experiment on
top of the base operator (finite-distance camera + Lommel-Seeliger
scattering); since the training pipeline (`train.py`) only ever calls the
extended curve functions (never the plain-Lambert/orthographic-only
originals), the extended versions are now the single implementation here
-- called with `camera_distance=None, scattering="lambert"` they are
numerically identical to the old "base" functions, so nothing about the
physics changed, only the duplication.

Dropped in this consolidation (verified unreachable from main.py's ADAM
training path, so removing them changes no runtime behaviour):
  - the old non-extended `intensity_curve`/`binary_curve`/
    `simulate_all_curves` (superseded by the generalized versions below)
  - `voxelize()` (torch ray-cast rasterizer): unused -- ground-truth /
    reconstruction comparison goes through `voxel_measure.py`'s
    trimesh-based voxelizer instead
  - the `_demo()` self-test block
  - the 9-direction `make_challenge_cameras` is kept (still the
    `ChallengeForward` default when no cameras are supplied), but note the
    active config always passes the 28-camera set explicitly.
"""
from __future__ import annotations
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

# ==============================================================
# spherical harmonics basis
# ==============================================================

def P(l: int, m: int, x: torch.Tensor) -> torch.Tensor:
    """Polinomio di Legendre associato P_l^m(x)."""
    pmm = torch.ones_like(x)
    if m > 0:
        somx2 = torch.sqrt((1.0 - x) * (1.0 + x))
        fact = 1.0
        for _ in range(1, m + 1):
            pmm = pmm * (-fact) * somx2
            fact += 2.0
    if l == m:
        return pmm
    pmmp1 = x * (2.0 * m + 1.0) * pmm
    if l == m + 1:
        return pmmp1
    pll = torch.zeros_like(x)
    for ll in range(m + 2, l + 1):
        pll = ((2.0 * ll - 1.0) * x * pmmp1 - (ll + m - 1.0) * pmm) / (ll - m)
        pmm = pmmp1
        pmmp1 = pll
    return pll


def K(l: int, m: int) -> float:
    num = (2.0 * l + 1.0) * math.factorial(l - m)
    den = 4.0 * math.pi * math.factorial(l + m)
    return math.sqrt(num / den)


def SH(l: int, m: int, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """Armonica sferica reale Y_l^m(theta, phi)."""
    cos_theta = torch.cos(theta)
    sqrt2 = math.sqrt(2.0)
    if m == 0:
        return K(l, 0) * P(l, 0, cos_theta)
    if m > 0:
        return sqrt2 * K(l, m) * torch.cos(m * phi) * P(l, m, cos_theta)
    am = -m
    return sqrt2 * K(l, am) * torch.sin(am * phi) * P(l, am, cos_theta)


# ==============================================================
# icosahedral-style sphere triangulation + octantoid shape
# ==============================================================

def triangulate_sphere(nrows: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    nvert = 4 * nrows ** 2 + 2
    nfac = 8 * nrows ** 2

    theta_list = [0.0]
    phi_list = [0.0]
    dth = math.pi / (2 * nrows)

    for i in range(1, nrows + 1):
        dph = math.pi / (2 * i)
        for j in range(4 * i):
            theta_list.append(i * dth)
            phi_list.append(j * dph)
    for i in range(nrows - 1, 0, -1):
        dph = math.pi / (2 * i)
        for j in range(4 * i):
            theta_list.append(math.pi - i * dth)
            phi_list.append(j * dph)
    theta_list.append(math.pi)
    phi_list.append(0.0)
    assert len(theta_list) == nvert

    nod = {(0, 0): 1}
    nnod = 1
    for i in range(1, nrows + 1):
        for j in range(4 * i):
            nnod += 1
            nod[(i, j)] = nnod
            if j == 0:
                nod[(i, 4 * i)] = nnod
    for i in range(nrows - 1, 0, -1):
        for j in range(4 * i):
            nnod += 1
            nod[(2 * nrows - i, j)] = nnod
            if j == 0:
                nod[(2 * nrows - i, 4 * i)] = nnod
    nod[(2 * nrows, 0)] = nnod + 1

    faces: List[List[int]] = []
    for j1 in range(1, nrows + 1):
        for j3 in range(1, 5):
            j0 = (j3 - 1) * j1
            faces.append([nod[(j1 - 1, j0 - (j3 - 1))],
                          nod[(j1, j0)],
                          nod[(j1, j0 + 1)]])
            for j2 in range(j0 + 1, j0 + j1):
                faces.append([nod[(j1, j2)],
                              nod[(j1 - 1, j2 - (j3 - 1))],
                              nod[(j1 - 1, j2 - (j3 - 1) - 1)]])
                faces.append([nod[(j1 - 1, j2 - (j3 - 1))],
                              nod[(j1, j2)],
                              nod[(j1, j2 + 1)]])
    for j1 in range(nrows + 1, 2 * nrows + 1):
        for j3 in range(1, 5):
            j0 = (j3 - 1) * (2 * nrows - j1)
            faces.append([nod[(j1, j0)],
                          nod[(j1 - 1, j0 + (j3 - 1) + 1)],
                          nod[(j1 - 1, j0 + (j3 - 1))]])
            for j2 in range(j0 + 1, j0 + (2 * nrows - j1) + 1):
                faces.append([nod[(j1, j2)],
                              nod[(j1 - 1, j2 + (j3 - 1))],
                              nod[(j1, j2 - 1)]])
                faces.append([nod[(j1, j2)],
                              nod[(j1 - 1, j2 + 1 + (j3 - 1))],
                              nod[(j1 - 1, j2 + (j3 - 1))]])
    assert len(faces) == nfac

    theta = torch.tensor(theta_list, dtype=torch.float64)
    phi = torch.tensor(phi_list, dtype=torch.float64)
    tlist = torch.tensor(faces, dtype=torch.long) - 1
    return theta, phi, tlist


class _OctantoidGrid:
    def __init__(self, LMAX: int, nrows: int,
                 dtype: torch.dtype = torch.float64,
                 device: Optional[torch.device] = None):
        theta, phi, tlist = triangulate_sphere(nrows)
        theta = theta.to(dtype=dtype, device=device)
        phi = phi.to(dtype=dtype, device=device)
        tlist = tlist.to(device=device)
        al = (LMAX + 1) ** 2
        nvert = theta.shape[0]
        B = torch.zeros(nvert, al, dtype=dtype, device=device)
        for j in range(LMAX + 1):
            for k in range(-j, j + 1):
                B[:, j * (j + 1) + k] = SH(j, k, theta, phi)
        self.LMAX = LMAX
        self.nrows = nrows
        self.theta = theta
        self.phi = phi
        self.tlist = tlist
        self.B = B


def octantoid_directions(LMAX: int, nrows: int,
                          dtype: torch.dtype = torch.float64,
                          device: Optional[torch.device] = None) -> torch.Tensor:
    """Unit vertex directions (nvert, 3) of the octantoid's own triangulated
    grid, independent of any shape coefficients / network weights. Several
    hypernet variants (SphereCoordHyperNet, GraphConvHyperNet,
    SphereNoiseGraphHyperNet) need exactly this -- factored out so callers
    can get it without instantiating a full hypernet (and its adjacency
    matrix / trainable layers) just to read `.directions`."""
    grid = _OctantoidGrid(LMAX, nrows, dtype=dtype, device=device)
    return torch.stack([
        torch.sin(grid.theta) * torch.cos(grid.phi),
        torch.sin(grid.theta) * torch.sin(grid.phi),
        torch.cos(grid.theta),
    ], dim=-1)


def octantoid_to_trimesh(a: torch.Tensor,
                          LMAX: int,
                          nrows: int,
                          cached_grid: Optional[_OctantoidGrid] = None
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
    grid = cached_grid or _OctantoidGrid(LMAX, nrows, dtype=a.dtype, device=a.device)
    al = (LMAX + 1) ** 2
    ax = a[:al]
    ay = a[al:2 * al]
    az = a[2 * al:3 * al]

    Bx = grid.B @ ax
    Bxy = grid.B @ (ax + ay)
    Bxz = grid.B @ (ax + az)

    sin_t = torch.sin(grid.theta)
    cos_t = torch.cos(grid.theta)
    cos_p = torch.cos(grid.phi)
    sin_p = torch.sin(grid.phi)

    x = torch.exp(Bx) * sin_t * cos_p
    y = torch.exp(Bxy) * sin_t * sin_p
    z = torch.exp(Bxz) * cos_t
    return grid.tlist, torch.stack([x, y, z], dim=-1)


# ==============================================================
# rotation restricted to the z axis
# ==============================================================

def rotate_z(omega: torch.Tensor,
             omega0: torch.Tensor,
             t: torch.Tensor) -> torch.Tensor:
    """Matrice di rotazione R_z(omega*t + omega0) per ogni epoca. (nE, 3, 3)"""
    dtype = t.dtype
    device = t.device
    omega = torch.as_tensor(omega, dtype=dtype, device=device)
    omega0 = torch.as_tensor(omega0, dtype=dtype, device=device)

    f = omega * t + omega0
    cf, sf = torch.cos(f), torch.sin(f)
    zn, on = torch.zeros_like(f), torch.ones_like(f)
    return torch.stack([
        torch.stack([cf, sf, zn], dim=-1),
        torch.stack([-sf, cf, zn], dim=-1),
        torch.stack([zn, zn, on], dim=-1),
    ], dim=-2)


def _rotate_directions(E_world: torch.Tensor,
                        E0_world: torch.Tensor,
                        omega: torch.Tensor,
                        omega0: torch.Tensor,
                        TIME: torch.Tensor
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Porta E e E0 nel body frame dell'asteroide ad ogni epoca."""
    M = rotate_z(omega, omega0, TIME)               # (nE, 3, 3)
    E = torch.einsum('eij,j->ei', M, E_world)        # (nE, 3)
    E0 = torch.einsum('eij,j->ei', M, E0_world)      # (nE, 3)
    return E, E0


# ==============================================================
# visibility / self-shadowing
# ==============================================================

def is_in_triangle(point: torch.Tensor,
                    direction: torch.Tensor,
                    v1: torch.Tensor,
                    v2: torch.Tensor,
                    v3: torch.Tensor,
                    tol: float = 1e-6) -> torch.Tensor:
    s1 = v1 - v2
    s2 = v1 - v3
    b = v1 - point
    s10, s11, s12 = s1[..., 0], s1[..., 1], s1[..., 2]
    s20, s21, s22 = s2[..., 0], s2[..., 1], s2[..., 2]
    d0, d1, d2 = direction[..., 0], direction[..., 1], direction[..., 2]
    b0, b1, b2 = b[..., 0], b[..., 1], b[..., 2]

    det = (s10 * s21 * d2 - s10 * d1 * s22 + s20 * d1 * s12
           - s20 * s11 * d2 + d0 * s11 * s22 - d0 * s21 * s12)
    valid = det.abs() >= tol
    safe_det = torch.where(valid, det, torch.ones_like(det))

    gamma = (d2 * (s10 * b1 - b0 * s11)
             + d1 * (b0 * s12 - s10 * b2)
             + d0 * (s11 * b2 - b1 * s12)) / safe_det
    beta = (b0 * (s21 * d2 - d1 * s22)
            + b1 * (d0 * s22 - s20 * d2)
            + b2 * (s20 * d1 - s21 * d0)) / safe_det
    return valid & (gamma >= 0) & (gamma <= 1) & (beta >= 0) & (beta <= 1 - gamma)


def _face_centroids_normals(tlist: torch.Tensor,
                             vlist: torch.Tensor
                             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    v1 = vlist[tlist[:, 0]]
    v2 = vlist[tlist[:, 1]]
    v3 = vlist[tlist[:, 2]]
    centroids = (v1 + v2 + v3) / 3.0
    cross = torch.linalg.cross(v2 - v1, v3 - v1)
    norms = torch.linalg.norm(cross, dim=-1, keepdim=True).clamp(min=1e-30)
    normals = cross / norms
    return centroids, normals, cross


def _face_areas_normals(tlist: torch.Tensor,
                         vlist: torch.Tensor
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
    v1 = vlist[tlist[:, 0]]
    v2 = vlist[tlist[:, 1]]
    v3 = vlist[tlist[:, 2]]
    cross = torch.linalg.cross(v2 - v1, v3 - v1)
    norms = torch.linalg.norm(cross, dim=-1, keepdim=True).clamp(min=1e-30)
    normals = cross / norms
    area = 0.5 * norms.squeeze(-1)
    return area, normals


MAX_FACES_HORIZON = 8000
"""Soglia oltre la quale `facets_over_horizon` rifiuta di allocare la matrice
(nfac, nfac, 3, 3). A 8000 facce siamo gia' a ~1.7 GB in float64 -- oltre e'
insensato per il forward model. Se serve gestire mesh piu' grandi, decimare
prima con `decimate_mesh`."""


@torch.no_grad()
def facets_over_horizon(tlist: torch.Tensor,
                         vlist: torch.Tensor
                         ) -> Tuple[torch.Tensor, torch.Tensor, int]:
    nfac = tlist.shape[0]
    if nfac > MAX_FACES_HORIZON:
        gib = (nfac * nfac * 9 * 8) / (1024 ** 3)
        raise MemoryError(
            f"facets_over_horizon: mesh troppo grande (nfac={nfac}); "
            f"servirebbero {gib:.1f} GiB per la matrice di candidati. "
            f"Decima la mesh con `decimate_mesh(tlist, vlist, target_faces=4000)` "
            f"oppure ricaricala con `load_stl_mesh(..., max_faces=4000)`."
        )
    centroids, normals, _ = _face_centroids_normals(tlist, vlist)
    face_verts = vlist[tlist]

    cvvec = face_verts.unsqueeze(0) - centroids[:, None, None, :]
    above_j = torch.einsum('jkvi,ji->jkv', cvvec, normals)
    facing = torch.einsum('jkvi,ki->jkv', cvvec, normals)
    horizon_hit = ((above_j > 0) & (facing < 0)).any(dim=-1)
    horizon_hit.fill_diagonal_(False)

    counts = horizon_hit.sum(dim=-1)
    max_M = int(counts.max().item()) if nfac > 0 else 0
    if max_M == 0:
        return (torch.full((nfac, 0), -1, dtype=torch.long, device=vlist.device),
                torch.zeros((nfac, 0), dtype=torch.bool, device=vlist.device),
                0)

    idx = torch.full((nfac, max_M), -1, dtype=torch.long, device=vlist.device)
    mask = torch.zeros((nfac, max_M), dtype=torch.bool, device=vlist.device)
    for j in range(nfac):
        nz = horizon_hit[j].nonzero(as_tuple=False).squeeze(-1)
        n_j = nz.shape[0]
        if n_j:
            idx[j, :n_j] = nz
            mask[j, :n_j] = True
    return idx, mask, max_M


_DENSE_PATH_MAX_ELEMENTS = 4_000_000
"""Soglia (in numero di entry nE*nfac*max_M) per switchare tra denso
(broadcast puro) e sparso (prune + gather) in `_direction_blocked`."""


@torch.no_grad()
def _direction_blocked(centroids: torch.Tensor,
                        normals: torch.Tensor,
                        face_verts: torch.Tensor,
                        D: torch.Tensor,
                        idx: torch.Tensor,
                        cand_mask: torch.Tensor,
                        max_M: int,
                        prune_dot_tol: float = 0.0
                        ) -> torch.Tensor:
    """Per ogni (epoca e, faccia j) restituisce True se il raggio da
    centroide_j lungo D e' bloccato da almeno un candidato."""
    nE = D.shape[0]
    nfac = centroids.shape[0]
    device = D.device

    if max_M == 0:
        return torch.zeros((nE, nfac), dtype=torch.bool, device=device)

    safe_idx = idx.clamp(min=0)                                # (nfac, max_M)

    mu = D @ normals.t()                                       # (nE, nfac)
    cand_mu = mu[:, safe_idx]                                  # (nE, nfac, max_M)

    if nE * nfac * max_M < _DENSE_PATH_MAX_ELEMENTS:
        active = cand_mask[None] & (cand_mu < 0)
        cand_verts = face_verts[safe_idx]                      # (nfac, max_M, 3, 3)
        v1_b = cand_verts[None, ..., 0, :]
        v2_b = cand_verts[None, ..., 1, :]
        v3_b = cand_verts[None, ..., 2, :]
        pt_b = centroids[None, :, None, :]
        D_b = D[:, None, None, :]
        hit = is_in_triangle(pt_b, D_b, v1_b, v2_b, v3_b)       # (nE, nfac, max_M)
        return (hit & active).any(dim=-1)

    cand_offset = centroids[safe_idx] - centroids[:, None, :]  # (nfac, max_M, 3)
    prune_dot = torch.einsum('ei,jmi->ejm', D, cand_offset)    # (nE, nfac, max_M)

    verts_off = face_verts - centroids[:, None, :]              # (nfac, 3, 3)
    max_radius = torch.linalg.norm(verts_off, dim=-1).max(dim=-1).values  # (nfac,)
    cand_max_r = max_radius[safe_idx]                           # (nfac, max_M)

    active = (cand_mask[None]
              & (cand_mu < 0)
              & (prune_dot > (prune_dot_tol - cand_max_r[None])))
    if not active.any():
        return torch.zeros((nE, nfac), dtype=torch.bool, device=device)

    flat = active.reshape(-1).nonzero(as_tuple=False).squeeze(-1)   # (K,)
    m_ax = flat % max_M
    jm = flat // max_M
    j_ax = jm % nfac
    e_ax = jm // nfac

    pt = centroids[j_ax]                          # (K, 3)
    dr = D[e_ax]                                   # (K, 3)
    tri = face_verts[safe_idx[j_ax, m_ax]]         # (K, 3, 3)

    hit = is_in_triangle(pt, dr, tri[:, 0], tri[:, 1], tri[:, 2])   # (K,) bool

    ej_hit = (e_ax * nfac + j_ax)[hit]
    blocked_flat = torch.zeros(nE * nfac, dtype=torch.bool, device=device)
    blocked_flat.index_fill_(0, ej_hit, True)
    return blocked_flat.view(nE, nfac)


@torch.no_grad()
def find_actual_blockers(tlist: torch.Tensor,
                          vlist: torch.Tensor,
                          E: torch.Tensor,
                          E0: torch.Tensor,
                          cand: Optional[Tuple[torch.Tensor, torch.Tensor, int]] = None,
                          precomputed_sun_blocked: Optional[torch.Tensor] = None
                          ) -> torch.Tensor:
    """Maschera di visibilita' (nE, nfac): True dove la faccia e' vista dalla
    camera E, illuminata dal Sole E0, e non ombreggiata da altre facce."""
    centroids, normals, _ = _face_centroids_normals(tlist, vlist)
    if cand is None:
        cand = facets_over_horizon(tlist, vlist)
    idx, cand_mask, max_M = cand

    mu = E @ normals.t()
    mu0 = E0 @ normals.t()
    lit_and_seen = (mu > 0) & (mu0 > 0)

    if max_M == 0:
        return lit_and_seen

    face_verts = vlist[tlist]

    blocked_view = _direction_blocked(centroids, normals, face_verts, E,
                                       idx, cand_mask, max_M)
    if precomputed_sun_blocked is None:
        blocked_sun = _direction_blocked(centroids, normals, face_verts, E0,
                                          idx, cand_mask, max_M)
    else:
        blocked_sun = precomputed_sun_blocked

    return lit_and_seen & ~blocked_view & ~blocked_sun


# ==============================================================
# challenge camera geometry
# ==============================================================

CHALLENGE_ANGLES_DEG: Tuple[float, ...] = (0.0, 45.0, 90.0, 135.0, 225.0, 270.0, 315.0)
"""Sette angoli di camera specificati dalla challenge (in gradi, attorno a z)."""

LIGHT_DIRECTION = (-1.0, 0.0, 0.0)
"""Direzione da cui arriva la luce: sorgente a (-inf, 0, 0)."""


def make_challenge_cameras(angles_deg: Sequence[float] = CHALLENGE_ANGLES_DEG,
                            include_horizontal: bool = True,
                            include_vertical: bool = True,
                            dtype: torch.dtype = torch.float64,
                            device: Optional[torch.device] = None
                            ) -> torch.Tensor:
    """Direzioni delle telecamere della challenge (9 di default: 7 orizzontali
    + 2 verticali). Usato solo come default di `ChallengeForward` quando non
    viene passato `E_cams_world`; la pipeline attiva passa sempre le 28
    camere di `make_challenge_cameras_28`."""
    dirs: List[List[float]] = []
    if include_horizontal:
        for th in angles_deg:
            r = math.radians(th)
            dirs.append([-math.cos(r), math.sin(r), 0.0])
    if include_vertical:
        dirs.append([0.0, 0.0, 1.0])
        dirs.append([0.0, 0.0, -1.0])
    return torch.tensor(dirs, dtype=dtype, device=device)


def light_direction_tensor(dtype: torch.dtype = torch.float64,
                            device: Optional[torch.device] = None) -> torch.Tensor:
    """Versore fisso E0 = (-1, 0, 0) -- luce a fascio parallelo da -x."""
    return torch.tensor(LIGHT_DIRECTION, dtype=dtype, device=device)


TOP_CAMERA_ELEVATION_DEG = {
    0.0: 21.0, 45.0: 26.0, 90.0: 26.0, 135.0: 26.0,
    225.0: 24.0, 270.0: 24.0, 315.0: 24.0,
}
"""Elevazione della camera "top" per ogni angolo di misura (dalla pagina
della challenge)."""


def make_challenge_cameras_28(angles_deg: Sequence[float] = CHALLENGE_ANGLES_DEG,
                               elevations_deg: Optional[dict] = None,
                               dtype: torch.dtype = torch.float64,
                               device: Optional[torch.device] = None
                               ) -> torch.Tensor:
    """Le 28 direzioni di vista nell'ordine esatto delle colonne 2..29 dei
    file di lightcurve della challenge (per ogni angolo: horizA, horizB,
    top, bottom). Returns (28, 3)."""
    if elevations_deg is None:
        elevations_deg = TOP_CAMERA_ELEVATION_DEG
    dirs: List[List[float]] = []
    for th in angles_deg:
        r = math.radians(th)
        al = math.radians(elevations_deg[float(th)])
        horiz = [-math.cos(r), math.sin(r), 0.0]
        top = [-math.cos(al) * math.cos(r), math.cos(al) * math.sin(r), math.sin(al)]
        bot = [-math.cos(al) * math.cos(r), math.cos(al) * math.sin(r), -math.sin(al)]
        dirs.extend([horiz, list(horiz), top, bot])

    dirs = torch.tensor(dirs, dtype=dtype, device=device)
    dirs[:, 1] = -dirs[:, 1]
    return dirs


def challenge_column_labels(angles_deg: Sequence[float] = CHALLENGE_ANGLES_DEG
                             ) -> List[str]:
    """Etichette leggibili per le 28 colonne, nello stesso ordine."""
    labels = []
    for th in angles_deg:
        labels.extend([f"{th:g}° horizA", f"{th:g}° horizB",
                        f"{th:g}° top", f"{th:g}° bottom"])
    return labels


# ==============================================================
# photometric scattering laws + finite-distance camera correction
# ==============================================================

def scattering_law(mu_p: torch.Tensor,
                    mu0_p: torch.Tensor,
                    law: str = "lambert",
                    ls_weight: float = 1.0,
                    eps: float = 1e-8) -> torch.Tensor:
    """Surface radiance L(mu, mu0) for the three supported photometric laws.
    `mu_p`, `mu0_p` are expected already clamped to >= 0 by the caller.

      "lambert"          -> L = mu0                 (default, matches the
                             original orthographic/Lambert-only operator)
      "lommel_seeliger"  -> L = mu0 / (mu + mu0)
      "mixed"            -> L = ls_weight*LS + (1-ls_weight)*Lambert
    """
    if law == "lambert":
        return mu0_p
    if law == "lommel_seeliger":
        return mu0_p / (mu_p + mu0_p).clamp(min=eps)
    if law == "mixed":
        ls = mu0_p / (mu_p + mu0_p).clamp(min=eps)
        return ls_weight * ls + (1.0 - ls_weight) * mu0_p
    raise ValueError(f"scattering law sconosciuta: {law!r}")


def _perspective_mu(centroids: torch.Tensor,          # (nfac, 3)  body frame, static
                     normals: torch.Tensor,            # (nfac, 3)
                     E_cam_world: torch.Tensor,         # (3,)
                     camera_distance: float,
                     omega: torch.Tensor,
                     omega0: torch.Tensor,
                     TIME: torch.Tensor
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    """True (non-orthographic) view cosine mu(t, facet) for a camera sitting
    at world-frame position `camera_distance * E_cam_world`, object centered
    at the origin. Returns (mu, dist), both (nE, nfac)."""
    cam_pos_world = E_cam_world.to(centroids.dtype) * camera_distance
    M = rotate_z(omega, omega0, TIME)                            # (nE,3,3)
    cam_pos_body = torch.einsum('eij,j->ei', M, cam_pos_world)   # (nE,3)
    delta = cam_pos_body[:, None, :] - centroids[None, :, :]     # (nE,nfac,3)
    dist = torch.linalg.norm(delta, dim=-1).clamp(min=1e-12)     # (nE,nfac)
    view_dir = delta / dist.unsqueeze(-1)
    mu = (view_dir * normals[None, :, :]).sum(-1)                 # (nE,nfac)
    return mu, dist


# ==============================================================
# per-camera lightcurves
# ==============================================================

def intensity_curve(tlist: torch.Tensor,
                     vlist: torch.Tensor,
                     E_cam_world: torch.Tensor,
                     TIME: torch.Tensor,
                     omega: torch.Tensor,
                     omega0: torch.Tensor,
                     E0_world: Optional[torch.Tensor] = None,
                     rel: bool = True,
                     cached_horizon: Optional[Tuple[torch.Tensor, torch.Tensor, int]] = None,
                     precomputed_sun_blocked: Optional[torch.Tensor] = None,
                     camera_distance: Optional[float] = None,
                     inverse_square: bool = False,
                     scattering: str = "lambert",
                     ls_weight: float = 1.0,
                     assume_convex: bool = False,
                     ) -> torch.Tensor:
    """Curva di intensita' per una singola telecamera:
    bright(t) = sum_j visible_j * mu_j * L(mu_j, mu0_j) * area_j

    With the defaults (camera_distance=None, scattering="lambert") this is
    the plain orthographic-Lambertian operator; passing camera_distance
    switches mu to a true finite-distance perspective cosine, and
    scattering="lommel_seeliger"/"mixed" switches the reflectance law.

    `assume_convex=True` skips the O(nfac^2) self-shadowing test
    (`find_actual_blockers`/`facets_over_horizon`) entirely and uses plain
    backface+sun culling instead (`mu>0 & mu0>0`). This is not an
    approximation for a genuinely convex body: a convex surface can never
    shadow itself, so the horizon test would always agree with the simple
    cull anyway -- skipping it is exact, not a shortcut, and is the single
    biggest cost in the forward model (see MAX_FACES_HORIZON's docstring).
    Only correct when `tlist`/`vlist` really do describe a convex polytope
    (e.g. from `convex_hull_mesh`); passing a non-convex mesh here would
    silently ignore real self-shadowing.
    """
    if E0_world is None:
        E0_world = light_direction_tensor(dtype=vlist.dtype, device=vlist.device)
    dt = vlist.dtype
    E_cam_world = E_cam_world.to(dtype=dt)
    E0_world = E0_world.to(dtype=dt)
    TIME = TIME.to(dtype=dt)

    E, E0 = _rotate_directions(E_cam_world, E0_world, omega, omega0, TIME)

    area, normals = _face_areas_normals(tlist, vlist)
    mu0 = E0 @ normals.t()                        # (nE, nfac)

    if camera_distance is None:
        mu = E @ normals.t()
        dist = None
    else:
        centroids, _, _ = _face_centroids_normals(tlist, vlist)
        mu, dist = _perspective_mu(centroids, normals, E_cam_world,
                                    camera_distance, omega, omega0, TIME)

    if assume_convex:
        vis_f = ((mu > 0) & (mu0 > 0)).to(vlist.dtype)
    else:
        with torch.no_grad():
            visible = find_actual_blockers(
                tlist, vlist.detach(), E.detach(), E0.detach(),
                cand=cached_horizon,
                precomputed_sun_blocked=precomputed_sun_blocked,
            )
        vis_f = visible.to(vlist.dtype)

    mu_p = mu.clamp(min=0.0)
    mu0_p = mu0.clamp(min=0.0)
    L = scattering_law(mu_p, mu0_p, law=scattering, ls_weight=ls_weight)

    integrand = vis_f * mu_p * L * area.unsqueeze(0)   # (nE, nfac)
    if camera_distance is not None and inverse_square:
        integrand = integrand / (dist ** 2)
    bright = integrand.sum(dim=-1)

    if rel:
        return bright.shape[0] * bright / bright.sum().clamp(min=1e-30)
    return bright


def binary_curve(tlist: torch.Tensor,
                  vlist: torch.Tensor,
                  E_cam_world: torch.Tensor,
                  TIME: torch.Tensor,
                  omega: torch.Tensor,
                  omega0: torch.Tensor,
                  E0_world: Optional[torch.Tensor] = None,
                  rel: bool = True,
                  soft_tau: Optional[float] = None,
                  radiance_threshold: float = 0.0,
                  cached_horizon: Optional[Tuple[torch.Tensor, torch.Tensor, int]] = None,
                  precomputed_sun_blocked: Optional[torch.Tensor] = None,
                  camera_distance: Optional[float] = None,
                  inverse_square: bool = False,
                  scattering: str = "lambert",
                  ls_weight: float = 1.0,
                  assume_convex: bool = False,
                  ) -> torch.Tensor:
    """Curva "binaria" -- approssima il conteggio dei pixel sopra soglia:
    count(t) = sum_j visible_j * lit_j * mu_j * area_j
    dove `lit_j` sogliona la radianza L(mu_j, mu0_j) (hard o sigmoide via
    `soft_tau`). Con scattering="lambert" (default) L=mu0, quindi
    `radiance_threshold` coincide esattamente con il vecchio `mu0_threshold`.

    `assume_convex=True`: vedi `intensity_curve` -- salta il test di
    self-shadowing O(nfac^2), esatto (non approssimato) per un corpo
    davvero convesso.
    """
    if E0_world is None:
        E0_world = light_direction_tensor(dtype=vlist.dtype, device=vlist.device)
    dt = vlist.dtype
    E_cam_world = E_cam_world.to(dtype=dt)
    E0_world = E0_world.to(dtype=dt)
    TIME = TIME.to(dtype=dt)

    E, E0 = _rotate_directions(E_cam_world, E0_world, omega, omega0, TIME)

    area, normals = _face_areas_normals(tlist, vlist)
    mu0 = E0 @ normals.t()

    if camera_distance is None:
        mu = E @ normals.t()
        dist = None
    else:
        centroids, _, _ = _face_centroids_normals(tlist, vlist)
        mu, dist = _perspective_mu(centroids, normals, E_cam_world,
                                    camera_distance, omega, omega0, TIME)

    if assume_convex:
        vis_f = ((mu > 0) & (mu0 > 0)).to(vlist.dtype)
    else:
        with torch.no_grad():
            visible = find_actual_blockers(
                tlist, vlist.detach(), E.detach(), E0.detach(),
                cand=cached_horizon,
                precomputed_sun_blocked=precomputed_sun_blocked,
            )
        vis_f = visible.to(vlist.dtype)

    mu_p = mu.clamp(min=0.0)
    mu0_p = mu0.clamp(min=0.0)
    L = scattering_law(mu_p, mu0_p, law=scattering, ls_weight=ls_weight)

    if soft_tau is None:
        lit = (L > radiance_threshold).to(vlist.dtype)
    else:
        lit = torch.sigmoid((L - radiance_threshold) / soft_tau)

    integrand = vis_f * lit * mu_p * area.unsqueeze(0)
    if camera_distance is not None and inverse_square:
        integrand = integrand / (dist ** 2)
    count = integrand.sum(dim=-1)

    if rel:
        return count.shape[0] * count / count.sum().clamp(min=1e-30)
    return count


def simulate_all_curves(tlist: torch.Tensor,
                         vlist: torch.Tensor,
                         E_cams_world: torch.Tensor,
                         TIME: torch.Tensor,
                         omega: torch.Tensor,
                         omega0: torch.Tensor,
                         E0_world: Optional[torch.Tensor] = None,
                         mode: str = "both",
                         rel: bool = True,
                         soft_tau: Optional[float] = None,
                         radiance_threshold: float = 0.0,
                         camera_distance: Optional[float] = None,
                         inverse_square: bool = False,
                         scattering: str = "lambert",
                         ls_weight: float = 1.0,
                         assume_convex: bool = False,
                         cached_horizon: Optional[Tuple[torch.Tensor, torch.Tensor, int]] = None,
                         ) -> torch.Tensor:
    """Simula le curve per una lista di telecamere.

    E_cams_world : (nC, 3) versori delle telecamere nel frame inerziale
    mode         : "intensity" | "binary" | "both" ("both" -> (2, nC, nE))

    Il candidato di orizzonte e il lato-Sole (che non dipende dalla camera)
    sono calcolati una sola volta e riusati per tutte le nC camere -- a
    meno che `assume_convex=True`, nel qual caso l'intero test di
    self-shadowing (compresa `facets_over_horizon`, il costo O(nfac^2)
    dominante del modello) viene saltato del tutto: vedi `intensity_curve`.

    `cached_horizon`, se dato, salta anche il ricalcolo di
    `facets_over_horizon` qui dentro e riusa quello passato -- utile
    quando si chiama questa funzione piu' volte nello stesso step con la
    STESSA mesh ma TIME diversi (es. training congiunto su lightcurve
    reali + Blender, vedi train.py): i candidati di orizzonte dipendono
    solo da (tlist, vlist), non da TIME/omega, quindi ricalcolarli per
    ogni chiamata sarebbe puro lavoro ripetuto. Il lato-Sole
    (`blocked_sun`) invece dipende da TIME/omega (la direzione del Sole
    nel body frame cambia nel tempo) e va sempre ricalcolato per ogni
    TIME diverso -- non e' quindi condivisibile allo stesso modo."""
    if E0_world is None:
        E0_world = light_direction_tensor(dtype=vlist.dtype, device=vlist.device)

    nC = E_cams_world.shape[0]

    if assume_convex:
        cached = None
        blocked_sun = None
    else:
        cached = cached_horizon if cached_horizon is not None else facets_over_horizon(tlist, vlist.detach())
        with torch.no_grad():
            dt = vlist.dtype
            v_det = vlist.detach()
            E0_wd = E0_world.to(dtype=dt).detach()
            TIME_d = TIME.to(dtype=dt).detach()
            om = omega.detach() if isinstance(omega, torch.Tensor) else omega
            om0 = omega0.detach() if isinstance(omega0, torch.Tensor) else omega0
            M = rotate_z(om, om0, TIME_d)
            E0_body = torch.einsum('eij,j->ei', M, E0_wd)          # (nE, 3)

            centroids, normals, _ = _face_centroids_normals(tlist, v_det)
            face_verts = v_det[tlist]
            idx_h, cand_mask_h, max_M_h = cached
            blocked_sun = _direction_blocked(centroids, normals, face_verts,
                                              E0_body, idx_h, cand_mask_h, max_M_h)

    def _one(fn):
        extra = ({"soft_tau": soft_tau, "radiance_threshold": radiance_threshold}
                 if fn is binary_curve else {})
        outs = [fn(tlist, vlist, E_cams_world[c], TIME, omega, omega0,
                   E0_world=E0_world, rel=rel, cached_horizon=cached,
                   precomputed_sun_blocked=blocked_sun,
                   camera_distance=camera_distance, inverse_square=inverse_square,
                   scattering=scattering, ls_weight=ls_weight,
                   assume_convex=assume_convex, **extra)
                for c in range(nC)]
        return torch.stack(outs, dim=0)

    if mode == "intensity":
        return _one(intensity_curve)
    if mode == "binary":
        return _one(binary_curve)
    if mode == "both":
        return torch.stack([_one(intensity_curve), _one(binary_curve)], dim=0)
    raise ValueError(f"mode sconosciuto: {mode!r}")


# ==============================================================
# mesh normalization / STL I/O
# ==============================================================

def normalize_to_unit_z(vlist: torch.Tensor,
                         recenter: bool = True) -> torch.Tensor:
    """Riscala la mesh in modo che z_min -> -1 e z_max -> +1 (convenzione
    della challenge). Se `recenter=True`, centra prima la mesh sull'origine
    in xy."""
    v = vlist.clone()
    if recenter:
        cx = 0.5 * (v[:, 0].min() + v[:, 0].max())
        cy = 0.5 * (v[:, 1].min() + v[:, 1].max())
        cz = 0.5 * (v[:, 2].min() + v[:, 2].max())
        v = v - torch.tensor([cx, cy, cz], dtype=v.dtype, device=v.device)
    zmin, zmax = v[:, 2].min(), v[:, 2].max()
    half = 0.5 * (zmax - zmin)
    return v / half.clamp(min=1e-30)


def _clean_mesh_arrays(v_np: np.ndarray,
                        f_np: np.ndarray,
                        verbose: bool = False
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Pulizia della mesh via trimesh: fonde vertici coincidenti, rimuove
    facce degeneri/duplicate/inutilizzate, tiene solo la componente
    connessa piu' grande, ripara le normali. Se trimesh non e' disponibile,
    ritorna gli array invariati."""
    try:
        import trimesh                                                # type: ignore
    except ImportError:
        return v_np, f_np

    m = trimesh.Trimesh(vertices=v_np, faces=f_np, process=False)
    n0 = (len(m.vertices), len(m.faces))

    try:    m.merge_vertices()
    except Exception: pass
    try:    m.update_faces(m.nondegenerate_faces())
    except Exception: pass
    try:    m.update_faces(m.unique_faces())
    except Exception: pass
    try:    m.remove_unreferenced_vertices()
    except Exception: pass

    faces_now = np.asarray(m.faces, dtype=np.int64)
    verts_now = np.asarray(m.vertices, dtype=np.float64)
    if len(faces_now) > 0:
        v1 = verts_now[faces_now[:, 0]]
        v2 = verts_now[faces_now[:, 1]]
        v3 = verts_now[faces_now[:, 2]]
        area = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1), axis=-1)
        med = float(np.median(area))
        AREA_RATIO_MAX = 100.0
        keep = area <= AREA_RATIO_MAX * med
        n_drop = int((~keep).sum())
        if n_drop > 0:
            if verbose:
                print(f"[_clean_mesh]  filtro area: rimosse {n_drop} facce "
                      f"con area > {AREA_RATIO_MAX}*median "
                      f"(area max={area.max():.3g}, median={med:.3g})")
            m.update_faces(np.where(keep)[0])
            try:    m.remove_unreferenced_vertices()
            except Exception: pass

    try:
        comps = m.split(only_watertight=False)
        if len(comps) > 1:
            m = max(comps, key=lambda c: len(c.faces))
            if verbose:
                print(f"[_clean_mesh]  tenuta componente piu' grande "
                      f"({len(comps)} totali, {len(m.faces)} facce)")
    except Exception as e:
        if verbose:
            print(f"[_clean_mesh]  split-components saltato ({type(e).__name__})")

    try:    m.fix_normals()
    except Exception as e:
        if verbose:
            print(f"[_clean_mesh]  fix_normals saltato ({type(e).__name__})")

    n1 = (len(m.vertices), len(m.faces))
    if verbose and n0 != n1:
        print(f"[_clean_mesh]  vertici {n0[0]} -> {n1[0]}, "
              f"facce {n0[1]} -> {n1[1]}")
    return (np.asarray(m.vertices, dtype=np.float64),
            np.asarray(m.faces, dtype=np.int64))


def decimate_mesh(tlist: torch.Tensor,
                   vlist: torch.Tensor,
                   target_faces: int,
                   clean_first: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """Riduce il numero di facce della mesh a ~`target_faces` (quadric-error
    decimation via trimesh / fast-simplification, con fallback voxel
    remeshing per mesh non-manifold). Necessario perche' `facets_over_horizon`
    scala O(nfac^2) in memoria."""
    if tlist.shape[0] <= target_faces and not clean_first:
        return tlist, vlist
    v_np = vlist.detach().cpu().numpy().astype(np.float64)
    f_np = tlist.detach().cpu().numpy().astype(np.int64)

    if clean_first:
        v_np, f_np = _clean_mesh_arrays(v_np, f_np, verbose=True)

    if len(f_np) <= target_faces:
        vlist_new = torch.from_numpy(v_np).to(dtype=vlist.dtype, device=vlist.device)
        tlist_new = torch.from_numpy(f_np).to(device=tlist.device)
        return tlist_new, vlist_new

    def _area_ratio(vs, fs):
        v1 = vs[fs[:, 0]]; v2 = vs[fs[:, 1]]; v3 = vs[fs[:, 2]]
        cross = np.cross(v2 - v1, v3 - v1)
        a = 0.5 * np.linalg.norm(cross, axis=-1)
        med = float(np.median(a))
        return float(a.max()) / med if med > 0 else float('inf')

    MAX_RATIO = 500.0
    new_v: Optional[np.ndarray] = None
    new_f: Optional[np.ndarray] = None
    last_err: Optional[str] = None

    try:
        import fast_simplification                                    # type: ignore
        nv, nf = fast_simplification.simplify(
            v_np, f_np, target_count=int(target_faces))
        nf = np.asarray(nf, dtype=np.int64)
        ratio = _area_ratio(nv, nf)
        if len(nf) >= 0.5 * target_faces and ratio < MAX_RATIO:
            new_v, new_f = nv, nf
            print(f"[decimate_mesh] fast_simplification -> {len(nf)} facce, "
                  f"area max/median={ratio:.1f}")
        else:
            last_err = (f"fast_simplification degenere: {len(nf)} facce, "
                        f"area max/median={ratio:.1f}")
    except ImportError:
        last_err = "fast_simplification non installato"
    except Exception as e:
        last_err = f"fast_simplification: {type(e).__name__}: {e}"

    if new_v is None:
        try:
            import trimesh                                            # type: ignore
            m = trimesh.Trimesh(vertices=v_np, faces=f_np, process=False)
            m2 = m.simplify_quadric_decimation(face_count=int(target_faces))
            nv = np.asarray(m2.vertices, dtype=np.float64)
            nf = np.asarray(m2.faces, dtype=np.int64)
            ratio = _area_ratio(nv, nf) if len(nf) else float('inf')
            if len(nf) >= 0.5 * target_faces and ratio < MAX_RATIO:
                new_v, new_f = nv, nf
                print(f"[decimate_mesh] trimesh -> {len(nf)} facce, "
                      f"area max/median={ratio:.1f}")
            else:
                err2 = (f"trimesh degenere: {len(nf)} facce, "
                        f"area max/median={ratio:.1f}")
                last_err = f"{last_err} ; {err2}" if last_err else err2
        except Exception as e:
            err2 = f"trimesh: {type(e).__name__}: {e}"
            last_err = f"{last_err} ; {err2}" if last_err else err2

    if new_v is None:
        try:
            import trimesh                                            # type: ignore
            m = trimesh.Trimesh(vertices=v_np, faces=f_np, process=True)
            extent = float(np.max(np.abs(np.asarray(m.bounds))))
            pitch = 2.0 * extent / 80.0
            vg = m.voxelized(pitch=pitch).fill()
            m_rem = vg.marching_cubes
            origin = np.asarray(vg.transform[:3, 3], dtype=np.float64)
            nv0 = np.asarray(m_rem.vertices, dtype=np.float64) * pitch + origin
            nf0 = np.asarray(m_rem.faces, dtype=np.int64)
            r0 = _area_ratio(nv0, nf0)
            print(f"[decimate_mesh] voxel remeshing: {len(nf0)} facce, "
                  f"area max/median={r0:.1f}")
            if len(nf0) > target_faces:
                import fast_simplification                            # type: ignore
                nv, nf = fast_simplification.simplify(
                    nv0, nf0, target_count=int(target_faces))
                nf = np.asarray(nf, dtype=np.int64)
            else:
                nv, nf = nv0, nf0
            r = _area_ratio(nv, nf)
            if r < MAX_RATIO:
                new_v, new_f = nv, nf
                print(f"[decimate_mesh] voxel remeshing + decim -> "
                      f"{len(nf)} facce, area max/median={r:.1f}")
            else:
                last_err = f"{last_err} ; voxel remesh degenere ratio={r:.1f}"
        except ImportError as e:
            last_err = f"{last_err} ; voxel remesh (serve scipy): {e}"
        except Exception as e:
            last_err = f"{last_err} ; voxel remesh: {type(e).__name__}: {e}"

    if new_v is None:
        raise RuntimeError(
            f"Impossibile decimare la mesh in modo sensato.\n"
            f"  Tentativi: {last_err}\n"
            f"  Suggerimento: installa scipy per abilitare il voxel-remeshing:\n"
            f"      pip install scipy"
        )

    vlist_new = torch.from_numpy(new_v).to(dtype=vlist.dtype, device=vlist.device)
    tlist_new = torch.from_numpy(new_f).to(device=tlist.device)
    return tlist_new, vlist_new


def load_stl_mesh(path: str,
                   dtype: torch.dtype = torch.float64,
                   device: Optional[torch.device] = None,
                   normalize: bool = True,
                   max_faces: Optional[int] = 2000,
                   clean: bool = True
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Carica una mesh da un file STL (trimesh, poi numpy-stl come
    fallback), la pulisce, la normalizza a z in [-1,1] e la decima a
    `max_faces` (necessario perche' `facets_over_horizon` scala O(nfac^2))."""
    verts_np: Optional[np.ndarray] = None
    faces_np: Optional[np.ndarray] = None
    try:
        import trimesh                                                # type: ignore
        mesh = trimesh.load_mesh(path, process=True)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        verts_np = np.asarray(mesh.vertices, dtype=np.float64)
        faces_np = np.asarray(mesh.faces, dtype=np.int64)
    except Exception:
        try:
            from stl import mesh as stl_mesh                          # type: ignore
            m = stl_mesh.Mesh.from_file(path)
            tri = m.vectors.reshape(-1, 3)                             # (nfac*3, 3)
            verts_np, inv = np.unique(tri, axis=0, return_inverse=True)
            faces_np = inv.reshape(-1, 3).astype(np.int64)
        except Exception as e:
            raise ImportError(
                "Per leggere gli STL serve `trimesh` (consigliato) oppure "
                "`numpy-stl`.  Installa con:  pip install trimesh"
            ) from e

    n_raw = len(faces_np)

    if clean:
        verts_np, faces_np = _clean_mesh_arrays(verts_np, faces_np, verbose=True)

    vlist = torch.from_numpy(verts_np).to(dtype=dtype, device=device)
    tlist = torch.from_numpy(faces_np).to(device=device)

    if normalize:
        vlist = normalize_to_unit_z(vlist)

    if max_faces is not None and tlist.shape[0] > max_faces:
        n_before = tlist.shape[0]
        tlist, vlist = decimate_mesh(tlist, vlist, max_faces, clean_first=False)
        print(f"[load_stl_mesh] STL:{n_raw} -> pulita:{n_before} -> "
              f"decimata:{tlist.shape[0]} facce")
    else:
        print(f"[load_stl_mesh] STL:{n_raw} -> pulita:{tlist.shape[0]} facce")

    return tlist, vlist


# ==============================================================
# convex hull mesh (used by train_convex.py's convex reconstruction stage)
# ==============================================================

def convex_hull_mesh(vlist: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convex hull of `vlist`'s points as a mesh, guaranteed convex by
    construction regardless of where the points sit -- unlike a radius-
    as-function-of-direction representation (which can only ever be a
    smooth "blob", never a genuine flat face / sharp edge, e.g. a cube),
    a hull over freely-placed points can represent exact flat facets,
    since it's literally built by finding them.

    Positions are still `vlist` itself, so gradients flow to whichever
    points end up on the hull -- exactly like every other mesh in this
    file, where `tlist` (topology) is a fixed/detached lookup table and
    only `vlist` (position) carries gradient. Here `tlist` is instead
    RECOMPUTED every call (topology isn't fixed -- which points are "on
    the hull" changes as the points move), via scipy's qhull wrapper,
    strictly on a detached copy: which points end up connected is a
    combinatorial decision, not something to backprop through, same as
    the horizon-visibility mask elsewhere in this file.

    A point strictly INSIDE the hull gets no gradient at all this step
    (perturbing it slightly doesn't change a convex hull) -- there is no
    per-vertex smoothing/weight-sharing here to indirectly reach it either
    (see hypernets.ConvexPointCloud's docstring). Initializing every point
    ON a convex warm-start surface (so ~all points start active) is the
    mitigation used by train_convex.py.
    """
    from scipy.spatial import ConvexHull

    v_np = vlist.detach().cpu().numpy().astype(np.float64)
    hull = ConvexHull(v_np)
    simplices = hull.simplices.copy()                  # (nfac, 3) indices into v_np
    outward = hull.equations[:, :3]                     # (nfac, 3) qhull's own outward normals

    v0 = v_np[simplices[:, 0]]
    v1 = v_np[simplices[:, 1]]
    v2 = v_np[simplices[:, 2]]
    tri_normal = np.cross(v1 - v0, v2 - v0)
    # qhull's simplex vertex order isn't guaranteed to match this file's
    # cross(v2-v1, v3-v1) outward-normal convention -- flip any triangle
    # that comes out backwards so downstream mu/mu0 clamping (which assumes
    # outward normals) stays correct.
    flip = (tri_normal * outward).sum(axis=-1) < 0
    simplices[flip, 1], simplices[flip, 2] = simplices[flip, 2].copy(), simplices[flip, 1].copy()

    tlist = torch.as_tensor(simplices, dtype=torch.long, device=vlist.device)
    return tlist, vlist


# ==============================================================
# convenience module: octantoid -> mesh -> 28 curves
# ==============================================================

class ChallengeForward(nn.Module):
    """Bundle octantoide -> mesh -> nC curve, per usare il forward operator
    dentro un ciclo di ottimizzazione. Rotazione attorno a z, luce fissa a
    `LIGHT_DIRECTION`."""

    def __init__(self,
                 LMAX: int = 6,
                 nrows: int = 5,
                 E_cams_world: Optional[torch.Tensor] = None,
                 dtype: torch.dtype = torch.float64,
                 device: Optional[torch.device] = None):
        super().__init__()
        self.LMAX = LMAX
        self.nrows = nrows
        grid = _OctantoidGrid(LMAX, nrows, dtype=dtype, device=device)
        self.register_buffer("theta", grid.theta)
        self.register_buffer("phi", grid.phi)
        self.register_buffer("tlist", grid.tlist)
        self.register_buffer("B", grid.B)

        if E_cams_world is None:
            E_cams_world = make_challenge_cameras(dtype=dtype, device=device)
        self.register_buffer("E_cams_world", E_cams_world)
        self.register_buffer("E0_world",
                              light_direction_tensor(dtype=dtype, device=device))

    @property
    def n_coeff(self) -> int:
        return (self.LMAX + 1) ** 2

    @property
    def nvert(self) -> int:
        return self.theta.shape[0]

    @property
    def nfac(self) -> int:
        return self.tlist.shape[0]

    @property
    def n_cameras(self) -> int:
        return self.E_cams_world.shape[0]

    def _grid(self) -> _OctantoidGrid:
        g = _OctantoidGrid.__new__(_OctantoidGrid)
        g.LMAX = self.LMAX
        g.nrows = self.nrows
        g.theta = self.theta
        g.phi = self.phi
        g.tlist = self.tlist
        g.B = self.B
        return g

    def mesh(self, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return octantoid_to_trimesh(a, self.LMAX, self.nrows, cached_grid=self._grid())

    def init_ellipsoid(self, semi_a: float = 1.0,
                        semi_b: float = 0.8,
                        semi_c: float = 0.7) -> torch.Tensor:
        Y00 = K(0, 0)
        a = torch.zeros(3 * self.n_coeff, dtype=self.B.dtype, device=self.B.device)
        a[0] = math.log(semi_a) / Y00
        a[self.n_coeff] = math.log(semi_b) / Y00 - a[0]
        a[2 * self.n_coeff] = math.log(semi_c) / Y00 - a[0]
        return a

    def forward(self,
                a: torch.Tensor,
                TIME: torch.Tensor,
                omega: torch.Tensor,
                omega0: torch.Tensor,
                mode: str = "intensity",
                rel: bool = True,
                soft_tau: Optional[float] = None,
                radiance_threshold: float = 0.0
                ) -> torch.Tensor:
        tlist, vlist = self.mesh(a)
        return simulate_all_curves(tlist, vlist, self.E_cams_world, TIME,
                                    omega, omega0,
                                    E0_world=self.E0_world,
                                    mode=mode, rel=rel, soft_tau=soft_tau,
                                    radiance_threshold=radiance_threshold)
