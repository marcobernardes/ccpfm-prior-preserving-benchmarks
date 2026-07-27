#!/usr/bin/env python3
"""
CCPFM / WLS Poisson validation on irregular 2D geometries.

Prior-preserving core-regularization branch.

Reference operator
------------------
- wls:
    Weighted least-squares Laplacian with quadratic reproduction.
    This remains the reference operator.

Research operator
-----------------
- ccpfm:
    Consistency-Corrected Packing Flux Method. A packing-induced virtual-volume
    diffusion operator corrected by global polynomial consistency constraints.

What this script does
---------------------
- Generates fixed cloud families across refinement levels on irregular domains.
- Runs Poisson manufactured-solution tests in parallel.
- Builds both WLS and CCPFm operators.
- Computes patch-test diagnostics.
- Produces robust aggregate summaries and unstable-case diagnostics.
- Optionally creates case-level diagnostic figures.

Equation solved
---------------
    -Delta u = f  in Omega,
          u = g  on dOmega,
with manufactured exact solution exact_u().
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags, eye
from scipy.sparse.linalg import lsqr, spsolve
from scipy.spatial import cKDTree

Array = np.ndarray
EPS = 1.0e-14

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


# -----------------------------------------------------------------------------
# Manufactured solution
# -----------------------------------------------------------------------------

def exact_u(x: Array, y: Array) -> Array:
    return (
        np.sin(np.pi * x) * np.sin(np.pi * y)
        + 0.15 * np.cos(2.0 * np.pi * x) * np.sin(3.0 * np.pi * y)
    )


def rhs_f(x: Array, y: Array) -> Array:
    term1 = 2.0 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)
    term2 = 13.0 * np.pi**2 * 0.15 * np.cos(2.0 * np.pi * x) * np.sin(3.0 * np.pi * y)
    return term1 + term2


# -----------------------------------------------------------------------------
# Domains
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Domain2D:
    name: str
    bbox: Tuple[float, float, float, float]
    inside: Callable[[Array, Array], Array]
    boundary_loops: Sequence[Callable[[Array], Array]]


def make_circle_domain(radius: float = 1.0, center: Tuple[float, float] = (0.0, 0.0)) -> Domain2D:
    cx, cy = center

    def inside(x: Array, y: Array) -> Array:
        return (x - cx) ** 2 + (y - cy) ** 2 <= radius**2 + 1.0e-14

    def gamma(t: Array) -> Array:
        th = 2.0 * np.pi * t
        return np.column_stack([cx + radius * np.cos(th), cy + radius * np.sin(th)])

    r = radius
    return Domain2D("circle", (cx - r, cx + r, cy - r, cy + r), inside, [gamma])


def make_rotated_ellipse_domain(a: float = 1.2, b: float = 0.7, angle_deg: float = 28.0) -> Domain2D:
    angle = np.deg2rad(angle_deg)
    ca, sa = np.cos(angle), np.sin(angle)

    def to_local(x: Array, y: Array) -> Tuple[Array, Array]:
        xr = ca * x + sa * y
        yr = -sa * x + ca * y
        return xr, yr

    def inside(x: Array, y: Array) -> Array:
        xr, yr = to_local(x, y)
        return (xr / a) ** 2 + (yr / b) ** 2 <= 1.0 + 1.0e-14

    def gamma(t: Array) -> Array:
        th = 2.0 * np.pi * t
        x0 = a * np.cos(th)
        y0 = b * np.sin(th)
        x = ca * x0 - sa * y0
        y = sa * x0 + ca * y0
        return np.column_stack([x, y])

    R = max(a, b) + 0.05
    return Domain2D("ellipse", (-R, R, -R, R), inside, [gamma])


def make_star_domain(r0: float = 0.78, r1: float = 0.22, k: int = 5) -> Domain2D:
    def r_of_theta(theta: Array) -> Array:
        return r0 + r1 * np.cos(k * theta)

    def inside(x: Array, y: Array) -> Array:
        theta = np.arctan2(y, x)
        rho = np.hypot(x, y)
        return rho <= r_of_theta(theta) + 1.0e-14

    def gamma(t: Array) -> Array:
        th = 2.0 * np.pi * t
        rr = r_of_theta(th)
        return np.column_stack([rr * np.cos(th), rr * np.sin(th)])

    R = r0 + abs(r1) + 0.05
    return Domain2D("star", (-R, R, -R, R), inside, [gamma])


def make_annulus_domain(rin: float = 0.35, rout: float = 1.0) -> Domain2D:
    def inside(x: Array, y: Array) -> Array:
        rho2 = x * x + y * y
        return (rho2 <= rout**2 + 1.0e-14) & (rho2 >= rin**2 - 1.0e-14)

    def gamma_outer(t: Array) -> Array:
        th = 2.0 * np.pi * t
        return np.column_stack([rout * np.cos(th), rout * np.sin(th)])

    def gamma_inner(t: Array) -> Array:
        th = 2.0 * np.pi * t
        return np.column_stack([rin * np.cos(th), rin * np.sin(th)])

    R = rout + 0.05
    return Domain2D("annulus", (-R, R, -R, R), inside, [gamma_outer, gamma_inner])


DOMAIN_FACTORIES = {
    "circle": make_circle_domain,
    "ellipse": make_rotated_ellipse_domain,
    "star": make_star_domain,
    "annulus": make_annulus_domain,
}


# -----------------------------------------------------------------------------
# Point generation and fixed cloud families
# -----------------------------------------------------------------------------

def sample_loop_equally(
    gamma: Callable[[Array], Array], spacing: float, phase: float = 0.0, oversample: int = 8000
) -> Array:
    t_dense = (np.linspace(0.0, 1.0, oversample, endpoint=False) + phase) % 1.0
    pts_dense = gamma(t_dense)
    pts_next = np.roll(pts_dense, -1, axis=0)
    seg = np.linalg.norm(pts_next - pts_dense, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    perimeter = cum[-1]
    n = max(12, int(np.ceil(perimeter / spacing)))
    targets = np.linspace(0.0, perimeter, n, endpoint=False)
    idx = np.searchsorted(cum, targets, side="right") - 1
    idx = np.clip(idx, 0, len(seg) - 1)
    local_s = targets - cum[idx]
    frac = np.divide(local_s, seg[idx], out=np.zeros_like(local_s), where=seg[idx] > EPS)
    pts = pts_dense[idx] + frac[:, None] * (pts_next[idx] - pts_dense[idx])
    return pts


def _boundary_normals(loop: Callable[[Array], Array], t: Array, inside: Callable[[Array, Array], Array]) -> Array:
    dt = 1.0e-4
    pts = loop(t)
    pts_f = loop((t + dt) % 1.0)
    pts_b = loop((t - dt) % 1.0)
    tangent = pts_f - pts_b
    n1 = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    nrm = np.linalg.norm(n1, axis=1, keepdims=True)
    n1 = np.divide(n1, nrm, out=np.zeros_like(n1), where=nrm > EPS)
    test_plus = pts + 1.0e-3 * n1
    plus_inside = inside(test_plus[:, 0], test_plus[:, 1])
    return np.where(plus_inside[:, None], n1, -n1)


def add_boundary_band_points(
    domain: Domain2D,
    h: float,
    base_phase: float,
    loops_pts: List[Array],
    band_levels: Sequence[float],
) -> Array:
    band_pts: List[Array] = []
    for loop_idx, loop in enumerate(domain.boundary_loops):
        loop_pts = loops_pts[loop_idx]
        n = len(loop_pts)
        t = (np.linspace(0.0, 1.0, n, endpoint=False) + base_phase + 0.13 * loop_idx) % 1.0
        pts = loop(t)
        normals = _boundary_normals(loop, t, domain.inside)
        for level in band_levels:
            cand = pts + (level * h) * normals
            mask = domain.inside(cand[:, 0], cand[:, 1])
            if np.any(mask):
                band_pts.append(cand[mask])
    if not band_pts:
        return np.empty((0, 2), dtype=float)
    return np.vstack(band_pts)


def deduplicate_points(points: Array, tol: float) -> Array:
    if len(points) == 0:
        return points
    scale = max(tol, 1.0e-12)
    keys = np.round(points / scale).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    idx = np.sort(idx)
    return points[idx]


def bridson_poisson_in_domain(
    inside: Callable[[Array, Array], Array],
    bbox: Tuple[float, float, float, float],
    radius: float,
    k: int = 30,
    seed: int = 0,
    initial_points: Array | None = None,
) -> Array:
    xmin, xmax, ymin, ymax = bbox
    width, height = xmax - xmin, ymax - ymin
    cell = radius / np.sqrt(2.0)
    nx = max(1, int(np.ceil(width / cell)))
    ny = max(1, int(np.ceil(height / cell)))
    grid = -np.ones((nx, ny), dtype=int)
    points: List[np.ndarray] = []
    active: List[int] = []
    rng = np.random.default_rng(seed)

    def grid_coords(p: Array) -> Tuple[int, int]:
        gx = int((p[0] - xmin) / cell)
        gy = int((p[1] - ymin) / cell)
        return gx, gy

    def far_enough(p: Array) -> bool:
        gx, gy = grid_coords(p)
        i0, i1 = max(gx - 2, 0), min(gx + 3, nx)
        j0, j1 = max(gy - 2, 0), min(gy + 3, ny)
        for ii in range(i0, i1):
            for jj in range(j0, j1):
                idx = grid[ii, jj]
                if idx >= 0:
                    if np.linalg.norm(points[idx] - p) < radius:
                        return False
        return True

    def add_point(p: Array) -> None:
        idx = len(points)
        points.append(np.asarray(p, dtype=float))
        active.append(idx)
        gx, gy = grid_coords(p)
        if 0 <= gx < nx and 0 <= gy < ny:
            grid[gx, gy] = idx

    if initial_points is not None and len(initial_points) > 0:
        for p in initial_points:
            if inside(np.array([p[0]]), np.array([p[1]]))[0]:
                if far_enough(p):
                    add_point(p)

    if not points:
        for _ in range(20000):
            p = np.array([rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)])
            if inside(np.array([p[0]]), np.array([p[1]]))[0]:
                add_point(p)
                break
        else:
            raise RuntimeError("Failed to seed Poisson-disk sampler inside domain.")

    while active:
        aidx = int(rng.integers(len(active)))
        center = points[active[aidx]]
        accepted = False
        for _ in range(k):
            ang = rng.uniform(0.0, 2.0 * np.pi)
            rad = radius * (1.0 + rng.uniform(0.0, 1.0))
            cand = center + rad * np.array([np.cos(ang), np.sin(ang)])
            if not (xmin <= cand[0] <= xmax and ymin <= cand[1] <= ymax):
                continue
            if not inside(np.array([cand[0]]), np.array([cand[1]]))[0]:
                continue
            if far_enough(cand):
                add_point(cand)
                accepted = True
                break
        if not accepted:
            active.pop(aidx)

    return np.array(points, dtype=float)


def make_interior_band_points(
    boundary_loops: Sequence[Array],
    inside: Callable[[Array, Array], Array],
    offset: float,
    stride: int = 2,
) -> Array:
    """Create a one-sided interior band by offsetting sampled boundary points inward."""
    band_points: List[Array] = []
    for loop in boundary_loops:
        if len(loop) < 3:
            continue
        pts = loop[::max(1, stride)]
        prev_pts = np.roll(pts, 1, axis=0)
        next_pts = np.roll(pts, -1, axis=0)
        tang = next_pts - prev_pts
        normals = np.column_stack([-tang[:, 1], tang[:, 0]])
        norms = np.linalg.norm(normals, axis=1)
        normals = normals / np.maximum(norms[:, None], 1e-12)
        cand1 = pts + offset * normals
        cand2 = pts - offset * normals
        in1 = inside(cand1[:, 0], cand1[:, 1])
        in2 = inside(cand2[:, 0], cand2[:, 1])
        chosen = np.where(in1[:, None], cand1, cand2)
        chosen_mask = in1 | in2
        if np.any(chosen_mask):
            band_points.append(chosen[chosen_mask])
    if not band_points:
        return np.empty((0, 2), dtype=float)
    return np.vstack(band_points)


def generate_point_cloud(
    domain: Domain2D,
    h: float,
    family_seed: int,
    boundary_spacing_factor: float = 0.80,
    interior_radius_factor: float = 0.92,
    boundary_clearance_factor: float = 0.45,
    band_offset_factor: float = 0.55,
    band_stride: int = 2,
) -> Tuple[Array, Array]:
    """Build boundary and interior points. Returns (points, boundary_mask)."""
    phase = 0.0
    boundary_spacing = boundary_spacing_factor * h
    if domain.name == "star":
        boundary_spacing *= 0.90
        band_stride = 1
    boundary_loops = [sample_loop_equally(loop, boundary_spacing, phase=phase) for loop in domain.boundary_loops]
    boundary_pts = deduplicate_points(np.vstack(boundary_loops), tol=0.04 * h)

    band_points = make_interior_band_points(
        boundary_loops=boundary_loops,
        inside=domain.inside,
        offset=band_offset_factor * h,
        stride=band_stride,
    )
    extra_band = add_boundary_band_points(
        domain=domain,
        h=h,
        base_phase=phase,
        loops_pts=boundary_loops,
        band_levels=(0.60, 1.10) if domain.name == "star" else (0.60,),
    )
    initial_band = np.vstack([a for a in [band_points, extra_band] if len(a) > 0]) if (len(band_points) > 0 or len(extra_band) > 0) else np.empty((0, 2), dtype=float)
    initial_band = deduplicate_points(initial_band, tol=0.08 * h)

    interior_radius = interior_radius_factor * h
    interior_pts = bridson_poisson_in_domain(
        inside=domain.inside,
        bbox=domain.bbox,
        radius=interior_radius,
        seed=family_seed,
        initial_points=initial_band,
    )

    if len(interior_pts) > 0:
        btree = cKDTree(boundary_pts)
        d_bdry, _ = btree.query(interior_pts, k=1)
        interior_pts = interior_pts[d_bdry > boundary_clearance_factor * h]
        interior_pts = deduplicate_points(interior_pts, tol=0.04 * h)

    pts = np.vstack([boundary_pts, interior_pts]) if len(interior_pts) else boundary_pts.copy()
    is_boundary = np.zeros(len(pts), dtype=bool)
    is_boundary[: len(boundary_pts)] = True
    return pts, is_boundary



# -----------------------------------------------------------------------------
# Geometry utilities and local stencils
# -----------------------------------------------------------------------------

def estimate_radii(points: Array, tree: cKDTree) -> Array:
    k = min(3, len(points))
    d, _ = tree.query(points, k=k)
    d = np.atleast_2d(d)
    nearest = np.maximum(d[:, 1], 1e-12)
    return 0.52 * nearest


def compute_boundary_distances(points: Array, is_boundary: Array) -> Array:
    d = np.zeros(len(points), dtype=float)
    if np.any(is_boundary) and np.any(~is_boundary):
        btree = cKDTree(points[is_boundary])
        d[~is_boundary], _ = btree.query(points[~is_boundary], k=1)
    return d


def local_neighbor_indices(
    i: int,
    points: Array,
    tree: cKDTree,
    radii: Array,
    alpha: float,
    min_neighbors: int,
    max_neighbors: int,
    is_boundary: Array | None = None,
    near_boundary: bool = False,
    preferred_interior_frac: float = 0.65,
    min_boundary_keep: int = 4,
) -> Array:
    k = min(max_neighbors + 1, len(points))
    d, idx = tree.query(points[i], k=k)
    d = np.atleast_1d(d)
    idx = np.atleast_1d(idx)
    mask_not_self = idx != i
    idx = idx[mask_not_self]
    d = d[mask_not_self]
    if len(idx) == 0:
        raise RuntimeError(f"Node {i} has no neighbors.")

    thresh = alpha * (radii[i] + radii[idx])
    cand_mask = d <= thresh
    cand_idx = idx[cand_mask]
    if len(cand_idx) < min_neighbors:
        cand_idx = idx[: min(max_neighbors, len(idx))]
    if len(cand_idx) < min_neighbors:
        cand_idx = idx[: min(min_neighbors, len(idx))]

    target_total = min(min_neighbors, len(cand_idx))

    if not near_boundary or is_boundary is None:
        nbrs = cand_idx[:target_total]
        if len(nbrs) < min_neighbors:
            nbrs = idx[: min(min_neighbors, len(idx))]
        return np.asarray(nbrs, dtype=int)

    is_b = is_boundary[cand_idx]
    interior = cand_idx[~is_b]
    boundary = cand_idx[is_b]
    target_interior = min(len(interior), max(1, int(np.ceil(preferred_interior_frac * min_neighbors))))
    target_boundary = min(len(boundary), max(min_boundary_keep, min_neighbors - target_interior))

    chosen: List[int] = []
    chosen.extend(interior[:target_interior].tolist())
    chosen.extend(boundary[:target_boundary].tolist())
    used = set(chosen)
    for j in cand_idx:
        jj = int(j)
        if jj not in used:
            chosen.append(jj)
            used.add(jj)
        if len(chosen) >= target_total:
            break
    if len(chosen) < min_neighbors:
        for j in idx:
            jj = int(j)
            if jj not in used:
                chosen.append(jj)
                used.add(jj)
            if len(chosen) >= min(min_neighbors, len(idx)):
                break
    return np.asarray(chosen[:target_total], dtype=int)


def laplacian_stencil_wls(
    center: Array,
    neighbors: Array,
    local_is_boundary: Array | None = None,
    near_boundary: bool = False,
    boundary_weight_boost: float = 1.8,
    rcond: float = 1e-12,
) -> Tuple[Array, float, float]:
    pts = np.vstack([center[None, :], neighbors])
    dx = pts[:, 0] - center[0]
    dy = pts[:, 1] - center[1]
    r = np.hypot(dx, dy)
    scale = max(np.max(r[1:]) if len(r) > 1 else 1.0, 1e-12)
    q = r / scale
    w = np.exp(-4.0 * q * q) + 1e-14
    w[0] = max(w[0], 1.0)
    if near_boundary and local_is_boundary is not None and len(local_is_boundary) == len(w):
        w = w * np.where(local_is_boundary, boundary_weight_boost, 1.0)
        w[0] = max(w[0], 1.0)

    dxs = dx / scale
    dys = dy / scale
    P = np.column_stack([
        np.ones(len(pts)),
        dxs,
        dys,
        dxs * dxs,
        dxs * dys,
        dys * dys,
    ])
    sqrtw = np.sqrt(w)
    Pw = P * sqrtw[:, None]
    try:
        u, s, vt = np.linalg.svd(Pw, full_matrices=False)
        cond = float(s[0] / max(s[-1], EPS))
        sigma_min = float(s[-1])
    except np.linalg.LinAlgError:
        cond = float("inf")
        sigma_min = 0.0
    pinv = np.linalg.pinv(Pw, rcond=rcond)
    l = np.array([0.0, 0.0, 0.0, 2.0 / (scale * scale), 0.0, 2.0 / (scale * scale)])
    c = (l @ pinv) * sqrtw
    return np.asarray(c, dtype=float), cond, sigma_min




def laplacian_stencil_wls_reference(center: Array, neighbors: Array, rcond: float = 1e-12) -> Tuple[Array, float, float]:
    """Reference WLS Laplacian used by the previously stable baseline."""
    pts = np.vstack([center[None, :], neighbors])
    dx = pts[:, 0] - center[0]
    dy = pts[:, 1] - center[1]
    r = np.hypot(dx, dy)
    scale = max(np.max(r[1:]) if len(r) > 1 else 1.0, 1e-12)
    q = r / scale
    w = np.exp(-4.0 * q * q) + 1e-12
    w[0] = max(w[0], 1.0)
    P = np.column_stack([np.ones(len(pts)), dx, dy, dx * dx, dx * dy, dy * dy])
    sqrtw = np.sqrt(w)
    Pw = P * sqrtw[:, None]
    try:
        u, s, vt = np.linalg.svd(Pw, full_matrices=False)
        cond = float(s[0] / max(s[-1], EPS))
        sigma_min = float(s[-1])
    except np.linalg.LinAlgError:
        cond = float("inf")
        sigma_min = 0.0
    pinv = np.linalg.pinv(Pw, rcond=rcond)
    l = np.array([0.0, 0.0, 0.0, 2.0, 0.0, 2.0])
    c = (l @ pinv) * sqrtw
    return np.asarray(c, dtype=float), cond, sigma_min


def build_wls_node_stencil(
    i: int,
    points: Array,
    tree: cKDTree,
    radii: Array,
    is_boundary: Array,
    dist_to_boundary: Array,
    alpha: float,
    min_neighbors: int,
    max_neighbors: int,
    boundary_neighbor_boost: int,
    boundary_layer_factor: float,
    boundary_weight_boost: float,
) -> Tuple[Array, Array, float, float]:
    # Use the previously stable reference stencil construction.
    best = None
    best_cond = float('inf')
    for extra in (0, 4):
        nbrs = local_neighbor_indices(
            i=i,
            points=points,
            tree=tree,
            radii=radii,
            alpha=alpha,
            min_neighbors=min(min_neighbors + extra, len(points) - 1),
            max_neighbors=min(max_neighbors + 2 * extra, len(points) - 1),
        )
        c_local, cond, sigma_min = laplacian_stencil_wls_reference(
            center=points[i],
            neighbors=points[nbrs],
        )
        if cond < best_cond:
            best = (nbrs, c_local, cond, sigma_min)
            best_cond = cond
    return best


# -----------------------------------------------------------------------------
# Graph, prior, and CCPFm constrained correction
# -----------------------------------------------------------------------------

def build_edge_list(
    points: Array,
    is_boundary: Array,
    radii: Array,
    alpha: float,
    min_neighbors: int,
    max_neighbors: int,
    boundary_neighbor_boost: int,
    boundary_layer_factor: float,
    selective_enrichment_neighbors: int = 1,
    selective_core_factor: float = 2.0,
    selective_cond_trigger: float = 2.5e4,
    selective_sigma_trigger: float = 2.5e-4,
    selective_distance_factor: float = 3.0,
    force_min_flagged_core_nodes: int = 2,
    force_relaxed_distance_factor: float = 5.0,
) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], int], List[List[int]], Array, Array, Array, Array, Array, Array]:
    """Build a CCPFm graph starting from the boundary-split version, then add
    *very sparse* rescue edges only for flagged interior-core nodes.

    This branch now has a deterministic fallback so the enrichment path is
    actually exercised: if the absolute triggers produce too few flagged core
    nodes, the worst-scoring core nodes are force-flagged. It also progressively
    relaxes the rescue-edge distance cap when a flagged node cannot find an
    admissible partner under the strict cap.

    Returns edge_is_added, node_is_flagged, node_is_force_flagged, and
    node_added_count arrays so diagnostics can separate original vs added rescue
    edges and show whether flagged nodes actually received rescue edges.
    """
    tree = cKDTree(points)
    dist_to_boundary = compute_boundary_distances(points, is_boundary)
    median_r = float(np.median(radii[~is_boundary])) if np.any(~is_boundary) else float(np.median(radii))

    edges_set: set[Tuple[int, int]] = set()
    adjacency_nodes: List[set[int]] = [set() for _ in range(len(points))]
    local_conds = np.full(len(points), np.nan, dtype=float)
    local_sigmins = np.full(len(points), np.nan, dtype=float)

    # Base graph: identical spirit to the boundary-split branch.
    for i in range(len(points)):
        if is_boundary[i]:
            continue
        near_boundary = dist_to_boundary[i] <= boundary_layer_factor * max(radii[i], median_r)
        min_eff = min_neighbors + (boundary_neighbor_boost if near_boundary else 0)
        max_eff = max_neighbors + (boundary_neighbor_boost if near_boundary else 0)
        nbrs_best = None
        best_score = float('inf')
        best_cond = float('inf')
        best_sigma = 0.0
        for extra in (0, 6):
            nbrs = local_neighbor_indices(
                i=i,
                points=points,
                tree=tree,
                radii=radii,
                alpha=alpha + 0.08 * (extra > 0),
                min_neighbors=min(min_eff + extra, len(points) - 1),
                max_neighbors=min(max_eff + 2 * extra, len(points) - 1),
                is_boundary=is_boundary,
                near_boundary=near_boundary,
            )
            local_idx = np.concatenate([[i], nbrs])
            _, cond, sigma_min = laplacian_stencil_wls(
                center=points[i],
                neighbors=points[nbrs],
                local_is_boundary=is_boundary[local_idx],
                near_boundary=near_boundary,
            )
            score = cond / max(sigma_min, 1.0e-16)
            if score < best_score:
                best_score = score
                best_cond = cond
                best_sigma = sigma_min
                nbrs_best = nbrs
        nbrs = np.asarray(nbrs_best, dtype=int)
        local_conds[i] = best_cond
        local_sigmins[i] = best_sigma
        for j in nbrs:
            j = int(j)
            e = (i, j) if i < j else (j, i)
            edges_set.add(e)
            adjacency_nodes[i].add(j)
            adjacency_nodes[j].add(i)

    base_edges_set = set(edges_set)
    node_is_flagged = np.zeros(len(points), dtype=bool)
    node_is_force_flagged = np.zeros(len(points), dtype=bool)
    node_added_count = np.zeros(len(points), dtype=int)

    # Sparse, flagged-node-only enrichment: only deep-interior nodes with weak
    # local clouds get 1-2 rescue edges, and only to nearby interior-core nodes.
    if selective_enrichment_neighbors > 0:
        interior_idx = np.where(~is_boundary)[0]
        if interior_idx.size:
            tree_int = cKDTree(points[interior_idx])
            core_thresh_global = selective_core_factor * median_r

            core_nodes: List[int] = []
            core_scores = np.full(len(points), -np.inf, dtype=float)
            threshold_flagged: List[int] = []

            for i in interior_idx:
                i = int(i)
                if dist_to_boundary[i] <= max(core_thresh_global, selective_core_factor * radii[i]):
                    continue
                core_nodes.append(i)
                cond_i = local_conds[i]
                sig_i = local_sigmins[i]
                score_i = (cond_i / max(sig_i, 1.0e-16)) if (np.isfinite(cond_i) and np.isfinite(sig_i)) else -np.inf
                core_scores[i] = score_i
                flagged = (np.isfinite(cond_i) and cond_i > selective_cond_trigger) or (
                    np.isfinite(sig_i) and sig_i < selective_sigma_trigger
                )
                if flagged:
                    threshold_flagged.append(i)
                    node_is_flagged[i] = True

            # Deterministic fallback: if the absolute thresholds are too strict for
            # the current cloud family, force-flag the worst-scoring core nodes so
            # the rescue-edge path is actually exercised.
            need_force = max(0, min(int(force_min_flagged_core_nodes), len(core_nodes)) - len(threshold_flagged))
            if need_force > 0:
                ranked_core = sorted(
                    [i for i in core_nodes if not node_is_flagged[i]],
                    key=lambda ii: core_scores[ii],
                    reverse=True,
                )
                for i in ranked_core[:need_force]:
                    node_is_flagged[i] = True
                    node_is_force_flagged[i] = True

            for i in core_nodes:
                i = int(i)
                if not node_is_flagged[i]:
                    continue
                cond_i = local_conds[i]
                sig_i = local_sigmins[i]
                extra_target = max(1, int(selective_enrichment_neighbors))
                # allow one extra edge only if both triggers are bad
                if (np.isfinite(cond_i) and cond_i > 1.8 * selective_cond_trigger) and (
                    np.isfinite(sig_i) and sig_i < 0.8 * selective_sigma_trigger
                ):
                    extra_target += 1

                # Ask for more candidates than before so the admissible core-core
                # search is less likely to fail spuriously.
                kq = min(len(interior_idx), max(len(adjacency_nodes[i]) + 20, 40))
                d, q = tree_int.query(points[i], k=kq)
                q = np.atleast_1d(q)
                d = np.atleast_1d(d)
                added = 0

                strict_cap = selective_distance_factor * max(radii[i], median_r)
                relaxed_cap = max(strict_cap, force_relaxed_distance_factor * max(radii[i], median_r))
                cap_schedule = [strict_cap]
                if relaxed_cap > strict_cap * 1.0001:
                    cap_schedule.append(relaxed_cap)

                for dist_cap in cap_schedule:
                    for dj, qq in zip(d, q):
                        j = int(interior_idx[int(qq)])
                        if j == i or j in adjacency_nodes[i]:
                            continue
                        if dj > dist_cap:
                            continue
                        if dist_to_boundary[j] <= max(core_thresh_global, selective_core_factor * radii[j]):
                            continue
                        e = (i, j) if i < j else (j, i)
                        if e in edges_set:
                            continue
                        edges_set.add(e)
                        adjacency_nodes[i].add(j)
                        adjacency_nodes[j].add(i)
                        added += 1
                        node_added_count[i] += 1
                        node_added_count[j] += 1
                        if added >= extra_target:
                            break
                    if added >= extra_target:
                        break

                # Final fallback for force-flagged or still-unserved flagged nodes:
                # if strict core-core search fails, connect to the nearest available
                # interior node away from the immediate boundary layer so the
                # enrichment path is actually exercised.
                if added < extra_target:
                    d_all, q_all = tree.query(points[i], k=min(len(points), max(len(adjacency_nodes[i]) + 40, 64)))
                    d_all = np.atleast_1d(d_all)
                    q_all = np.atleast_1d(q_all)
                    for dj, qq in zip(d_all, q_all):
                        j = int(qq)
                        if j == i or is_boundary[j] or j in adjacency_nodes[i]:
                            continue
                        if dist_to_boundary[j] <= 0.75 * max(core_thresh_global, selective_core_factor * radii[j]):
                            continue
                        e = (i, j) if i < j else (j, i)
                        if e in edges_set:
                            continue
                        edges_set.add(e)
                        adjacency_nodes[i].add(j)
                        adjacency_nodes[j].add(i)
                        added += 1
                        node_added_count[i] += 1
                        node_added_count[j] += 1
                        if added >= extra_target:
                            break

    edges = sorted(edges_set)
    edge_index = {e: k for k, e in enumerate(edges)}
    edge_is_added = np.array([e not in base_edges_set for e in edges], dtype=bool)
    adjacency = [sorted(list(s)) for s in adjacency_nodes]
    return edges, edge_index, adjacency, local_conds, local_sigmins, edge_is_added, node_is_flagged, node_is_force_flagged, node_added_count


def compute_node_volumes(radii: Array) -> Array:
    return np.pi * np.maximum(radii, 1.0e-12) ** 2


def compute_packing_prior(points: Array, radii: Array, edges: List[Tuple[int, int]]) -> Tuple[Array, Array, Array]:
    m = len(edges)
    lambda0 = np.zeros(m, dtype=float)
    weights = np.zeros(m, dtype=float)
    distances = np.zeros(m, dtype=float)
    for k, (i, j) in enumerate(edges):
        dij = float(np.linalg.norm(points[j] - points[i]))
        dij = max(dij, 1.0e-12)
        distances[k] = dij
        aij = 2.0 * min(radii[i], radii[j])
        lambda0[k] = aij / dij
        weights[k] = 1.0 / max(dij * dij, 1.0e-10)
    return lambda0, weights, distances

def classify_ccpfm_edges(
    edges: List[Tuple[int, int]],
    is_boundary: Array,
    dist_to_boundary: Array,
    h_ref: float,
    edge_is_added: Array,
    core_factor: float = 2.0,
) -> Dict[str, Array]:
    m = len(edges)
    is_core = np.zeros(m, dtype=bool)
    touches_boundary_band = np.zeros(m, dtype=bool)
    both_interior = np.zeros(m, dtype=bool)
    core_threshold = core_factor * h_ref
    for k, (i, j) in enumerate(edges):
        ii = not bool(is_boundary[i])
        jj = not bool(is_boundary[j])
        both_interior[k] = ii and jj
        if ii and jj and (dist_to_boundary[i] > core_threshold) and (dist_to_boundary[j] > core_threshold):
            is_core[k] = True
        if bool(is_boundary[i]) or bool(is_boundary[j]) or (dist_to_boundary[i] <= core_threshold) or (dist_to_boundary[j] <= core_threshold):
            touches_boundary_band[k] = True
    return {
        'edge_is_core': is_core,
        'edge_touches_boundary_band': touches_boundary_band,
        'edge_both_interior': both_interior,
        'edge_is_added': edge_is_added.copy(),
        'edge_is_original': (~edge_is_added).copy(),
    }


def _subset_rel_stats(delta: Array, base: Array, subset: Array) -> Dict[str, float]:
    if subset is None or np.sum(subset) == 0:
        return {
            'median': float('nan'),
            'p90': float('nan'),
            'max': float('nan'),
            'negfrac': float('nan'),
            'n': 0,
        }
    rel = np.abs(delta[subset]) / np.maximum(np.abs(base[subset]), 1.0e-14)
    return {
        'median': float(np.median(rel)),
        'p90': float(np.quantile(rel, 0.90)),
        'max': float(np.max(rel)),
        'negfrac': float(np.mean((base[subset] + delta[subset]) < 0.0)),
        'n': int(np.sum(subset)),
    }


def compute_ccpfm_prior(
    points: Array,
    radii: Array,
    edges: List[Tuple[int, int]],
    edge_index: Dict[Tuple[int, int], int],
    adjacency: List[List[int]],
    is_boundary: Array,
    volumes: Array,
    edge_is_added: Array,
    added_edge_penalty: float = 12.0,
    prior_blend_geom: float = 0.35,
    prior_blend_wls: float = 0.65,
) -> Tuple[Array, Array, Array]:
    """Hybrid prior on a fixed graph.

    Original edges blend geometric and WLS-derived suggestions. Added edges keep a
    geometric prior and can still be penalized separately in the correction
    objective so they act as last-resort degrees of freedom.
    """
    lambda_geom, weights, distances = compute_packing_prior(points, radii, edges)
    m = len(edges)
    sugg = [[] for _ in range(m)]
    for i in range(len(points)):
        if is_boundary[i]:
            continue
        nbrs = np.asarray(adjacency[i], dtype=int)
        if len(nbrs) < 6:
            continue
        c_local, _, _ = laplacian_stencil_wls_reference(points[i], points[nbrs])
        offdiag = -c_local[1:]
        for j, coeff in zip(nbrs, offdiag):
            e = (i, int(j)) if i < int(j) else (int(j), i)
            ek = edge_index[e]
            lam = volumes[i] * float(coeff)
            if np.isfinite(lam):
                sugg[ek].append(lam)
    lambda_wls = np.full(m, np.nan, dtype=float)
    for k in range(m):
        vals = np.asarray(sugg[k], dtype=float)
        if vals.size:
            pos = vals[vals > 0.0]
            if pos.size:
                lambda_wls[k] = float(np.median(pos))
            else:
                lambda_wls[k] = float(np.median(np.abs(vals)))
    lambda0 = lambda_geom.copy()
    use_blend = (~edge_is_added) & np.isfinite(lambda_wls) & (lambda_wls > 0.0)
    blend_sum = max(float(prior_blend_geom + prior_blend_wls), 1.0e-14)
    blend_geom = float(prior_blend_geom) / blend_sum
    blend_wls = float(prior_blend_wls) / blend_sum
    lambda0[use_blend] = blend_geom * lambda_geom[use_blend] + blend_wls * lambda_wls[use_blend]
    weights = weights * (1.0 + 0.5 * use_blend.astype(float))
    if len(edge_is_added):
        weights[edge_is_added] *= max(1.0, float(added_edge_penalty))
    return lambda0, weights, distances


def assemble_ccpfm_constraints(
    points: Array,
    is_boundary: Array,
    adjacency: List[List[int]],
    edge_index: Dict[Tuple[int, int], int],
    volumes: Array,
) -> Tuple[csr_matrix, Array, Dict[str, Array]]:
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    rhs: List[float] = []
    row_meta: List[Tuple[int, str]] = []
    scales = np.ones(len(points), dtype=float)

    basis_labels = ["x", "y", "x2h", "xy", "y2h"]
    row_counter = 0
    for i in range(len(points)):
        if is_boundary[i]:
            continue
        nbrs = adjacency[i]
        if len(nbrs) < 5:
            raise RuntimeError(f"Interior node {i} has too few graph neighbors ({len(nbrs)}) for CCPFm.")
        dx = points[nbrs, 0] - points[i, 0]
        dy = points[nbrs, 1] - points[i, 1]
        dloc = np.hypot(dx, dy)
        s_i = max(float(np.median(dloc)), 1.0e-12)
        scales[i] = s_i
        rhs_local = [0.0, 0.0, volumes[i] / (s_i * s_i), 0.0, volumes[i] / (s_i * s_i)]
        for label, target in zip(basis_labels, rhs_local):
            rhs.append(float(target))
            row_meta.append((i, label))

        for j in nbrs:
            j = int(j)
            e = (i, j) if i < j else (j, i)
            ek = edge_index[e]
            xi = (points[j, 0] - points[i, 0]) / s_i
            yi = (points[j, 1] - points[i, 1]) / s_i
            coeffs = [xi, yi, 0.5 * xi * xi, xi * yi, 0.5 * yi * yi]
            for local_row, coeff in enumerate(coeffs):
                rows.append(row_counter + local_row)
                cols.append(ek)
                data.append(float(coeff))
        row_counter += 5

    A = coo_matrix((data, (rows, cols)), shape=(row_counter, len(edge_index))).tocsr()
    b = np.asarray(rhs, dtype=float)
    meta = {
        "row_node": np.array([i for i, _ in row_meta], dtype=int),
        "row_label": np.array([lbl for _, lbl in row_meta], dtype=object),
        "scales": scales,
    }
    return A, b, meta


def solve_ccpfm_correction(
    lambda0: Array,
    weights: Array,
    A: csr_matrix,
    b: Array,
    regularization: float = 1.0e-12,
    edge_classes: Dict[str, Array] | None = None,
    original_prior_anchor: float = 1.0,
    core_prior_anchor: float = 1.0,
    boundary_band_prior_anchor: float = 1.0,
    added_prior_anchor: float = 1.0,
) -> Tuple[Array, Dict[str, float]]:
    def empty_stats() -> Dict[str, float]:
        return {
            "constraint_resid_inf": 0.0,
            "constraint_resid_l2": 0.0,
            "correction_rel_l2": 0.0,
            "correction_abs_median": 0.0,
            "correction_abs_max": 0.0,
            "correction_rel_median": 0.0,
            "correction_rel_p90": 0.0,
            "correction_rel_max": 0.0,
            "negative_lambda_fraction": float(np.mean(lambda0 < 0.0)) if len(lambda0) else 0.0,
            "lambda_ratio_median": 1.0,
            "original_edge_fraction": 1.0,
            "added_edge_fraction": 0.0,
            "original_edge_correction_rel_median": float('nan'),
            "original_edge_correction_rel_p90": float('nan'),
            "original_edge_correction_rel_max": float('nan'),
            "original_edge_negative_lambda_fraction": float('nan'),
            "added_edge_correction_rel_median": float('nan'),
            "added_edge_correction_rel_p90": float('nan'),
            "added_edge_correction_rel_max": float('nan'),
            "added_edge_negative_lambda_fraction": float('nan'),
            "core_edge_correction_rel_median": float('nan'),
            "core_edge_correction_rel_p90": float('nan'),
            "core_edge_correction_rel_max": float('nan'),
            "core_edge_negative_lambda_fraction": float('nan'),
            "boundary_edge_correction_rel_median": float('nan'),
            "boundary_edge_correction_rel_p90": float('nan'),
            "boundary_edge_correction_rel_max": float('nan'),
            "boundary_edge_negative_lambda_fraction": float('nan'),
            "original_core_edge_correction_rel_median": float('nan'),
            "original_core_edge_correction_rel_p90": float('nan'),
            "original_core_edge_correction_rel_max": float('nan'),
            "original_core_edge_negative_lambda_fraction": float('nan'),
            "original_boundary_edge_correction_rel_median": float('nan'),
            "original_boundary_edge_correction_rel_p90": float('nan'),
            "original_boundary_edge_correction_rel_max": float('nan'),
            "original_boundary_edge_negative_lambda_fraction": float('nan'),
            "effective_weight_median": float('nan'),
            "effective_weight_p90": float('nan'),
        }

    if A.shape[0] == 0:
        return lambda0.copy(), empty_stats()

    weights_eff = np.asarray(weights, dtype=float).copy()
    if edge_classes is not None and len(weights_eff):
        edge_is_original = edge_classes.get('edge_is_original')
        edge_is_core = edge_classes.get('edge_is_core')
        edge_touches_boundary_band = edge_classes.get('edge_touches_boundary_band')
        edge_is_added = edge_classes.get('edge_is_added')
        if edge_is_original is not None:
            weights_eff[edge_is_original] *= max(float(original_prior_anchor), 1.0e-12)
        if edge_is_core is not None:
            weights_eff[edge_is_core] *= max(float(core_prior_anchor), 1.0e-12)
        if edge_touches_boundary_band is not None:
            weights_eff[edge_touches_boundary_band] *= max(float(boundary_band_prior_anchor), 1.0e-12)
        if edge_is_added is not None:
            weights_eff[edge_is_added] *= max(float(added_prior_anchor), 1.0e-12)

    row_norm = np.sqrt(np.asarray(A.power(2).sum(axis=1)).ravel())
    row_scale = 1.0 / np.maximum(row_norm, 1.0e-14)
    As = diags(row_scale) @ A
    bs = row_scale * b

    winv = 1.0 / np.maximum(weights_eff, 1.0e-16)
    AT = As.transpose().tocsr()
    M = (As @ diags(winv) @ AT).tocsr()
    if regularization > 0.0:
        reg = regularization * max(1.0, float(M.diagonal().max()) if M.shape[0] else 1.0)
        M = M + reg * eye(M.shape[0], format="csr")

    rhs = As @ lambda0 - bs
    try:
        mu = spsolve(M, rhs)
    except Exception:
        mu = lsqr(M, rhs, atol=1e-13, btol=1e-13, iter_lim=50000)[0]
    lambdas = lambda0 - winv * (AT @ mu)

    cres = As @ lambdas - bs
    if np.max(np.abs(cres)) > 1.0e-10:
        try:
            dmu = spsolve(M, cres)
        except Exception:
            dmu = lsqr(M, cres, atol=1e-13, btol=1e-13, iter_lim=50000)[0]
        lambdas = lambdas - winv * (AT @ dmu)
        cres = As @ lambdas - bs

    cres_raw = A @ lambdas - b
    delta = lambdas - lambda0
    rel = float(np.linalg.norm(delta) / max(np.linalg.norm(lambda0), 1.0e-14))
    abs_delta = np.abs(delta)
    rel_delta = abs_delta / np.maximum(np.abs(lambda0), 1.0e-14)
    stats = {
        "constraint_resid_inf": float(np.max(np.abs(cres_raw))) if len(cres_raw) else 0.0,
        "constraint_resid_l2": float(np.linalg.norm(cres_raw)) if len(cres_raw) else 0.0,
        "correction_rel_l2": rel,
        "correction_abs_median": float(np.median(abs_delta)) if len(abs_delta) else 0.0,
        "correction_abs_max": float(np.max(abs_delta)) if len(abs_delta) else 0.0,
        "correction_rel_median": float(np.median(rel_delta)) if len(rel_delta) else 0.0,
        "correction_rel_p90": float(np.quantile(rel_delta, 0.90)) if len(rel_delta) else 0.0,
        "correction_rel_max": float(np.max(rel_delta)) if len(rel_delta) else 0.0,
        "negative_lambda_fraction": float(np.mean(lambdas < 0.0)) if len(lambdas) else 0.0,
        "lambda_ratio_median": float(np.median(np.abs(lambdas) / np.maximum(np.abs(lambda0), 1.0e-14))) if len(lambdas) else 1.0,
        "effective_weight_median": float(np.median(weights_eff)) if len(weights_eff) else float('nan'),
        "effective_weight_p90": float(np.quantile(weights_eff, 0.90)) if len(weights_eff) else float('nan'),
    }
    if edge_classes is not None:
        edge_is_original = edge_classes.get('edge_is_original', np.zeros(0, dtype=bool))
        edge_is_added = edge_classes.get('edge_is_added', np.zeros(0, dtype=bool))
        edge_is_core = edge_classes.get('edge_is_core', np.zeros(0, dtype=bool))
        edge_touches_boundary_band = edge_classes.get('edge_touches_boundary_band', np.zeros(0, dtype=bool))
        orig_core = edge_is_original & edge_is_core
        orig_bnd = edge_is_original & edge_touches_boundary_band

        core_stats = _subset_rel_stats(delta, lambda0, edge_is_core)
        bnd_stats = _subset_rel_stats(delta, lambda0, edge_touches_boundary_band)
        add_stats = _subset_rel_stats(delta, lambda0, edge_is_added)
        orig_stats = _subset_rel_stats(delta, lambda0, edge_is_original)
        orig_core_stats = _subset_rel_stats(delta, lambda0, orig_core)
        orig_bnd_stats = _subset_rel_stats(delta, lambda0, orig_bnd)
        n_all = max(len(lambda0), 1)
        stats.update({
            "original_edge_fraction": float(np.sum(edge_is_original) / n_all),
            "added_edge_fraction": float(np.sum(edge_is_added) / n_all),
            "original_edge_correction_rel_median": orig_stats['median'],
            "original_edge_correction_rel_p90": orig_stats['p90'],
            "original_edge_correction_rel_max": orig_stats['max'],
            "original_edge_negative_lambda_fraction": orig_stats['negfrac'],
            "added_edge_correction_rel_median": add_stats['median'],
            "added_edge_correction_rel_p90": add_stats['p90'],
            "added_edge_correction_rel_max": add_stats['max'],
            "added_edge_negative_lambda_fraction": add_stats['negfrac'],
            "core_edge_correction_rel_median": core_stats['median'],
            "core_edge_correction_rel_p90": core_stats['p90'],
            "core_edge_correction_rel_max": core_stats['max'],
            "core_edge_negative_lambda_fraction": core_stats['negfrac'],
            "boundary_edge_correction_rel_median": bnd_stats['median'],
            "boundary_edge_correction_rel_p90": bnd_stats['p90'],
            "boundary_edge_correction_rel_max": bnd_stats['max'],
            "boundary_edge_negative_lambda_fraction": bnd_stats['negfrac'],
            "original_core_edge_correction_rel_median": orig_core_stats['median'],
            "original_core_edge_correction_rel_p90": orig_core_stats['p90'],
            "original_core_edge_correction_rel_max": orig_core_stats['max'],
            "original_core_edge_negative_lambda_fraction": orig_core_stats['negfrac'],
            "original_boundary_edge_correction_rel_median": orig_bnd_stats['median'],
            "original_boundary_edge_correction_rel_p90": orig_bnd_stats['p90'],
            "original_boundary_edge_correction_rel_max": orig_bnd_stats['max'],
            "original_boundary_edge_negative_lambda_fraction": orig_bnd_stats['negfrac'],
            "n_added_edges": int(add_stats['n']),
        })
    return lambdas, stats


def build_ccpfm_matrix(
    points: Array,
    is_boundary: Array,
    adjacency: List[List[int]],
    edge_index: Dict[Tuple[int, int], int],
    lambdas: Array,
    volumes: Array,
) -> csr_matrix:
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    n = len(points)
    for i in range(n):
        if is_boundary[i]:
            rows.append(i)
            cols.append(i)
            data.append(1.0)
            continue
        diag = 0.0
        Vi = max(volumes[i], 1.0e-14)
        for j in adjacency[i]:
            e = (i, j) if i < j else (j, i)
            lam = float(lambdas[edge_index[e]])
            coeff = lam / Vi
            diag += coeff
            rows.append(i)
            cols.append(j)
            data.append(float(-coeff))
        rows.append(i)
        cols.append(i)
        data.append(float(diag))
    return coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()


# -----------------------------------------------------------------------------
# Patch tests
# -----------------------------------------------------------------------------

def patch_test_metrics(A: csr_matrix, points: Array, is_boundary: Array) -> Dict[str, float]:
    x = points[:, 0]
    y = points[:, 1]
    interior = ~is_boundary
    metrics: Dict[str, float] = {}

    tests = {
        "const": (np.ones(len(points)), np.zeros(len(points))),
        "x": (x, np.zeros(len(points))),
        "y": (y, np.zeros(len(points))),
        "x2h": (0.5 * x * x, -np.ones(len(points))),
        "xy": (x * y, np.zeros(len(points))),
        "y2h": (0.5 * y * y, -np.ones(len(points))),
        "quad": (0.5 * (x * x + y * y), -2.0 * np.ones(len(points))),
    }
    for key, (u, rhs) in tests.items():
        r = A @ u - rhs
        metrics[f"patch_{key}_max"] = float(np.max(np.abs(r[interior]))) if np.any(interior) else 0.0
    return metrics


# -----------------------------------------------------------------------------
# Assembly and solve
# -----------------------------------------------------------------------------

def assemble_operator_and_diagnostics(
    points: Array,
    is_boundary: Array,
    operator: str,
    alpha: float,
    min_neighbors: int,
    max_neighbors: int,
    boundary_neighbor_boost: int,
    boundary_layer_factor: float,
    boundary_weight_boost: float,
    selective_enrichment_neighbors: int,
    selective_core_factor: float,
    selective_cond_trigger: float,
    selective_sigma_trigger: float,
    selective_distance_factor: float,
    added_edge_penalty: float,
    force_min_flagged_core_nodes: int,
    force_relaxed_distance_factor: float,
    freeze_ccpfm_attempt: int,
    original_prior_anchor: float,
    core_prior_anchor: float,
    boundary_band_prior_anchor: float,
    added_prior_anchor: float,
    constraint_regularization: float,
    prior_blend_geom: float,
    prior_blend_wls: float,
) -> Tuple[csr_matrix, Array, Dict[str, float], Dict[str, Array | float]]:
    n = len(points)
    x, y = points[:, 0], points[:, 1]
    tree = cKDTree(points)
    radii = estimate_radii(points, tree)
    volumes = compute_node_volumes(radii)
    dist_to_boundary = compute_boundary_distances(points, is_boundary)
    u_bdry = exact_u(x, y)
    f_rhs = rhs_f(x, y)

    if operator == "wls":
        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []
        b = np.zeros(n, dtype=float)
        local_conds = np.full(n, np.nan, dtype=float)
        local_sigmins = np.full(n, np.nan, dtype=float)
        local_sizes = np.full(n, np.nan, dtype=float)
        for i in range(n):
            if is_boundary[i]:
                rows.append(i); cols.append(i); data.append(1.0)
                b[i] = u_bdry[i]
                continue
            nbrs, c_local, cond, sigma_min = build_wls_node_stencil(
                i=i, points=points, tree=tree, radii=radii, is_boundary=is_boundary,
                dist_to_boundary=dist_to_boundary, alpha=alpha, min_neighbors=min_neighbors,
                max_neighbors=max_neighbors, boundary_neighbor_boost=boundary_neighbor_boost,
                boundary_layer_factor=boundary_layer_factor, boundary_weight_boost=boundary_weight_boost,
            )
            local_conds[i] = cond
            local_sigmins[i] = sigma_min
            local_idx = np.concatenate([[i], nbrs])
            local_sizes[i] = len(local_idx)
            for jj, coeff in zip(local_idx, -c_local):
                rows.append(i); cols.append(int(jj)); data.append(float(coeff))
            b[i] = f_rhs[i]
        A = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
        corr_stats = {
            "constraint_resid_inf": 0.0,
            "constraint_resid_l2": 0.0,
            "correction_rel_l2": 0.0,
            "negative_lambda_fraction": 0.0,
            "lambda_ratio_median": 1.0,
        }
        lambdas = np.array([], dtype=float)
        adjacency = [[] for _ in range(n)]
        edges = []
        edge_index = {}
        edge_is_added = np.array([], dtype=bool)
        node_is_flagged = np.zeros(n, dtype=bool)
        node_is_force_flagged = np.zeros(n, dtype=bool)
        node_added_count = np.zeros(n, dtype=int)
        selected_attempt_idx = -1
        selected_attempt_alpha = float("nan")
        selected_attempt_min_neighbors = -1
        selected_attempt_max_neighbors = -1
    elif operator == "ccpfm":
        best = None
        best_score = float('inf')
        # Start from the boundary-split branch graph; allow only very sparse flagged-node rescue edges.
        attempt_configs = [
            (max(alpha, 2.35), max(min_neighbors + 6, 24), max(max_neighbors + 12, 64)),
            (max(alpha, 2.55), max(min_neighbors + 10, 28), max(max_neighbors + 20, 80)),
            (max(alpha, 2.75), max(min_neighbors + 14, 32), max(max_neighbors + 28, 96)),
        ]
        h_ref = float(np.median(radii[~is_boundary])) if np.any(~is_boundary) else float(np.median(radii))
        for attempt_idx, (alpha_cc, min_cc, max_cc) in enumerate(attempt_configs):
            if freeze_ccpfm_attempt >= 0 and attempt_idx != freeze_ccpfm_attempt:
                continue
            edges, edge_index, adjacency, local_conds, local_sigmins, edge_is_added, node_is_flagged, node_is_force_flagged, node_added_count = build_edge_list(
                points=points, is_boundary=is_boundary, radii=radii, alpha=alpha_cc,
                min_neighbors=min_cc, max_neighbors=max_cc,
                boundary_neighbor_boost=boundary_neighbor_boost + 4,
                boundary_layer_factor=boundary_layer_factor,
                selective_enrichment_neighbors=selective_enrichment_neighbors,
                selective_core_factor=selective_core_factor,
                selective_cond_trigger=selective_cond_trigger,
                selective_sigma_trigger=selective_sigma_trigger,
                selective_distance_factor=selective_distance_factor,
                force_min_flagged_core_nodes=force_min_flagged_core_nodes,
                force_relaxed_distance_factor=force_relaxed_distance_factor,
            )
            lambda0, weights, distances = compute_ccpfm_prior(
                points, radii, edges, edge_index, adjacency, is_boundary, volumes,
                edge_is_added=edge_is_added, added_edge_penalty=added_edge_penalty,
                prior_blend_geom=prior_blend_geom, prior_blend_wls=prior_blend_wls,
            )
            Acons, bcons, meta = assemble_ccpfm_constraints(points, is_boundary, adjacency, edge_index, volumes)
            edge_classes = classify_ccpfm_edges(edges, is_boundary, dist_to_boundary, h_ref=h_ref, edge_is_added=edge_is_added, core_factor=selective_core_factor)
            lambdas, corr_stats = solve_ccpfm_correction(
                lambda0, weights, Acons, bcons, regularization=constraint_regularization,
                edge_classes=edge_classes,
                original_prior_anchor=original_prior_anchor,
                core_prior_anchor=core_prior_anchor,
                boundary_band_prior_anchor=boundary_band_prior_anchor,
                added_prior_anchor=added_prior_anchor,
            )
            A = build_ccpfm_matrix(points, is_boundary, adjacency, edge_index, lambdas, volumes)
            patches_tmp = patch_test_metrics(A, points, is_boundary)
            # Score prefers patch-tightness first, then smaller relative correction.
            score = max(patches_tmp["patch_x_max"], patches_tmp["patch_y_max"], patches_tmp["patch_quad_max"]) + 1.0e-2 * corr_stats["correction_rel_p90"]
            if score < best_score:
                best_score = score
                best = (A, adjacency, edges, edge_index, local_conds, local_sigmins, lambda0, lambdas, distances, corr_stats, patches_tmp, meta, edge_is_added, node_is_flagged, node_is_force_flagged, node_added_count, attempt_idx, alpha_cc, min_cc, max_cc)
            if score <= 5.0e-7:
                break
        if best is None:
            raise RuntimeError(f"No valid CCPFm attempt found for freeze_ccpfm_attempt={freeze_ccpfm_attempt}.")
        A, adjacency, edges, edge_index, local_conds, local_sigmins, lambda0, lambdas, distances, corr_stats, patches, meta, edge_is_added, node_is_flagged, node_is_force_flagged, node_added_count, selected_attempt_idx, selected_attempt_alpha, selected_attempt_min_neighbors, selected_attempt_max_neighbors = best
        b = np.where(is_boundary, u_bdry, f_rhs)
        diagnostics_extra = {
            "constraint_row_node": meta["row_node"],
            "constraint_row_label": meta["row_label"],
            "constraint_scales": meta["scales"],
            "lambda0": lambda0,
            "lambdas": lambdas,
            "edge_distances": distances,
            "edge_is_added": edge_is_added,
            "node_is_flagged": node_is_flagged,
            "node_is_force_flagged": node_is_force_flagged,
            "node_added_count": node_added_count,
            "selected_attempt_idx": np.array([selected_attempt_idx], dtype=int),
            "selected_attempt_alpha": np.array([selected_attempt_alpha], dtype=float),
            "selected_attempt_min_neighbors": np.array([selected_attempt_min_neighbors], dtype=int),
            "selected_attempt_max_neighbors": np.array([selected_attempt_max_neighbors], dtype=int),
        }
        local_sizes = np.array([len(a) + 1 for a in adjacency], dtype=float)
    else:
        raise ValueError(f"Unknown operator '{operator}'.")

    if operator == "wls":
        patches = patch_test_metrics(A, points, is_boundary)
    max_local_size = int(np.nanmax(local_sizes[~is_boundary])) if np.any(~is_boundary) else 0

    stats = {
        "n_points": int(n),
        "n_boundary": int(np.sum(is_boundary)),
        "n_interior": int(np.sum(~is_boundary)),
        "max_local_size": int(max_local_size),
        "median_local_cond": float(np.nanmedian(local_conds[~is_boundary])) if np.any(~is_boundary) else float("nan"),
        "max_local_cond": float(np.nanmax(local_conds[~is_boundary])) if np.any(~is_boundary) else float("nan"),
        "min_local_sigma_min": float(np.nanmin(local_sigmins[~is_boundary])) if np.any(~is_boundary) else float("nan"),
        "median_radius": float(np.median(radii)),
        "median_dist_boundary": float(np.median(dist_to_boundary[~is_boundary])) if np.any(~is_boundary) else 0.0,
        "n_edges": int(len(edges)),
        **corr_stats,
        **patches,
    }
    if operator == "ccpfm":
        flagged_with_added = int(np.sum(node_is_flagged & (node_added_count > 0)))
        stats.update({
            "n_added_edges": int(np.sum(edge_is_added)),
            "n_original_edges": int(len(edge_is_added) - np.sum(edge_is_added)),
            "added_edge_fraction": float(np.mean(edge_is_added)) if len(edge_is_added) else 0.0,
            "n_flagged_enrichment_nodes": int(np.sum(node_is_flagged)),
            "n_force_flagged_enrichment_nodes": int(np.sum(node_is_force_flagged)),
            "n_flagged_nodes_with_added_edges": flagged_with_added,
            "n_flagged_nodes_without_added_edges": int(np.sum(node_is_flagged) - flagged_with_added),
            "mean_added_edges_per_flagged_node": float(np.mean(node_added_count[node_is_flagged])) if np.any(node_is_flagged) else 0.0,
            "max_added_edges_on_flagged_node": int(np.max(node_added_count[node_is_flagged])) if np.any(node_is_flagged) else 0,
            "selected_attempt_idx": int(selected_attempt_idx),
            "selected_attempt_alpha": float(selected_attempt_alpha),
            "selected_attempt_min_neighbors": int(selected_attempt_min_neighbors),
            "selected_attempt_max_neighbors": int(selected_attempt_max_neighbors),
            "original_prior_anchor": float(original_prior_anchor),
            "core_prior_anchor": float(core_prior_anchor),
            "boundary_band_prior_anchor": float(boundary_band_prior_anchor),
            "added_prior_anchor": float(added_prior_anchor),
            "constraint_regularization": float(constraint_regularization),
            "prior_blend_geom": float(prior_blend_geom),
            "prior_blend_wls": float(prior_blend_wls),
        })
    diagnostics = {
        "radii": radii,
        "volumes": volumes,
        "dist_to_boundary": dist_to_boundary,
        "local_conds": local_conds,
        "local_sigmins": local_sigmins,
        "adjacency_sizes": np.array([len(a) for a in adjacency], dtype=int),
        "lambdas": lambdas,
    }
    if operator == "ccpfm":
        diagnostics.update(diagnostics_extra)
    return A, b, stats, diagnostics


def maybe_plot_case_diagnostics(
    outdir: Path,
    tag: str,
    points: Array,
    abs_err: Array,
    is_boundary: Array,
    local_conds: Array,
    dist_to_boundary: Array,
) -> None:
    if plt is None:
        return
    max_idx = int(np.argmax(abs_err))
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    sc = ax.scatter(points[:, 0], points[:, 1], c=abs_err, s=22, cmap="viridis")
    ax.scatter(points[max_idx, 0], points[max_idx, 1], s=95, marker="x")
    txt = (
        f"max|e|={abs_err[max_idx]:.3e}\n"
        f"cond={local_conds[max_idx]:.3e}\n"
        f"dist_b={dist_to_boundary[max_idx]:.3e}\n"
        f"boundary={bool(is_boundary[max_idx])}"
    )
    ax.annotate(txt, xy=(points[max_idx, 0], points[max_idx, 1]), xytext=(8, 8), textcoords="offset points")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(tag)
    fig.colorbar(sc, ax=ax, shrink=0.85, label="|error|")
    fig.tight_layout()
    fig.savefig(outdir / f"{tag}_max_error.png", dpi=220)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Run a case
# -----------------------------------------------------------------------------

def make_case_tag(domain_name: str, operator: str, h: float, family_seed: int, trial: int) -> str:
    return f"{domain_name}_{operator}_h{h:.4f}_trial{trial:02d}_family{family_seed}"


def solve_poisson_case(
    domain_name: str,
    operator: str,
    h: float,
    family_seed: int,
    trial: int,
    outdir: str,
    alpha: float,
    min_neighbors: int,
    max_neighbors: int,
    boundary_neighbor_boost: int,
    boundary_layer_factor: float,
    boundary_weight_boost: float,
    boundary_split_factor: float,
    selective_enrichment_neighbors: int,
    selective_core_factor: float,
    selective_cond_trigger: float,
    selective_sigma_trigger: float,
    selective_distance_factor: float,
    added_edge_penalty: float,
    force_min_flagged_core_nodes: int,
    force_relaxed_distance_factor: float,
    freeze_ccpfm_attempt: int,
    original_prior_anchor: float,
    core_prior_anchor: float,
    boundary_band_prior_anchor: float,
    added_prior_anchor: float,
    constraint_regularization: float,
    prior_blend_geom: float,
    prior_blend_wls: float,
    save_points: bool,
    plot_diagnostics: bool,
) -> Dict[str, float]:
    t0 = time.perf_counter()
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    domain = DOMAIN_FACTORIES[domain_name]()
    points, is_boundary = generate_point_cloud(domain, h=h, family_seed=family_seed)

    A, b, stats, diagnostics = assemble_operator_and_diagnostics(
        points=points,
        is_boundary=is_boundary,
        operator=operator,
        alpha=alpha,
        min_neighbors=min_neighbors,
        max_neighbors=max_neighbors,
        boundary_neighbor_boost=boundary_neighbor_boost,
        boundary_layer_factor=boundary_layer_factor,
        boundary_weight_boost=boundary_weight_boost,
        selective_enrichment_neighbors=selective_enrichment_neighbors,
        selective_core_factor=selective_core_factor,
        selective_cond_trigger=selective_cond_trigger,
        selective_sigma_trigger=selective_sigma_trigger,
        selective_distance_factor=selective_distance_factor,
        added_edge_penalty=added_edge_penalty,
        force_min_flagged_core_nodes=force_min_flagged_core_nodes,
        force_relaxed_distance_factor=force_relaxed_distance_factor,
        freeze_ccpfm_attempt=freeze_ccpfm_attempt,
        original_prior_anchor=original_prior_anchor,
        core_prior_anchor=core_prior_anchor,
        boundary_band_prior_anchor=boundary_band_prior_anchor,
        added_prior_anchor=added_prior_anchor,
        constraint_regularization=constraint_regularization,
        prior_blend_geom=prior_blend_geom,
        prior_blend_wls=prior_blend_wls,
    )
    t_asm = time.perf_counter()
    u_num = spsolve(A, b)
    t_solve = time.perf_counter()

    x, y = points[:, 0], points[:, 1]
    u_ex = exact_u(x, y)
    err = u_num - u_ex
    abs_err = np.abs(err)
    vols = diagnostics["volumes"]
    l2 = float(np.sqrt(np.sum(vols * err**2) / np.sum(vols)))
    linf = float(np.max(abs_err))
    rms = float(np.sqrt(np.mean(err**2)))
    max_idx = int(np.argmax(abs_err))
    interior = ~is_boundary
    dist_bdry = diagnostics["dist_to_boundary"]
    near_bdry_interior = interior & (dist_bdry <= boundary_split_factor * h)
    core_interior = interior & (dist_bdry > boundary_split_factor * h)

    def split_metrics(mask: Array) -> Tuple[float, float, float, int]:
        if not np.any(mask):
            return float("nan"), float("nan"), float("nan"), 0
        vm = vols[mask]
        em = err[mask]
        aem = abs_err[mask]
        l2m = float(np.sqrt(np.sum(vm * em**2) / np.sum(vm)))
        linfm = float(np.max(aem))
        rmsm = float(np.sqrt(np.mean(em**2)))
        return l2m, linfm, rmsm, int(np.sum(mask))

    l2_near, linf_near, rms_near, n_near = split_metrics(near_bdry_interior)
    l2_core, linf_core, rms_core, n_core = split_metrics(core_interior)
    l2_interior, linf_interior, rms_interior, n_interior_eval = split_metrics(interior)
    err2_int = np.sum(vols[interior] * err[interior] ** 2)
    err2_near = np.sum(vols[near_bdry_interior] * err[near_bdry_interior] ** 2) if n_near > 0 else 0.0
    frac_err2_near = float(err2_near / err2_int) if err2_int > 0 else float("nan")
    if is_boundary[max_idx]:
        max_region = "boundary"
    elif near_bdry_interior[max_idx]:
        max_region = "near_boundary_interior"
    else:
        max_region = "interior_core"

    tag = make_case_tag(domain_name, operator, h, family_seed, trial)
    if save_points:
        np.savez_compressed(
            out_path / f"{tag}.npz",
            points=points,
            is_boundary=is_boundary,
            u_num=u_num,
            u_exact=u_ex,
            error=err,
            abs_error=abs_err,
            radii=diagnostics["radii"],
            volumes=diagnostics["volumes"],
            dist_to_boundary=diagnostics["dist_to_boundary"],
            local_conds=diagnostics["local_conds"],
            local_sigmins=diagnostics["local_sigmins"],
            adjacency_sizes=diagnostics["adjacency_sizes"],
            near_bdry_interior=near_bdry_interior,
            core_interior=core_interior,
            lambdas=diagnostics.get("lambdas", np.array([], dtype=float)),
            edge_is_added=diagnostics.get("edge_is_added", np.array([], dtype=bool)),
            node_is_flagged=diagnostics.get("node_is_flagged", np.array([], dtype=bool)),
            node_is_force_flagged=diagnostics.get("node_is_force_flagged", np.array([], dtype=bool)),
            node_added_count=diagnostics.get("node_added_count", np.array([], dtype=int)),
        )
    if plot_diagnostics and (domain_name == "star" or linf > 2.5):
        maybe_plot_case_diagnostics(
            outdir=out_path,
            tag=tag,
            points=points,
            abs_err=abs_err,
            is_boundary=is_boundary,
            local_conds=diagnostics["local_conds"],
            dist_to_boundary=diagnostics["dist_to_boundary"],
        )

    # Unstable-case flags (outlier-aware group-level thresholds come later; this is per-run hard screening).
    unstable_reasons: List[str] = []
    if operator == "wls":
        if linf > 10.0:
            unstable_reasons.append("linf_large")
        if stats["max_local_cond"] > 1.0e5:
            unstable_reasons.append("high_condition")
        if stats["min_local_sigma_min"] < 1.0e-10:
            unstable_reasons.append("small_sigma")
        if max(stats["patch_const_max"], stats["patch_x_max"], stats["patch_y_max"], stats["patch_quad_max"]) > 1.0e-8:
            unstable_reasons.append("patch_failure")
    else:
        if max(stats["patch_x_max"], stats["patch_y_max"], stats["patch_quad_max"]) > 1.0e-6:
            unstable_reasons.append("patch_failure")
        if stats["constraint_resid_inf"] > 1.0e-8:
            unstable_reasons.append("constraint_failure")

    result = {
        "domain": domain_name,
        "operator": operator,
        "h_target": float(h),
        "family_seed": int(family_seed),
        "trial": int(trial),
        "l2": l2,
        "linf": linf,
        "rms": rms,
        "assembly_seconds": float(t_asm - t0),
        "solve_seconds": float(t_solve - t_asm),
        "total_seconds": float(t_solve - t0),
        "max_error_idx": max_idx,
        "max_error_x": float(points[max_idx, 0]),
        "max_error_y": float(points[max_idx, 1]),
        "max_error_is_boundary": bool(is_boundary[max_idx]),
        "max_error_local_cond": float(diagnostics["local_conds"][max_idx]) if not np.isnan(diagnostics["local_conds"][max_idx]) else float("nan"),
        "max_error_local_sigma_min": float(diagnostics["local_sigmins"][max_idx]) if not np.isnan(diagnostics["local_sigmins"][max_idx]) else float("nan"),
        "max_error_dist_boundary": float(diagnostics["dist_to_boundary"][max_idx]),
        "max_error_region": max_region,
        "frac_max_error_on_boundary": float(np.mean(is_boundary[np.argmax(abs_err)][None])),
        "boundary_split_factor": float(boundary_split_factor),
        "n_interior_eval": int(n_interior_eval),
        "n_near_boundary_interior": int(n_near),
        "n_core_interior": int(n_core),
        "l2_interior": l2_interior,
        "linf_interior": linf_interior,
        "rms_interior": rms_interior,
        "l2_near_boundary_interior": l2_near,
        "linf_near_boundary_interior": linf_near,
        "rms_near_boundary_interior": rms_near,
        "l2_core_interior": l2_core,
        "linf_core_interior": linf_core,
        "rms_core_interior": rms_core,
        "frac_err2_near_boundary_interior": frac_err2_near,
        "unstable": bool(len(unstable_reasons) > 0),
        "unstable_reasons": ";".join(unstable_reasons),
        **stats,
        "status": "ok",
    }
    with open(out_path / f"{tag}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


# -----------------------------------------------------------------------------
# Robust summaries
# -----------------------------------------------------------------------------

def _trimmed_mean(a: Array, frac: float = 0.2) -> float:
    a = np.asarray(a, dtype=float)
    if len(a) == 0:
        return float("nan")
    if len(a) < 3:
        return float(np.mean(a))
    k = int(np.floor(frac * len(a)))
    if 2 * k >= len(a):
        return float(np.mean(a))
    aa = np.sort(a)[k: len(a) - k]
    return float(np.mean(aa))


def _mad(a: Array) -> float:
    med = float(np.median(a))
    return float(np.median(np.abs(a - med)))


def fit_global_order(xs: Array, ys: Array) -> float:
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    mask = np.isfinite(xs) & np.isfinite(ys) & (xs > 0.0) & (ys > 0.0)
    if np.sum(mask) < 2:
        return float("nan")
    p = np.polyfit(np.log(xs[mask]), np.log(ys[mask]), 1)
    return float(p[0])


def _safe_nanmean(a: Array) -> float:
    a = np.asarray(a, dtype=float)
    mask = np.isfinite(a)
    return float(np.mean(a[mask])) if np.any(mask) else float("nan")


def _safe_nanmedian(a: Array) -> float:
    a = np.asarray(a, dtype=float)
    mask = np.isfinite(a)
    return float(np.median(a[mask])) if np.any(mask) else float("nan")


def aggregate_rows(rows: List[Dict[str, float]]) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    if not rows:
        return [], []
    groups: Dict[Tuple[str, str, float], List[Dict[str, float]]] = {}
    for r in rows:
        key = (str(r["domain"]), str(r["operator"]), float(r["h_target"]))
        groups.setdefault(key, []).append(r)

    agg: List[Dict[str, float]] = []
    unstable_rows: List[Dict[str, float]] = []
    by_domain_op: Dict[Tuple[str, str], List[Dict[str, float]]] = {}

    for key, grp in sorted(groups.items()):
        domain, operator, h = key
        l2 = np.array([float(r["l2"]) for r in grp], dtype=float)
        linf = np.array([float(r["linf"]) for r in grp], dtype=float)
        rms = np.array([float(r["rms"]) for r in grp], dtype=float)
        cond = np.array([float(r["max_local_cond"]) for r in grp], dtype=float)
        sigma = np.array([float(r["min_local_sigma_min"]) for r in grp], dtype=float)
        l2_med = np.median(l2)
        linf_med = np.median(linf)
        l2_mad = max(_mad(l2), 1.0e-14)
        linf_mad = max(_mad(linf), 1.0e-14)
        outlier_mask = (np.abs(l2 - l2_med) > 6.0 * l2_mad) | (np.abs(linf - linf_med) > 6.0 * linf_mad)
        n_unstable = 0
        for rr, is_out in zip(grp, outlier_mask):
            if bool(rr.get("unstable", False)) or bool(is_out):
                n_unstable += 1
                urow = dict(rr)
                if bool(is_out):
                    reason = urow.get("unstable_reasons", "")
                    extra = []
                    if abs(float(rr["l2"]) - l2_med) > 6.0 * l2_mad:
                        extra.append("l2_outlier")
                    if abs(float(rr["linf"]) - linf_med) > 6.0 * linf_mad:
                        extra.append("linf_outlier")
                    urow["unstable_reasons"] = ";".join([x for x in [reason, *extra] if x])
                unstable_rows.append(urow)

        row = {
            "domain": domain,
            "operator": operator,
            "h_target": h,
            "n_trials": len(grp),
            "mean_l2": float(np.mean(l2)),
            "std_l2": float(np.std(l2, ddof=0)),
            "median_l2": float(np.median(l2)),
            "trimmed_l2": _trimmed_mean(l2),
            "mean_linf": float(np.mean(linf)),
            "std_linf": float(np.std(linf, ddof=0)),
            "median_linf": float(np.median(linf)),
            "trimmed_linf": _trimmed_mean(linf),
            "median_rms": float(np.median(rms)),
            "n_unstable": int(n_unstable),
            "outlier_fraction": float(n_unstable / max(len(grp), 1)),
            "mean_max_local_cond": float(np.mean(cond)),
            "median_max_local_cond": float(np.median(cond)),
            "mean_min_local_sigma_min": float(np.mean(sigma)),
            "median_min_local_sigma_min": float(np.median(sigma)),
            "frac_max_error_on_boundary": float(np.mean([float(r["frac_max_error_on_boundary"]) for r in grp])),
        }
        # Carry mean patch / correction metrics.
        keys_to_average = [
            "patch_const_max", "patch_x_max", "patch_y_max", "patch_x2h_max", "patch_xy_max", "patch_y2h_max", "patch_quad_max",
            "constraint_resid_inf", "constraint_resid_l2", "correction_rel_l2", "correction_abs_median", "correction_abs_max",
            "correction_rel_median", "correction_rel_p90", "correction_rel_max", "negative_lambda_fraction", "lambda_ratio_median",
            "original_edge_fraction", "added_edge_fraction", "n_added_edges", "n_original_edges", "n_flagged_enrichment_nodes",
            "n_force_flagged_enrichment_nodes", "n_flagged_nodes_with_added_edges", "n_flagged_nodes_without_added_edges",
            "mean_added_edges_per_flagged_node", "max_added_edges_on_flagged_node",
            "selected_attempt_idx", "selected_attempt_alpha", "selected_attempt_min_neighbors", "selected_attempt_max_neighbors",
            "original_edge_correction_rel_median", "original_edge_correction_rel_p90", "original_edge_correction_rel_max", "original_edge_negative_lambda_fraction",
            "added_edge_correction_rel_median", "added_edge_correction_rel_p90", "added_edge_correction_rel_max", "added_edge_negative_lambda_fraction",
            "core_edge_correction_rel_median", "core_edge_correction_rel_p90", "core_edge_correction_rel_max", "core_edge_negative_lambda_fraction",
            "boundary_edge_correction_rel_median", "boundary_edge_correction_rel_p90", "boundary_edge_correction_rel_max", "boundary_edge_negative_lambda_fraction",
            "l2_interior", "linf_interior", "rms_interior",
            "l2_near_boundary_interior", "linf_near_boundary_interior", "rms_near_boundary_interior",
            "l2_core_interior", "linf_core_interior", "rms_core_interior",
            "frac_err2_near_boundary_interior", "n_near_boundary_interior", "n_core_interior",
        ]
        for kk in keys_to_average:
            vals = np.array([float(r.get(kk, np.nan)) for r in grp], dtype=float)
            row[f"mean_{kk}"] = _safe_nanmean(vals)
            row[f"median_{kk}"] = _safe_nanmedian(vals)
        agg.append(row)
        by_domain_op.setdefault((domain, operator), []).append(row)

    # Domain/operator global orders from median and trimmed metrics.
    for key, sub in by_domain_op.items():
        hs = np.array([float(r["h_target"]) for r in sub], dtype=float)
        med_l2 = np.array([float(r["median_l2"]) for r in sub], dtype=float)
        med_linf = np.array([float(r["median_linf"]) for r in sub], dtype=float)
        tr_l2 = np.array([float(r["trimmed_l2"]) for r in sub], dtype=float)
        tr_linf = np.array([float(r["trimmed_linf"]) for r in sub], dtype=float)
        near_l2 = np.array([float(r.get("median_l2_near_boundary_interior", np.nan)) for r in sub], dtype=float)
        core_l2 = np.array([float(r.get("median_l2_core_interior", np.nan)) for r in sub], dtype=float)
        near_linf = np.array([float(r.get("median_linf_near_boundary_interior", np.nan)) for r in sub], dtype=float)
        core_linf = np.array([float(r.get("median_linf_core_interior", np.nan)) for r in sub], dtype=float)
        order_l2_med = fit_global_order(hs, med_l2)
        order_linf_med = fit_global_order(hs, med_linf)
        order_l2_trim = fit_global_order(hs, tr_l2)
        order_linf_trim = fit_global_order(hs, tr_linf)
        order_l2_near = fit_global_order(hs, near_l2)
        order_l2_core = fit_global_order(hs, core_l2)
        order_linf_near = fit_global_order(hs, near_linf)
        order_linf_core = fit_global_order(hs, core_linf)
        for r in sub:
            r["global_order_l2_median"] = order_l2_med
            r["global_order_linf_median"] = order_linf_med
            r["global_order_l2_trimmed"] = order_l2_trim
            r["global_order_linf_trimmed"] = order_linf_trim
            r["global_order_l2_near_boundary_interior_median"] = order_l2_near
            r["global_order_l2_core_interior_median"] = order_l2_core
            r["global_order_linf_near_boundary_interior_median"] = order_linf_near
            r["global_order_linf_core_interior_median"] = order_linf_core

    return agg, unstable_rows


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------

def save_csv(rows: List[Dict[str, float]], filename: Path) -> None:
    filename.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        filename.write_text("", encoding="utf-8")
        return
    keys = sorted(set().union(*(r.keys() for r in rows)))
    with open(filename, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def save_json(rows: List[Dict[str, float]], filename: Path) -> None:
    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


# -----------------------------------------------------------------------------
# CLI and driver
# -----------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CCPFM / WLS Poisson validation on irregular 2D geometries (prior-preserving core-regularization branch).")
    p.add_argument("--domains", nargs="+", default=["circle", "ellipse", "star", "annulus"], choices=list(DOMAIN_FACTORIES.keys()))
    p.add_argument("--operators", nargs="+", default=["wls", "ccpfm"], choices=["wls", "ccpfm", "experimental_conservative"])
    p.add_argument("--hs", nargs="+", type=float, default=[0.20, 0.16, 0.12, 0.10, 0.08])
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--outdir", type=str, default="/mnt/data/ccpfm_poisson_results_prior_preserving_core_reg")
    p.add_argument("--alpha", type=float, default=2.0)
    p.add_argument("--min-neighbors", type=int, default=18)
    p.add_argument("--max-neighbors", type=int, default=56)
    p.add_argument("--boundary-neighbor-boost", type=int, default=10)
    p.add_argument("--boundary-layer-factor", type=float, default=2.25)
    p.add_argument("--boundary-weight-boost", type=float, default=1.8)
    p.add_argument("--boundary-split-factor", type=float, default=2.0, help="Interior nodes with dist(boundary) <= factor*h are classified as near-boundary.")
    p.add_argument("--selective-enrichment-neighbors", type=int, default=0, help="Very sparse rescue edges added only for flagged interior-core nodes in CCPFm. Use 0 to freeze the graph and study only the correction objective.")
    p.add_argument("--selective-core-factor", type=float, default=2.0, help="dist(boundary) > factor*h_ref defines interior-core nodes for selective enrichment and diagnostics.")
    p.add_argument("--selective-cond-trigger", type=float, default=2.5e4, help="Flag CCPFm nodes for rescue-edge enrichment when local cloud condition exceeds this value.")
    p.add_argument("--selective-sigma-trigger", type=float, default=2.5e-4, help="Flag CCPFm nodes for rescue-edge enrichment when local sigma_min falls below this value.")
    p.add_argument("--selective-distance-factor", type=float, default=3.0, help="Maximum rescue-edge length as a multiple of local reference spacing.")
    p.add_argument("--added-edge-penalty", type=float, default=12.0, help="Quadratic correction penalty multiplier applied to added rescue edges.")
    p.add_argument("--force-min-flagged-core-nodes", type=int, default=0, help="If the absolute enrichment triggers flag too few deep-core nodes, force-flag at least this many worst-scoring core nodes so rescue-edge enrichment is exercised.")
    p.add_argument("--force-relaxed-distance-factor", type=float, default=5.0, help="Relaxed rescue-edge distance cap used only after the strict selective-distance-factor search fails for a flagged node.")
    p.add_argument("--freeze-ccpfm-attempt", type=int, default=0, help="Set to 0,1,2 to freeze the CCPFm graph attempt; use -1 to keep the original best-attempt selection.")
    p.add_argument("--original-prior-anchor", type=float, default=1.0, help="Multiplier on quadratic anchoring weights for original edges in the CCPFm correction solve.")
    p.add_argument("--core-prior-anchor", type=float, default=1.0, help="Additional multiplier on quadratic anchoring weights for core edges in the CCPFm correction solve.")
    p.add_argument("--boundary-band-prior-anchor", type=float, default=1.0, help="Additional multiplier on quadratic anchoring weights for boundary-band edges in the CCPFm correction solve.")
    p.add_argument("--added-prior-anchor", type=float, default=1.0, help="Additional multiplier on quadratic anchoring weights for added rescue edges in the CCPFm correction solve.")
    p.add_argument("--constraint-regularization", type=float, default=1.0e-13, help="Tiny diagonal regularization added to the Schur complement in the CCPFm correction solve.")
    p.add_argument("--prior-blend-geom", type=float, default=0.35, help="Geometric-prior weight in the original-edge geometric/WLS prior blend.")
    p.add_argument("--prior-blend-wls", type=float, default=0.65, help="WLS-prior weight in the original-edge geometric/WLS prior blend.")
    p.add_argument("--family-seed-base", type=int, default=1000)
    p.add_argument("--save-points", action="store_true")
    p.add_argument("--plot-diagnostics", action="store_true")
    return p.parse_args(argv)


def normalize_operators(op_list: Sequence[str]) -> List[str]:
    out: List[str] = []
    for op in op_list:
        if op == "experimental_conservative":
            out.append("ccpfm")
        else:
            out.append(op)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    operators = normalize_operators(args.operators)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for domain in args.domains:
        for trial in range(args.trials):
            family_seed = args.family_seed_base + 1000 * list(DOMAIN_FACTORIES.keys()).index(domain) + trial
            for h in args.hs:
                for op in operators:
                    tasks.append(
                        dict(
                            domain_name=domain,
                            operator=op,
                            h=float(h),
                            family_seed=int(family_seed),
                            trial=int(trial),
                            outdir=str(outdir),
                            alpha=float(args.alpha),
                            min_neighbors=int(args.min_neighbors),
                            max_neighbors=int(args.max_neighbors),
                            boundary_neighbor_boost=int(args.boundary_neighbor_boost),
                            boundary_layer_factor=float(args.boundary_layer_factor),
                            boundary_weight_boost=float(args.boundary_weight_boost),
                            boundary_split_factor=float(args.boundary_split_factor),
                            selective_enrichment_neighbors=int(args.selective_enrichment_neighbors),
                            selective_core_factor=float(args.selective_core_factor),
                            selective_cond_trigger=float(args.selective_cond_trigger),
                            selective_sigma_trigger=float(args.selective_sigma_trigger),
                            selective_distance_factor=float(args.selective_distance_factor),
                            added_edge_penalty=float(args.added_edge_penalty),
                            force_min_flagged_core_nodes=int(args.force_min_flagged_core_nodes),
                            force_relaxed_distance_factor=float(args.force_relaxed_distance_factor),
                            freeze_ccpfm_attempt=int(args.freeze_ccpfm_attempt),
                            original_prior_anchor=float(args.original_prior_anchor),
                            core_prior_anchor=float(args.core_prior_anchor),
                            boundary_band_prior_anchor=float(args.boundary_band_prior_anchor),
                            added_prior_anchor=float(args.added_prior_anchor),
                            constraint_regularization=float(args.constraint_regularization),
                            prior_blend_geom=float(args.prior_blend_geom),
                            prior_blend_wls=float(args.prior_blend_wls),
                            save_points=bool(args.save_points),
                            plot_diagnostics=bool(args.plot_diagnostics),
                        )
                    )

    print(f"Running {len(tasks)} Poisson cases -> {outdir}")
    rows: List[Dict[str, float]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        fut_map = {ex.submit(solve_poisson_case, **task): task for task in tasks}
        for fut in as_completed(fut_map):
            task = fut_map[fut]
            try:
                row = fut.result()
                rows.append(row)
                extra_msg = ""
                if row["operator"] == "ccpfm":
                    extra_msg = (
                        f" flags={int(row.get('n_flagged_enrichment_nodes', 0))}"
                        f" forced={int(row.get('n_force_flagged_enrichment_nodes', 0))}"
                        f" hit={int(row.get('n_flagged_nodes_with_added_edges', 0))}"
                        f" added={int(row.get('n_added_edges', 0))}"
                        f" att={int(row.get('selected_attempt_idx', -1))}"
                    )
                print(
                    f"[done] {row['domain']:>8s} {row['operator']:>24s} h={row['h_target']:.4f} "
                    f"N={row['n_points']:4d} L2={row['l2']:.3e} Linf={row['linf']:.3e}{extra_msg}"
                )
            except Exception as exc:  # pragma: no cover
                print(f"[fail] {task['domain_name']} {task['operator']} h={task['h']}: {exc}")
                traceback.print_exc()

    case_csv = outdir / "summary_cases.csv"
    case_json = outdir / "summary_cases.json"
    save_csv(rows, case_csv)
    save_json(rows, case_json)

    agg_rows, unstable_rows = aggregate_rows(rows)
    save_csv(agg_rows, outdir / "summary_aggregate.csv")
    save_json(agg_rows, outdir / "summary_aggregate.json")
    save_csv(unstable_rows, outdir / "summary_unstable_cases.csv")
    save_json(unstable_rows, outdir / "summary_unstable_cases.json")

    print("\nPatch-test inspection (mean max interior residual):")
    for r in sorted(agg_rows, key=lambda z: (z["domain"], z["operator"], z["h_target"])):
        print(
            f"{r['domain']:>10s} {r['operator']:>24s} h={r['h_target']:.4f} | "
            f"const={r['mean_patch_const_max']:.2e}, x={r['mean_patch_x_max']:.2e}, "
            f"y={r['mean_patch_y_max']:.2e}, quad={r['mean_patch_quad_max']:.2e}"
        )

    print("\nReference WLS operator summary:")
    for r in sorted([a for a in agg_rows if a['operator'] == 'wls'], key=lambda z: z['domain']):
        if float(r['h_target']) == min(args.hs):
            print(
                f"{r['domain']:>10s}: h={r['h_target']:.4f}, median L2={r['median_l2']:.3e}, "
                f"median Linf={r['median_linf']:.3e}, global order(L2, median)={r['global_order_l2_median']:.3f}, "
                f"unstable={int(r['n_unstable'])}/{int(r['n_trials'])}"
            )

    print("\nCCPFM prior-preserving branch summary:")
    for r in sorted([a for a in agg_rows if a['operator'] == 'ccpfm'], key=lambda z: z['domain']):
        if float(r['h_target']) == min(args.hs):
            print(
                f"{r['domain']:>10s}: h={r['h_target']:.4f}, median L2={r['median_l2']:.3e}, "
                f"median Linf={r['median_linf']:.3e}, global order(L2, median)={r['global_order_l2_median']:.3f}, "
                f"mean patch quad={r['mean_patch_quad_max']:.2e}, mean neg(lambda)={r['mean_negative_lambda_fraction']:.3f}, "
                f"flagged={r.get('mean_n_flagged_enrichment_nodes', float('nan')):.2f}, force-flagged={r.get('mean_n_force_flagged_enrichment_nodes', float('nan')):.2f}, "
                f"flagged->added={r.get('mean_n_flagged_nodes_with_added_edges', float('nan')):.2f}, added edges={r.get('mean_n_added_edges', float('nan')):.2f}, "
                f"added frac={r.get('mean_added_edge_fraction', float('nan')):.3f}, orig corr p90={r.get('mean_original_edge_correction_rel_p90', float('nan')):.2f}, "
                f"orig-core corr p90={r.get('mean_original_core_edge_correction_rel_p90', float('nan')):.2f}, added corr p90={r.get('mean_added_edge_correction_rel_p90', float('nan')):.2f}, "
                f"attempt={r.get('mean_selected_attempt_idx', float('nan')):.2f}"
            )

    print("\nGraph-activation summary (mean counts at finest h):")
    for r in sorted([a for a in agg_rows if a['operator'] == 'ccpfm' and float(a['h_target']) == min(args.hs)], key=lambda z: z['domain']):
        print(
            f"  {r['domain']:>8s}: flagged={r.get('mean_n_flagged_enrichment_nodes', float('nan')):.2f}, "
            f"force-flagged={r.get('mean_n_force_flagged_enrichment_nodes', float('nan')):.2f}, "
            f"flagged->added={r.get('mean_n_flagged_nodes_with_added_edges', float('nan')):.2f}, "
            f"flagged no-add={r.get('mean_n_flagged_nodes_without_added_edges', float('nan')):.2f}, "
            f"added edges={r.get('mean_n_added_edges', float('nan')):.2f}, "
            f"added/flagged={r.get('mean_mean_added_edges_per_flagged_node', float('nan')):.2f}, "
            f"attempt={r.get('mean_selected_attempt_idx', float('nan')):.2f}"
        )

    print("\nBoundary/interior split summary (median L2 at finest h):")
    finest_h = min(args.hs)
    for op in ["wls", "ccpfm"]:
        label = "Reference WLS" if op == "wls" else "CCPFM"
        print(f"  {label}:")
        for r in sorted([a for a in agg_rows if a["operator"] == op and float(a["h_target"]) == finest_h], key=lambda z: z["domain"]):
            print(
                f"    {r['domain']:>8s}: near-boundary L2={r.get('median_l2_near_boundary_interior', float('nan')):.3e}, "
                f"core L2={r.get('median_l2_core_interior', float('nan')):.3e}, "
                f"frac(err^2 near)={r.get('mean_frac_err2_near_boundary_interior', float('nan')):.3f}, "
                f"order_near={r.get('global_order_l2_near_boundary_interior_median', float('nan')):.3f}, "
                f"order_core={r.get('global_order_l2_core_interior_median', float('nan')):.3f}"
            )

    print("\nAll cases completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
