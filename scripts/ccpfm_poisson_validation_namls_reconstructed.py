#!/usr/bin/env python3
"""Reconstructed NAMLS comparator for CAMWA-D-26-00757.

This script reuses the point-cloud generator and reporting utilities from
``ccpfm_poisson_validation_prior_preserving_core_reg_complete.py`` and adds a
neighbor-adaptive normalized moving-least-squares Laplacian.

The historical NAMLS source was not recovered.  This implementation follows
exactly the description retained in the manuscript:

* quadratic polynomial reproduction;
* normalized local coordinates;
* a weighting strategy distinct from the WLS reference;
* small Tikhonov regularization;
* multiple neighbor configurations;
* local selection by a score combining condition number and minimum singular
  value.

All reconstruction parameters are written into every case row.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import time
import traceback
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

Array = np.ndarray
EPS = 1.0e-14


def load_base_module(path: str):
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(p)
    spec = importlib.util.spec_from_file_location("ccpfm_base_complete", p)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import base benchmark from {p}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def wendland_c2(q: Array) -> Array:
    """Compact C2 kernel on [0,1], with a tiny positive floor."""
    z = np.clip(1.0 - q, 0.0, None)
    return z**4 * (4.0 * q + 1.0) + 1.0e-14


def gaussian_weight(q: Array, beta: float) -> Array:
    return np.exp(-beta * q * q) + 1.0e-14


def namls_stencil(
    center: Array,
    neighbors: Array,
    regularization: float,
    kernel: str,
    kernel_beta: float,
) -> Tuple[Array, float, float, float]:
    """Return center-plus-neighbor Laplacian coefficients and diagnostics.

    The local polynomial basis is [1, xi, eta, xi^2, xi*eta, eta^2].
    Coordinates are normalized by the maximum neighbor radius.  The weighted
    design columns are additionally normalized before the regularized solve.
    """
    pts = np.vstack([center[None, :], neighbors])
    dxy = pts - center[None, :]
    r = np.linalg.norm(dxy, axis=1)
    scale = max(float(np.max(r[1:])) if len(r) > 1 else 1.0, 1.0e-12)
    xi = dxy[:, 0] / scale
    eta = dxy[:, 1] / scale
    q = r / scale

    if kernel == "wendland_c2":
        w = wendland_c2(q)
    elif kernel == "gaussian":
        w = gaussian_weight(q, kernel_beta)
    else:
        raise ValueError(f"Unknown NAMLS kernel: {kernel}")
    w[0] = max(float(w[0]), 1.0)

    P = np.column_stack([
        np.ones(len(pts)), xi, eta, xi * xi, xi * eta, eta * eta,
    ])
    sqrtw = np.sqrt(w)
    Pw = P * sqrtw[:, None]

    # Column normalization is the 'normalized' part of NAMLS.  It reduces
    # disparities between constant, linear, and quadratic columns.
    col_norm = np.linalg.norm(Pw, axis=0)
    col_norm = np.maximum(col_norm, 1.0e-14)
    Qw = Pw / col_norm[None, :]

    try:
        s = np.linalg.svd(Qw, compute_uv=False)
        sigma_min = float(s[-1]) if len(s) else 0.0
        cond = float(s[0] / max(sigma_min, EPS)) if len(s) else float("inf")
    except np.linalg.LinAlgError:
        sigma_min = 0.0
        cond = float("inf")

    # P^T c = ell.  Since P = Q diag(col_norm/sqrt(w)-consistent in the
    # weighted design), the normalized target is ell/col_norm.
    ell = np.array([0.0, 0.0, 0.0, 2.0 / scale**2, 0.0, 2.0 / scale**2])
    ell_n = ell / col_norm
    gram = Qw.T @ Qw
    reg_scale = max(float(np.trace(gram)) / gram.shape[0], 1.0)
    gram_reg = gram + regularization * reg_scale * np.eye(gram.shape[0])
    try:
        coeff_n = np.linalg.solve(gram_reg, ell_n)
    except np.linalg.LinAlgError:
        coeff_n = np.linalg.lstsq(gram_reg, ell_n, rcond=1.0e-13)[0]
    c = sqrtw * (Qw @ coeff_n)

    # Reproduction defect used as a secondary diagnostic/tie-breaker.
    reproduction = P.T @ c - ell
    repro_inf = float(np.max(np.abs(reproduction)))
    return np.asarray(c, dtype=float), cond, sigma_min, repro_inf


def candidate_neighbor_counts(min_neighbors: int, max_neighbors: int, step: int) -> List[int]:
    start = max(8, int(min_neighbors))
    stop = max(start, int(max_neighbors))
    vals = list(range(start, stop + 1, max(1, int(step))))
    if vals[-1] != stop:
        vals.append(stop)
    return sorted(set(vals))


def build_namls_node_stencil(
    i: int,
    points: Array,
    tree: cKDTree,
    is_boundary: Array,
    dist_to_boundary: Array,
    h_target: float,
    min_neighbors: int,
    max_neighbors: int,
    neighbor_step: int,
    boundary_neighbor_boost: int,
    boundary_layer_factor: float,
    regularization: float,
    kernel: str,
    kernel_beta: float,
    size_penalty: float,
) -> Tuple[Array, Array, float, float, float, float]:
    n_available = len(points) - 1
    near_boundary = dist_to_boundary[i] <= boundary_layer_factor * h_target
    boost = boundary_neighbor_boost if near_boundary else 0
    counts = candidate_neighbor_counts(
        min(min_neighbors + boost, n_available),
        min(max_neighbors + boost, n_available),
        neighbor_step,
    )
    kmax = min(max(counts), n_available)
    _, idx = tree.query(points[i], k=min(kmax + 1, len(points)))
    idx = np.atleast_1d(idx).astype(int)
    idx = idx[idx != i]

    best = None
    best_score = float("inf")
    for k in counts:
        k = min(k, len(idx))
        if k < 6:
            continue
        nbrs = idx[:k]
        c, cond, sigma_min, repro_inf = namls_stencil(
            points[i], points[nbrs], regularization, kernel, kernel_beta
        )
        # Manuscript-retained score: conditioning and minimum singular value.
        # A mild size penalty avoids always selecting the largest stencil.
        score = (cond / max(sigma_min, 1.0e-16)) * (1.0 + size_penalty * (k / max(counts[0], 1))**2)
        # Strongly reject visibly defective reproduction.
        score *= 1.0 + min(1.0e8, repro_inf / 1.0e-10)
        if score < best_score:
            best_score = score
            best = (nbrs, c, cond, sigma_min, repro_inf, score)
    if best is None:
        raise RuntimeError(f"No admissible NAMLS stencil at node {i}")
    return best


def assemble_namls(
    base: Any,
    points: Array,
    is_boundary: Array,
    h_target: float,
    min_neighbors: int,
    max_neighbors: int,
    neighbor_step: int,
    boundary_neighbor_boost: int,
    boundary_layer_factor: float,
    regularization: float,
    kernel: str,
    kernel_beta: float,
    size_penalty: float,
) -> Tuple[csr_matrix, Array, Dict[str, float], Dict[str, Array]]:
    n = len(points)
    tree = cKDTree(points)
    radii = base.estimate_radii(points, tree)
    volumes = base.compute_node_volumes(radii)
    dist_to_boundary = base.compute_boundary_distances(points, is_boundary)
    x, y = points[:, 0], points[:, 1]
    u_bdry = base.exact_u(x, y)
    f_rhs = base.rhs_f(x, y)

    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    b = np.zeros(n, dtype=float)
    local_conds = np.full(n, np.nan)
    local_sigmins = np.full(n, np.nan)
    local_sizes = np.full(n, np.nan)
    local_repro = np.full(n, np.nan)
    local_scores = np.full(n, np.nan)

    for i in range(n):
        if is_boundary[i]:
            rows.append(i); cols.append(i); data.append(1.0)
            b[i] = u_bdry[i]
            continue
        nbrs, c, cond, sigma, repro, score = build_namls_node_stencil(
            i=i, points=points, tree=tree, is_boundary=is_boundary,
            dist_to_boundary=dist_to_boundary, h_target=h_target,
            min_neighbors=min_neighbors, max_neighbors=max_neighbors,
            neighbor_step=neighbor_step,
            boundary_neighbor_boost=boundary_neighbor_boost,
            boundary_layer_factor=boundary_layer_factor,
            regularization=regularization, kernel=kernel,
            kernel_beta=kernel_beta, size_penalty=size_penalty,
        )
        local_conds[i] = cond
        local_sigmins[i] = sigma
        local_sizes[i] = len(nbrs) + 1
        local_repro[i] = repro
        local_scores[i] = score
        local_idx = np.concatenate([[i], nbrs])
        # PDE is -Delta u = f.
        for j, coeff in zip(local_idx, -c):
            rows.append(i); cols.append(int(j)); data.append(float(coeff))
        b[i] = f_rhs[i]

    A = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    patches = base.patch_test_metrics(A, points, is_boundary)
    interior = ~is_boundary
    stats: Dict[str, float] = {
        **patches,
        "constraint_resid_inf": 0.0,
        "constraint_resid_l2": 0.0,
        "correction_rel_l2": 0.0,
        "negative_lambda_fraction": 0.0,
        "lambda_ratio_median": 1.0,
        "median_local_cond": float(np.nanmedian(local_conds[interior])),
        "max_local_cond": float(np.nanmax(local_conds[interior])),
        "min_local_sigma_min": float(np.nanmin(local_sigmins[interior])),
        "median_local_sigma_min": float(np.nanmedian(local_sigmins[interior])),
        "mean_local_stencil_size": float(np.nanmean(local_sizes[interior])),
        "max_namls_reproduction_defect": float(np.nanmax(local_repro[interior])),
        "median_namls_selection_score": float(np.nanmedian(local_scores[interior])),
    }
    diagnostics = {
        "radii": radii,
        "volumes": volumes,
        "dist_to_boundary": dist_to_boundary,
        "local_conds": local_conds,
        "local_sigmins": local_sigmins,
        "adjacency_sizes": local_sizes,
    }
    return A, b, stats, diagnostics


def solve_case(task: Dict[str, Any]) -> Dict[str, Any]:
    base = load_base_module(task["base_script"])
    t0 = time.perf_counter()
    domain = base.DOMAIN_FACTORIES[task["domain_name"]]()
    points, is_boundary = base.generate_point_cloud(
        domain, h=task["h"], family_seed=task["family_seed"]
    )
    A, b, stats, diag = assemble_namls(
        base=base, points=points, is_boundary=is_boundary,
        h_target=task["h"], min_neighbors=task["min_neighbors"],
        max_neighbors=task["max_neighbors"], neighbor_step=task["neighbor_step"],
        boundary_neighbor_boost=task["boundary_neighbor_boost"],
        boundary_layer_factor=task["boundary_layer_factor"],
        regularization=task["regularization"], kernel=task["kernel"],
        kernel_beta=task["kernel_beta"], size_penalty=task["size_penalty"],
    )
    t_asm = time.perf_counter()
    u_num = spsolve(A, b)
    t_solve = time.perf_counter()

    x, y = points[:, 0], points[:, 1]
    u_ex = base.exact_u(x, y)
    err = u_num - u_ex
    abs_err = np.abs(err)
    vols = diag["volumes"]
    l2 = float(np.sqrt(np.sum(vols * err**2) / np.sum(vols)))
    linf = float(np.max(abs_err))
    rms = float(np.sqrt(np.mean(err**2)))
    interior = ~is_boundary
    dist = diag["dist_to_boundary"]
    near = interior & (dist <= task["boundary_split_factor"] * task["h"])
    core = interior & ~near

    def split(mask: Array) -> Tuple[float, float, float, int]:
        if not np.any(mask):
            return float("nan"), float("nan"), float("nan"), 0
        vm, em = vols[mask], err[mask]
        return (
            float(np.sqrt(np.sum(vm * em**2) / np.sum(vm))),
            float(np.max(np.abs(em))),
            float(np.sqrt(np.mean(em**2))), int(np.sum(mask)),
        )

    l2_near, linf_near, rms_near, n_near = split(near)
    l2_core, linf_core, rms_core, n_core = split(core)
    l2_int, linf_int, rms_int, n_int = split(interior)
    err2_int = float(np.sum(vols[interior] * err[interior]**2))
    err2_near = float(np.sum(vols[near] * err[near]**2)) if n_near else 0.0

    max_idx = int(np.argmax(abs_err))
    frac_max_error_on_boundary = float(bool(is_boundary[max_idx]))

    unstable_reasons: List[str] = []
    if linf > 10.0:
        unstable_reasons.append("linf_large")
    if stats["max_local_cond"] > task["unstable_cond"]:
        unstable_reasons.append("high_condition")
    if stats["min_local_sigma_min"] < task["unstable_sigma"]:
        unstable_reasons.append("small_sigma")
    if max(stats["patch_const_max"], stats["patch_x_max"], stats["patch_y_max"], stats["patch_quad_max"]) > task["unstable_patch"]:
        unstable_reasons.append("patch_failure")

    result: Dict[str, Any] = {
        "domain": task["domain_name"], "operator": "namls",
        "method_id": "NAMLS-reconstructed-v1",
        "h_target": float(task["h"]), "family_seed": int(task["family_seed"]),
        "trial": int(task["trial"]), "n_points": int(len(points)),
        "n_boundary": int(np.sum(is_boundary)),
        "l2": l2, "linf": linf, "rms": rms,
        "max_error_idx": max_idx,
        "max_error_x": float(points[max_idx, 0]),
        "max_error_y": float(points[max_idx, 1]),
        "max_error_is_boundary": bool(is_boundary[max_idx]),
        "frac_max_error_on_boundary": frac_max_error_on_boundary,
        "assembly_seconds": float(t_asm - t0),
        "solve_seconds": float(t_solve - t_asm),
        "total_seconds": float(t_solve - t0),
        "n_interior_eval": n_int, "n_near_boundary_interior": n_near,
        "n_core_interior": n_core,
        "l2_interior": l2_int, "linf_interior": linf_int, "rms_interior": rms_int,
        "l2_near_boundary_interior": l2_near,
        "linf_near_boundary_interior": linf_near,
        "rms_near_boundary_interior": rms_near,
        "l2_core_interior": l2_core, "linf_core_interior": linf_core,
        "rms_core_interior": rms_core,
        "frac_err2_near_boundary_interior": err2_near / err2_int if err2_int > 0 else float("nan"),
        "unstable": bool(unstable_reasons),
        "unstable_reasons": ";".join(unstable_reasons),
        "namls_kernel": task["kernel"],
        "namls_kernel_beta": float(task["kernel_beta"]),
        "namls_regularization": float(task["regularization"]),
        "namls_min_neighbors": int(task["min_neighbors"]),
        "namls_max_neighbors": int(task["max_neighbors"]),
        "namls_neighbor_step": int(task["neighbor_step"]),
        "namls_size_penalty": float(task["size_penalty"]),
        **stats, "status": "ok",
    }
    out = Path(task["outdir"])
    out.mkdir(parents=True, exist_ok=True)
    tag = base.make_case_tag(task["domain_name"], "namls", task["h"], task["family_seed"], task["trial"])
    base.save_json(result, out / f"{tag}.json") if False else None
    # save_json in base expects a list; write directly.
    import json
    with (out / f"{tag}.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-script", default="ccpfm_poisson_validation_prior_preserving_core_reg_complete.py")
    p.add_argument("--domains", nargs="+", default=["circle", "ellipse", "star", "annulus"])
    p.add_argument("--hs", nargs="+", type=float, default=[0.20, 0.16, 0.12, 0.10, 0.08])
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)//2))
    p.add_argument("--outdir", default="original_results/namls")
    p.add_argument("--family-seed-base", type=int, default=1000)
    p.add_argument("--min-neighbors", type=int, default=14)
    p.add_argument("--max-neighbors", type=int, default=26)
    p.add_argument("--neighbor-step", type=int, default=4)
    p.add_argument("--boundary-neighbor-boost", type=int, default=2)
    p.add_argument("--boundary-layer-factor", type=float, default=2.25)
    p.add_argument("--boundary-split-factor", type=float, default=2.0)
    p.add_argument("--regularization", type=float, default=1.0e-10)
    p.add_argument("--kernel", choices=["wendland_c2", "gaussian"], default="wendland_c2")
    p.add_argument("--kernel-beta", type=float, default=3.0)
    p.add_argument("--size-penalty", type=float, default=2.0)
    p.add_argument("--unstable-cond", type=float, default=1.0e6)
    p.add_argument("--unstable-sigma", type=float, default=1.0e-10)
    p.add_argument("--unstable-patch", type=float, default=1.0e-6)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    base_path = str(Path(args.base_script).expanduser().resolve())
    base = load_base_module(base_path)
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    tasks: List[Dict[str, Any]] = []
    domain_names = list(base.DOMAIN_FACTORIES.keys())
    for domain in args.domains:
        if domain not in base.DOMAIN_FACTORIES:
            raise ValueError(f"Unknown domain {domain!r}")
        for trial in range(args.trials):
            seed = args.family_seed_base + 1000 * domain_names.index(domain) + trial
            for h in args.hs:
                tasks.append({
                    "base_script": base_path, "domain_name": domain,
                    "h": float(h), "family_seed": int(seed), "trial": int(trial),
                    "outdir": str(outdir), "min_neighbors": args.min_neighbors,
                    "max_neighbors": args.max_neighbors, "neighbor_step": args.neighbor_step,
                    "boundary_neighbor_boost": args.boundary_neighbor_boost,
                    "boundary_layer_factor": args.boundary_layer_factor,
                    "boundary_split_factor": args.boundary_split_factor,
                    "regularization": args.regularization, "kernel": args.kernel,
                    "kernel_beta": args.kernel_beta, "size_penalty": args.size_penalty,
                    "unstable_cond": args.unstable_cond, "unstable_sigma": args.unstable_sigma,
                    "unstable_patch": args.unstable_patch,
                })

    print(f"Running {len(tasks)} reconstructed NAMLS cases -> {outdir}")
    print(f"NAMLS config: kernel={args.kernel}, reg={args.regularization:.1e}, neighbors={args.min_neighbors}:{args.neighbor_step}:{args.max_neighbors}")
    rows: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        future_map = {ex.submit(solve_case, t): t for t in tasks}
        for n, fut in enumerate(as_completed(future_map), start=1):
            task = future_map[fut]
            try:
                row = fut.result(); rows.append(row)
                print(f"[{n:3d}/{len(tasks)}] {row['domain']:8s} namls h={row['h_target']:.3f} N={row['n_points']:4d} L2={row['l2']:.3e} Linf={row['linf']:.3e}")
            except Exception as exc:
                print(f"[fail] {task['domain_name']} h={task['h']}: {exc}")
                traceback.print_exc()
                raise

    rows.sort(key=lambda r: (r["domain"], r["trial"], -r["h_target"]))
    base.save_csv(rows, outdir / "summary_cases.csv")
    base.save_json(rows, outdir / "summary_cases.json")
    agg, unstable = base.aggregate_rows(rows)
    base.save_csv(agg, outdir / "summary_aggregate.csv")
    base.save_json(agg, outdir / "summary_aggregate.json")
    base.save_csv(unstable, outdir / "summary_unstable_cases.csv")
    base.save_json(unstable, outdir / "summary_unstable_cases.json")
    print(f"Wrote {len(rows)} cases and {len(agg)} aggregate rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
