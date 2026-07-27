#!/usr/bin/env python3
"""Regenerate the 17 figures referenced by CAMWA-D-26-00757.

The script reads case-level CSV files produced by the CCPFM benchmark branches.
It never fabricates missing curves: by default it validates that every required
method/domain/metric is present and exits with a clear error if anything is
missing.

Expected inputs
---------------
1. Frozen/base results: contains WLS and frozen CCPFM rows.
2. NAMLS results: contains NAMLS rows (may be the same file as the base input).
3. Livegate results: contains the live-boundary CCPFM rows.
4. Forced sparsecore results: contains the force-driven enrichment rows.
5. Selective sparsecore results: contains the natural selective-enrichment rows.

Each argument may point either to a CSV file or to a directory containing
``summary_cases.csv``. Aggregate CSVs are not sufficient for activated-run-only
statistics; use case-level results.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DOMAINS = ("circle", "ellipse", "star", "annulus")
METHOD_ORDER = ("WLS", "NAMLS", "CCPFM-frozen", "CCPFM-livegate", "CCPFM-sparsecore", "CCPFM-selective")
METHOD_SHORT = {
    "WLS": "WLS",
    "NAMLS": "NAMLS",
    "CCPFM-frozen": "Frozen",
    "CCPFM-livegate": "Livegate",
    "CCPFM-sparsecore": "Sparse forced",
    "CCPFM-selective": "Sparse selective",
}
REQUIRED_NUMERIC = ("h_target", "l2", "linf")
OPTIONAL_NUMERIC = (
    "l2_near_boundary_interior", "l2_core_interior", "patch_quad_max",
    "boundary_band_live", "n_added_edges", "n_flagged_enrichment_nodes",
    "n_force_flagged_enrichment_nodes", "unstable", "trial", "family_seed",
)


@dataclass(frozen=True)
class Row:
    method: str
    values: dict[str, object]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-results",
                   help="CSV or directory containing WLS and frozen CCPFM case rows. Auto-discovered when omitted.")
    p.add_argument("--namls-results",
                   help="CSV or directory containing NAMLS rows. Omit only if base-results already contains NAMLS.")
    p.add_argument("--livegate-results",
                   help="CSV or directory containing live-boundary CCPFM case rows. Auto-discovered when omitted.")
    p.add_argument("--sparsecore-results",
                   help="CSV or directory containing forced sparsecore CCPFM case rows. Auto-discovered when omitted.")
    p.add_argument("--selective-results",
                   help="CSV or directory containing selective sparsecore CCPFM case rows. Auto-discovered when omitted.")
    p.add_argument("--search-root", default=".",
                   help="Root searched recursively for summary_cases.csv/case_results.csv when inputs are omitted.")
    p.add_argument("--list-candidates", action="store_true",
                   help="Print discovered case CSV candidates and exit.")
    p.add_argument("--output-dir", default=".", help="Directory for the 17 PDF figures.")
    p.add_argument("--dpi", type=int, default=220, help="Raster preview DPI embedded by some backends.")
    p.add_argument("--allow-missing", action="store_true",
                   help="Draw available data and annotate missing panels instead of failing. Not recommended for submission.")
    return p.parse_args(argv)


def resolve_csv(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        candidates = [
            path / "summary_cases.csv",
            path / "case_results.csv",
            path / "cases.csv",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"No case CSV found in {path}. Expected one of: "
            + ", ".join(c.name for c in candidates)
        )
    if not path.is_file():
        raise FileNotFoundError(path)
    return path



def csv_header(path: Path) -> set[str]:
    try:
        with path.open(newline="", encoding="utf-8-sig") as f:
            return set(next(csv.reader(f), []))
    except (OSError, StopIteration):
        return set()


def sample_operators(path: Path, limit: int = 100) -> set[str]:
    ops: set[str] = set()
    try:
        with path.open(newline="", encoding="utf-8-sig") as f:
            for i, row in enumerate(csv.DictReader(f)):
                ops.add(raw_operator(row))
                if i + 1 >= limit:
                    break
    except OSError:
        pass
    return ops


def discover_candidates(root_value: str) -> list[Path]:
    root = Path(root_value).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Search root does not exist: {root}")
    names = {"summary_cases.csv", "case_results.csv", "cases.csv"}
    candidates = sorted(p for p in root.rglob("*.csv") if p.name in names and p.is_file())
    return candidates


def role_score(path: Path, role: str) -> int:
    header = csv_header(path)
    text = str(path).lower()
    ops = sample_operators(path)
    score = 0
    if role == "livegate":
        score += 100 if "live_boundary_band_active" in header else 0
        score += 35 if any(k in text for k in ("live_boundary", "livegate", "gate2")) else 0
    elif role == "sparsecore":
        score += 75 if "n_force_flagged_enrichment_nodes" in header else 0
        score += 45 if any(k in text for k in ("forced", "force", "sparsecore")) else 0
        score -= 60 if "live_boundary_band_active" in header else 0
    elif role == "selective":
        score += 50 if "n_flagged_enrichment_nodes" in header else 0
        score += 45 if "selective" in text and "forced" not in text else 0
        score += 20 if "n_force_flagged_enrichment_nodes" not in header else -20
        score -= 60 if "live_boundary_band_active" in header else 0
    elif role == "base":
        score += 30 if any(op in {"wls", "namls"} or "namls" in op for op in ops) else 0
        score += 25 if any(k in text for k in ("core_reg_complete", "core_reg", "baseline", "frozen")) else 0
        score -= 80 if "live_boundary_band_active" in header else 0
        score -= 35 if "selective" in text or "forced" in text else 0
    elif role == "namls":
        score += 100 if any("namls" in op for op in ops) else 0
    return score


def auto_select(candidates: list[Path], role: str) -> Path | None:
    ranked = sorted(((role_score(p, role), p) for p in candidates), key=lambda x: (-x[0], str(x[1])))
    positive = [(score, p) for score, p in ranked if score > 0]
    if not positive:
        return None
    if len(positive) > 1 and positive[0][0] == positive[1][0]:
        choices = "\n".join(f"  score={score:3d} {p}" for score, p in positive[:10])
        raise RuntimeError(f"Ambiguous auto-discovery for role {role!r}. Pass --{role.replace('_','-')}-results explicitly.\n{choices}")
    return positive[0][1]


def print_candidates(candidates: list[Path]) -> None:
    if not candidates:
        print("No summary_cases.csv, case_results.csv, or cases.csv files were found.")
        return
    print("Discovered case CSV candidates:")
    for p in candidates:
        scores = ", ".join(f"{r}={role_score(p,r)}" for r in ("base","namls","livegate","sparsecore","selective"))
        print(f"  {p}  [{scores}]")

def as_float(value: object) -> float:
    if value is None:
        return float("nan")
    text = str(value).strip()
    if text == "":
        return float("nan")
    if text.lower() in {"true", "yes"}:
        return 1.0
    if text.lower() in {"false", "no"}:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return float("nan")


def normalize_domain(value: object) -> str:
    text = str(value).strip().lower()
    aliases = {"rotated_ellipse": "ellipse", "five_point_star": "star"}
    return aliases.get(text, text)


def raw_operator(row: dict[str, str]) -> str:
    for key in ("operator", "method", "variant", "scheme"):
        value = row.get(key)
        if value:
            return value.strip().lower()
    return ""


def classify_method(row: dict[str, str], source_role: str) -> str | None:
    op = raw_operator(row)
    if op in {"wls", "wls_strong", "weighted_least_squares"}:
        return "WLS"
    if "namls" in op:
        return "NAMLS"
    # Source role disambiguates branches whose CSV operator is simply "ccpfm".
    if "ccpfm" in op or op in {"experimental_conservative", ""}:
        return {
            "base": "CCPFM-frozen",
            "namls": "NAMLS",
            "livegate": "CCPFM-livegate",
            "sparsecore": "CCPFM-sparsecore",
            "selective": "CCPFM-selective",
        }.get(source_role)
    return None


def read_rows(path: Path, source_role: str) -> list[Row]:
    out: list[Row] = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        for line_no, raw in enumerate(reader, start=2):
            method = classify_method(raw, source_role)
            if method is None:
                continue
            domain = normalize_domain(raw.get("domain", ""))
            if domain not in DOMAINS:
                continue
            values: dict[str, object] = dict(raw)
            values["domain"] = domain
            for key in REQUIRED_NUMERIC + OPTIONAL_NUMERIC:
                values[key] = as_float(raw.get(key))
            missing = [key for key in REQUIRED_NUMERIC if not math.isfinite(float(values[key]))]
            if missing:
                raise ValueError(f"{path}:{line_no}: missing/non-numeric {missing}")
            out.append(Row(method=method, values=values))
    if not out:
        raise ValueError(f"No recognized benchmark rows found in {path} for role {source_role!r}")
    return out


def load_all(args: argparse.Namespace) -> list[Row]:
    candidates = discover_candidates(args.search_root)
    if args.list_candidates:
        print_candidates(candidates)
        raise SystemExit(0)

    supplied = {
        "base": args.base_results,
        "namls": args.namls_results,
        "livegate": args.livegate_results,
        "sparsecore": args.sparsecore_results,
        "selective": args.selective_results,
    }
    selected: dict[str, Path | None] = {}
    for role, value in supplied.items():
        if value:
            selected[role] = resolve_csv(value)
        elif role == "namls":
            # NAMLS is normally present in the base/complete benchmark CSV.
            # Do not auto-select a second file because several branches may repeat NAMLS rows.
            selected[role] = None
        else:
            selected[role] = auto_select(candidates, role)

    # NAMLS commonly lives in the base CSV. Avoid forcing a separate source.
    required = ("base", "livegate", "sparsecore", "selective")
    missing = [role for role in required if selected[role] is None]
    if missing:
        print_candidates(candidates)
        flags = " ".join(f"--{r.replace('_','-')}-results PATH" for r in missing)
        raise FileNotFoundError(
            f"Could not auto-discover required result roles: {', '.join(missing)}. "
            f"Pass explicit paths: {flags}"
        )

    print("Selected benchmark inputs:")
    for role in ("base", "namls", "livegate", "sparsecore", "selective"):
        if selected[role] is not None:
            print(f"  {role:10s}: {selected[role]}")

    rows: list[Row] = []
    seen_files: set[tuple[Path, str]] = set()
    for role, path in selected.items():
        if path is None:
            continue
        key = (path, role)
        if key in seen_files:
            continue
        seen_files.add(key)
        rows.extend(read_rows(path, role))
    return rows


def finite_values(rows: Iterable[Row], method: str, domain: str, key: str, h: float | None = None) -> np.ndarray:
    values: list[float] = []
    for row in rows:
        v = row.values
        if row.method != method or v["domain"] != domain:
            continue
        if h is not None and not np.isclose(float(v["h_target"]), h, rtol=0.0, atol=1e-12):
            continue
        x = as_float(v.get(key))
        if math.isfinite(x):
            values.append(x)
    return np.asarray(values, dtype=float)


def med(rows: Iterable[Row], method: str, domain: str, key: str, h: float) -> float:
    vals = finite_values(rows, method, domain, key, h)
    return float(np.median(vals)) if vals.size else float("nan")


def unique_h(rows: Iterable[Row], method: str, domain: str) -> np.ndarray:
    vals = finite_values(rows, method, domain, "h_target")
    return np.asarray(sorted(set(float(x) for x in vals)), dtype=float)


def available_methods(rows: Iterable[Row]) -> set[str]:
    return {r.method for r in rows}


def validate(rows: list[Row], allow_missing: bool) -> None:
    errors: list[str] = []
    present = available_methods(rows)
    for method in METHOD_ORDER:
        if method not in present:
            errors.append(f"missing method family: {method}")
    for method in METHOD_ORDER:
        for domain in DOMAINS:
            n = len(finite_values(rows, method, domain, "l2"))
            if n == 0:
                errors.append(f"no L2 rows for {method} / {domain}")
    finest_by_domain = {}
    for domain in DOMAINS:
        hs = unique_h(rows, "CCPFM-frozen", domain)
        if hs.size:
            finest_by_domain[domain] = float(np.min(hs))
        else:
            errors.append(f"cannot determine finest h for frozen CCPFM / {domain}")
    for domain, h in finest_by_domain.items():
        for method in METHOD_ORDER:
            for key in ("linf", "l2_near_boundary_interior", "l2_core_interior"):
                if not math.isfinite(med(rows, method, domain, key, h)):
                    errors.append(f"missing {key} at finest h={h:g} for {method} / {domain}")
    if errors and not allow_missing:
        joined = "\n  - ".join(errors)
        raise ValueError(
            "Required manuscript data are incomplete; no figures were written.\n  - " + joined
            + "\nUse --allow-missing only for diagnostics, not for the manuscript."
        )
    if errors:
        print("WARNING: incomplete data:\n  - " + "\n  - ".join(errors), file=sys.stderr)


def save(fig: plt.Figure, path: Path, dpi: int) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    if not path.is_file() or path.stat().st_size < 1000:
        raise RuntimeError(f"Figure was not written correctly: {path}")


def plot_geometries(out: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(11.0, 2.8))
    t = np.linspace(0.0, 2.0 * np.pi, 1000)
    # Circle
    axes[0].plot(np.cos(t), np.sin(t))
    axes[0].fill(np.cos(t), np.sin(t), alpha=0.12)
    axes[0].set_title("Circle")
    # Rotated ellipse
    a, b, angle = 1.2, 0.7, np.deg2rad(28.0)
    x0, y0 = a * np.cos(t), b * np.sin(t)
    x = np.cos(angle) * x0 - np.sin(angle) * y0
    y = np.sin(angle) * x0 + np.cos(angle) * y0
    axes[1].plot(x, y)
    axes[1].fill(x, y, alpha=0.12)
    axes[1].set_title("Rotated ellipse")
    # Five-point star domain used by the benchmark.
    r = 0.78 + 0.22 * np.cos(5.0 * t)
    x, y = r * np.cos(t), r * np.sin(t)
    axes[2].plot(x, y)
    axes[2].fill(x, y, alpha=0.12)
    axes[2].set_title("Five-point star")
    # Annulus
    xo, yo = np.cos(t), np.sin(t)
    xi, yi = 0.35 * np.cos(t), 0.35 * np.sin(t)
    axes[3].plot(xo, yo)
    axes[3].plot(xi, yi)
    axes[3].fill(xo, yo, alpha=0.12)
    axes[3].fill(xi, yi, color="white")
    axes[3].set_title("Annulus")
    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-1.35, 1.35)
        ax.set_ylim(-1.2, 1.2)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    save(fig, out / "benchmark_geometries.pdf", dpi)


def plot_convergence(rows: list[Row], domain: str, metric: str, out: Path, dpi: int, allow_missing: bool) -> None:
    fig, ax = plt.subplots(figsize=(5.3, 4.0))
    plotted = 0
    for method in METHOD_ORDER:
        hs = unique_h(rows, method, domain)
        xs, ys = [], []
        for h in hs:
            y = med(rows, method, domain, metric, float(h))
            if math.isfinite(y) and y > 0:
                xs.append(float(h)); ys.append(y)
        if xs:
            order = np.argsort(xs)[::-1]
            ax.loglog(np.asarray(xs)[order], np.asarray(ys)[order], marker="o", label=METHOD_SHORT[method])
            plotted += 1
    if plotted == 0:
        if not allow_missing:
            raise ValueError(f"No convergence data for {domain} / {metric}")
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
    ax.set_xlabel("Target spacing $h$")
    ax.set_ylabel("Median $L_2$ error" if metric == "l2" else "Median $L_\\infty$ error")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    save(fig, out / f"convergence_{metric}_{domain}.pdf", dpi)


def finest_h(rows: list[Row], domain: str) -> float:
    hs = unique_h(rows, "CCPFM-frozen", domain)
    if not hs.size:
        return float("nan")
    return float(np.min(hs))


def plot_boundary_core(rows: list[Row], domain: str, out: Path, dpi: int) -> None:
    h = finest_h(rows, domain)
    near = [med(rows, m, domain, "l2_near_boundary_interior", h) for m in METHOD_ORDER]
    core = [med(rows, m, domain, "l2_core_interior", h) for m in METHOD_ORDER]
    x = np.arange(len(METHOD_ORDER), dtype=float)
    width = 0.38
    fig, ax = plt.subplots(figsize=(5.7, 4.0))
    ax.bar(x - width / 2, near, width, label="Near-boundary interior")
    ax.bar(x + width / 2, core, width, label="Core interior")
    ax.set_xticks(x, [METHOD_SHORT[m] for m in METHOD_ORDER], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Median $L_2$ error")
    ax.set_title(f"Finest spacing $h={h:g}$")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    save(fig, out / f"finest_boundary_core_l2_{domain}.pdf", dpi)


def instability_count(rows: list[Row], method: str, domain: str, h: float) -> int:
    vals = finite_values(rows, method, domain, "unstable", h)
    return int(np.sum(vals > 0.5)) if vals.size else 0


def metric_winner(rows: list[Row], domain: str, h: float, key: str, robustness: bool = False) -> str:
    if robustness:
        scores = {m: instability_count(rows, m, domain, h) for m in METHOD_ORDER}
    else:
        scores = {m: med(rows, m, domain, key, h) for m in METHOD_ORDER}
        scores = {m: v for m, v in scores.items() if math.isfinite(v)}
    if not scores:
        return ""
    return min(scores, key=scores.get)


def overall_practical(rows: list[Row], domain: str, h: float, margin: float = 0.01) -> tuple[str, bool]:
    # Composite geometric mean of normalized L2, Linf, near, and core errors,
    # with a robustness penalty. A sub-1% challenger advantage does not displace
    # the frozen baseline (the manuscript's margin-safe interpretation).
    keys = ("l2", "linf", "l2_near_boundary_interior", "l2_core_interior")
    raw: dict[str, float] = {}
    minima = {k: min(v for v in (med(rows, m, domain, k, h) for m in METHOD_ORDER) if math.isfinite(v) and v > 0) for k in keys}
    for method in METHOD_ORDER:
        ratios = []
        for key in keys:
            value = med(rows, method, domain, key, h)
            if not math.isfinite(value) or value <= 0:
                ratios = []
                break
            ratios.append(value / minima[key])
        if ratios:
            penalty = 1.0 + 0.05 * instability_count(rows, method, domain, h)
            raw[method] = float(np.exp(np.mean(np.log(ratios))) * penalty)
    winner = min(raw, key=raw.get)
    frozen = raw.get("CCPFM-frozen", float("inf"))
    margin_held = winner != "CCPFM-frozen" and frozen <= raw[winner] * (1.0 + margin)
    return ("CCPFM-frozen" if margin_held else winner), margin_held


def plot_winner_summary(rows: list[Row], out: Path, dpi: int) -> None:
    categories = ("Robustness", "$L_2$", "$L_\\infty$", "Near boundary", "Core", "Overall")
    methods = list(METHOD_ORDER)
    matrix = np.zeros((len(DOMAINS), len(categories)), dtype=int)
    text = np.empty(matrix.shape, dtype=object)
    for i, domain in enumerate(DOMAINS):
        h = finest_h(rows, domain)
        winners = [
            metric_winner(rows, domain, h, "l2", robustness=True),
            metric_winner(rows, domain, h, "l2"),
            metric_winner(rows, domain, h, "linf"),
            metric_winner(rows, domain, h, "l2_near_boundary_interior"),
            metric_winner(rows, domain, h, "l2_core_interior"),
            overall_practical(rows, domain, h)[0],
        ]
        for j, winner in enumerate(winners):
            matrix[i, j] = methods.index(winner) if winner in methods else 0
            text[i, j] = METHOD_SHORT.get(winner, "missing")
    fig, ax = plt.subplots(figsize=(10.2, 3.4))
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", vmin=-0.5, vmax=len(methods)-0.5)
    ax.set_xticks(np.arange(len(categories)), categories)
    ax.set_yticks(np.arange(len(DOMAINS)), [d.title() for d in DOMAINS])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, text[i, j], ha="center", va="center", fontsize=7,
                    color="white" if matrix[i, j] >= len(methods) / 2 else "black")
    cbar = fig.colorbar(im, ax=ax, ticks=np.arange(len(methods)), fraction=0.035, pad=0.02)
    cbar.ax.set_yticklabels([METHOD_SHORT[m] for m in methods], fontsize=7)
    ax.set_title("Finest-spacing winner summary")
    save(fig, out / "winner_summary_finest.pdf", dpi)


def percent_delta(variant: float, baseline: float) -> float:
    if not (math.isfinite(variant) and math.isfinite(baseline)) or baseline == 0:
        return float("nan")
    return 100.0 * (variant - baseline) / baseline


def paired_rows(rows: list[Row], method: str, domain: str, h: float) -> dict[tuple[object, ...], Row]:
    return {
        row_pair_key(r): r
        for r in rows
        if r.method == method and r.values["domain"] == domain
        and np.isclose(float(r.values["h_target"]), h, atol=1e-12, rtol=0)
    }


def paired_percent_deltas(rows: list[Row], domain: str, h: float, key: str) -> np.ndarray:
    base = paired_rows(rows, "CCPFM-frozen", domain, h)
    live = paired_rows(rows, "CCPFM-livegate", domain, h)
    common = sorted(set(base).intersection(live))
    deltas: list[float] = []
    for k in common:
        b = as_float(base[k].values.get(key))
        v = as_float(live[k].values.get(key))
        d = percent_delta(v, b)
        if math.isfinite(d):
            deltas.append(d)
    return np.asarray(deltas, dtype=float)


def paired_instability_deltas(rows: list[Row], domain: str, h: float) -> np.ndarray:
    base = paired_rows(rows, "CCPFM-frozen", domain, h)
    live = paired_rows(rows, "CCPFM-livegate", domain, h)
    common = sorted(set(base).intersection(live))
    deltas: list[float] = []
    for k in common:
        b = 100.0 * as_float(base[k].values.get("unstable"))
        v = 100.0 * as_float(live[k].values.get("unstable"))
        if math.isfinite(b) and math.isfinite(v):
            deltas.append(v - b)
    return np.asarray(deltas, dtype=float)


def median_iqr(values: np.ndarray) -> tuple[float, float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan")
    q1, q2, q3 = np.percentile(vals, [25, 50, 75])
    return float(q2), float(q1), float(q3)


def _annotate_bar(ax: plt.Axes, x: float, y: float, ylim: float) -> None:
    if not math.isfinite(y):
        ax.text(x, 0.02 * ylim, "n/a", ha="center", va="bottom", fontsize=7, rotation=90)
        return
    offset = max(0.8, 0.025 * ylim)
    va = "bottom" if y >= 0 else "top"
    yy = y + offset if y >= 0 else y - offset
    ax.text(x, yy, f"{y:+.1f}%", ha="center", va=va, fontsize=7)


def _apply_symmetric_ylim(ax: plt.Axes, series: list[tuple[float, float, float]]) -> float:
    max_abs = 1.0
    for medv, q1, q3 in series:
        if math.isfinite(medv):
            max_abs = max(max_abs, abs(medv))
        if math.isfinite(q1):
            max_abs = max(max_abs, abs(q1))
        if math.isfinite(q3):
            max_abs = max(max_abs, abs(q3))
    ylim = 1.15 * max_abs
    if ylim <= 5.0:
        ylim = 5.0
    elif ylim <= 10.0:
        ylim = 10.0
    else:
        ylim = math.ceil(ylim / 5.0) * 5.0
    ax.set_ylim(-ylim, ylim)
    return ylim


def _panel_grouped_bars(ax: plt.Axes, rows: list[Row], metric_specs: list[tuple[str, str]], instability_panel: bool = False) -> None:
    x = np.arange(len(DOMAINS), dtype=float)
    n = len(metric_specs)
    width = 0.28 if n <= 2 else 0.18
    offsets = (np.arange(n, dtype=float) - 0.5 * (n - 1)) * width
    all_series: list[tuple[float, float, float]] = []
    bar_records: list[tuple[float, float]] = []
    for j, (key, label) in enumerate(metric_specs):
        meds, lows, highs = [], [], []
        for domain in DOMAINS:
            h = finest_h(rows, domain)
            vals = paired_instability_deltas(rows, domain, h) if key == "unstable_pp" else paired_percent_deltas(rows, domain, h, key)
            medv, q1, q3 = median_iqr(vals)
            meds.append(medv)
            lows.append(medv - q1 if math.isfinite(medv) and math.isfinite(q1) else float("nan"))
            highs.append(q3 - medv if math.isfinite(medv) and math.isfinite(q3) else float("nan"))
            all_series.append((medv, q1, q3))
        xpos = x + offsets[j]
        yerr = np.vstack([np.asarray(lows, dtype=float), np.asarray(highs, dtype=float)])
        ax.bar(xpos, meds, width=width*0.92, label=label, yerr=yerr, capsize=3, linewidth=0.6)
        for xx, yy in zip(xpos, meds):
            bar_records.append((xx, yy))
    ylim = _apply_symmetric_ylim(ax, all_series)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_xticks(x, [d.title() for d in DOMAINS])
    if instability_panel:
        ax.set_ylabel("Median change vs. frozen baseline (%)\nIQR error bars; instability = percentage-point change")
    else:
        ax.set_ylabel("Median change vs. frozen baseline (%)\nIQR error bars; negative = improvement")
    for xx, yy in bar_records:
        _annotate_bar(ax, xx, yy, ylim)
    ax.legend(fontsize=8, ncol=len(metric_specs), loc="upper center")


def plot_livegate_gain(rows: list[Row], out: Path, dpi: int) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 7.2), sharex=True)
    accuracy_specs = [("l2", "$L_2$"), ("linf", "$L_\\infty$")]
    cost_specs = [("total_seconds", "Runtime"), ("unstable_pp", "Instability")]
    _panel_grouped_bars(axes[0], rows, accuracy_specs, instability_panel=False)
    axes[0].set_title("Livegate branch behavior at the finest spacing: accuracy")
    _panel_grouped_bars(axes[1], rows, cost_specs, instability_panel=True)
    axes[1].set_title("Livegate branch behavior at the finest spacing: cost and robustness")
    fig.suptitle(
        "Relative changes of the live-boundary branch with respect to frozen CCPFM\nMedians over matched trials with interquartile-range error bars",
        y=0.98, fontsize=11,
    )
    save(fig, out / "livegate_gain_finest.pdf", dpi)


def activation_count(rows: list[Row], method: str, domain: str, h: float) -> tuple[int, int, float, float]:
    selected = [r for r in rows if r.method == method and r.values["domain"] == domain
                and np.isclose(float(r.values["h_target"]), h, atol=1e-12, rtol=0)]
    added = np.asarray([as_float(r.values.get("n_added_edges")) for r in selected], dtype=float)
    flagged = np.asarray([as_float(r.values.get("n_flagged_enrichment_nodes")) for r in selected], dtype=float)
    valid_added = added[np.isfinite(added)]
    valid_flagged = flagged[np.isfinite(flagged)]
    activated = int(np.sum(valid_added > 0.0)) if valid_added.size else 0
    return activated, len(selected), float(np.mean(valid_flagged)) if valid_flagged.size else float("nan"), float(np.mean(valid_added)) if valid_added.size else float("nan")


def plot_sparse_counts(rows: list[Row], out: Path, dpi: int) -> None:
    x = np.arange(len(DOMAINS), dtype=float)
    width = 0.36
    forced = []
    selective = []
    for domain in DOMAINS:
        h = finest_h(rows, domain)
        forced.append(activation_count(rows, "CCPFM-sparsecore", domain, h)[3])
        selective.append(activation_count(rows, "CCPFM-selective", domain, h)[3])
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.bar(x - width/2, forced, width, label="Forced sparsecore")
    ax.bar(x + width/2, selective, width, label="Selective sparsecore")
    ax.set_xticks(x, [d.title() for d in DOMAINS])
    ax.set_ylabel("Mean added edges per run")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    save(fig, out / "sparse_activation_counts_finest.pdf", dpi)


def row_pair_key(row: Row) -> tuple[object, ...]:
    v = row.values
    # Trial and family seed are the preferred exact pairing fields.
    trial = as_float(v.get("trial")); seed = as_float(v.get("family_seed"))
    if math.isfinite(seed):
        return (v["domain"], float(v["h_target"]), int(seed))
    if math.isfinite(trial):
        return (v["domain"], float(v["h_target"]), int(trial))
    return (v["domain"], float(v["h_target"]), str(v.get("case_id", "")))


def activated_only_delta(rows: list[Row], domain: str, h: float, key: str) -> float:
    base = {row_pair_key(r): r for r in rows if r.method == "CCPFM-frozen" and r.values["domain"] == domain
            and np.isclose(float(r.values["h_target"]), h, atol=1e-12, rtol=0)}
    deltas = []
    for r in rows:
        if r.method != "CCPFM-selective" or r.values["domain"] != domain:
            continue
        if not np.isclose(float(r.values["h_target"]), h, atol=1e-12, rtol=0):
            continue
        if as_float(r.values.get("n_added_edges")) <= 0:
            continue
        b = base.get(row_pair_key(r))
        if b is None:
            continue
        deltas.append(percent_delta(as_float(r.values.get(key)), as_float(b.values.get(key))))
    vals = np.asarray([x for x in deltas if math.isfinite(x)], dtype=float)
    return float(np.median(vals)) if vals.size else float("nan")


def plot_selective_activation(rows: list[Row], out: Path, dpi: int) -> None:
    freq = []
    annotations = []
    for domain in DOMAINS:
        h = finest_h(rows, domain)
        activated, total, _, _ = activation_count(rows, "CCPFM-selective", domain, h)
        frequency = 100.0 * activated / total if total else 0.0
        freq.append(frequency)
        if activated == 0:
            annotations.append("inactive\nact-only: inactive")
        else:
            d_l2 = activated_only_delta(rows, domain, h, "l2")
            d_linf = activated_only_delta(rows, domain, h, "linf")
            scale = max(abs(d_l2) if math.isfinite(d_l2) else 0.0, abs(d_linf) if math.isfinite(d_linf) else 0.0)
            verdict = "conditional negligible" if scale < 1.0 else ("conditional gain" if d_l2 < 0 and d_linf <= 0 else "conditional mixed")
            annotations.append(f"{activated}/{total} active\nact-only: {verdict}")
    x = np.arange(len(DOMAINS), dtype=float)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    bars = ax.bar(x, freq)
    ax.set_xticks(x, [d.title() for d in DOMAINS])
    ax.set_ylim(0, max(100.0, max(freq, default=0.0) * 1.35 + 5.0))
    ax.set_ylabel("Activation frequency (%)")
    ax.grid(True, axis="y", alpha=0.25)
    for bar, text in zip(bars, annotations):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2.0, text,
                ha="center", va="bottom", fontsize=7)
    save(fig, out / "sparse_selective_activation_finest.pdf", dpi)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = load_all(args)
    validate(rows, args.allow_missing)

    plot_geometries(out, args.dpi)
    for domain in DOMAINS:
        plot_convergence(rows, domain, "l2", out, args.dpi, args.allow_missing)
        plot_convergence(rows, domain, "linf", out, args.dpi, args.allow_missing)
        plot_boundary_core(rows, domain, out, args.dpi)
    plot_winner_summary(rows, out, args.dpi)
    plot_livegate_gain(rows, out, args.dpi)
    plot_sparse_counts(rows, out, args.dpi)
    plot_selective_activation(rows, out, args.dpi)

    expected = [
        "benchmark_geometries.pdf",
        *(f"convergence_l2_{d}.pdf" for d in DOMAINS),
        *(f"convergence_linf_{d}.pdf" for d in DOMAINS),
        *(f"finest_boundary_core_l2_{d}.pdf" for d in DOMAINS),
        "winner_summary_finest.pdf", "livegate_gain_finest.pdf",
        "sparse_activation_counts_finest.pdf", "sparse_selective_activation_finest.pdf",
    ]
    missing = [name for name in expected if not (out / name).is_file()]
    if missing:
        raise RuntimeError(f"Internal error: figures not created: {missing}")
    print(f"Created {len(expected)} nonempty PDF figures in {out}")
    for name in expected:
        print(f"  {name}: {(out/name).stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
