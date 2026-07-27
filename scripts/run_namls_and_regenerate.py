#!/usr/bin/env python3
"""Run reconstructed NAMLS and regenerate all CAMWA figures."""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", default=".")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--trials", type=int, default=10)
    args = p.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    py = sys.executable
    namls = [
        py, str(root / "ccpfm_poisson_validation_namls_reconstructed.py"),
        "--base-script", str(root / "ccpfm_poisson_validation_prior_preserving_core_reg_complete.py"),
        "--domains", "circle", "ellipse", "star", "annulus",
        "--hs", "0.20", "0.16", "0.12", "0.10", "0.08",
        "--trials", str(args.trials), "--workers", str(args.workers),
        "--outdir", str(root / "original_results" / "namls"),
    ]
    subprocess.run(namls, check=True, cwd=root)
    figures = [
        py, str(root / "regenerate_camwa_figures_v3.py"),
        "--base-results", str(root / "original_results/frozen/summary_cases.csv"),
        "--namls-results", str(root / "original_results/namls/summary_cases.csv"),
        "--livegate-results", str(root / "original_results/livegate/summary_cases.csv"),
        "--sparsecore-results", str(root / "original_results/sparsecore_forced/summary_cases.csv"),
        "--selective-results", str(root / "original_results/sparsecore_selective/summary_cases.csv"),
        "--output-dir", str(root / "manuscript_figures"),
    ]
    subprocess.run(figures, check=True, cwd=root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
