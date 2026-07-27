# Extended CCPFM diffusion benchmarks

This package generates the new evidence requested in the CAMWA review:

- smooth variable-coefficient diffusion on the four original geometries;
- a discontinuous circular interface problem on the circle;
- WLS strong-form and coefficient-aware CCPFM flux comparisons;
- case and aggregate CSV/JSON files;
- convergence, conditioning, and runtime figures;
- LaTeX tables generated only from measured runs.

## Run a quick validation

```bash
python run_extended_diffusion_benchmarks.py --h 0.30 --trials 1 --domains circle --workers 1
python make_extended_manuscript_assets.py
```

## Run the manuscript study

```bash
python run_all_extended.py --h 0.20 0.16 0.125 0.10 --trials 5 --workers 4
```

The interface implementation uses harmonic edge coefficients and a protected band around the discontinuity. Polynomial correction is applied only away from that band. This is intentional: smooth strong-form polynomial identities are not valid across a coefficient jump.

Copy or `\input` the generated files:

- `extended_results/manuscript_assets/extended_accuracy_table.tex`
- `extended_results/manuscript_assets/extended_cost_table.tex`
- generated PDF figures in the same directory.

## Python multiprocessing and restart behavior

The runner sends only strings and numeric values to worker processes and constructs
problem callables inside each worker. This is required by Python 3.12 multiprocessing,
which cannot pickle nested interface-problem functions.

Successful cases are checkpointed after every completed job. Resume mode is enabled
by default, so rerunning the same command skips cases already present in
`extended_results/case_results.csv`. Use `--no-resume` to intentionally start the
requested matrix again.
