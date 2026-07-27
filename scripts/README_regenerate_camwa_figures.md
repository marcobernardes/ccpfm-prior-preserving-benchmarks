# Regenerating the CAMWA manuscript figures

The script `regenerate_camwa_figures.py` reads **case-level** benchmark CSVs and creates all 17 PDFs referenced by `CAMWA-D-26-00757_Original.tex`.

## Required datasets

Provide the result directory (or `summary_cases.csv` itself) for:

1. the base run containing WLS and frozen CCPFM;
2. the NAMLS run;
3. the live-boundary gated run;
4. the force-driven sparsecore run;
5. the naturally selective sparsecore run.

A directory input is searched for `summary_cases.csv`, then `case_results.csv`, then `cases.csv`.

## Example

```bash
python regenerate_camwa_figures.py \
  --base-results results/core_reg_complete \
  --namls-results results/namls \
  --livegate-results results/live_boundary_gate2 \
  --sparsecore-results results/selective_enrichment_forced \
  --selective-results results/selective_enrichment \
  --output-dir manuscript_figures
```

The exact directory names may differ on your machine. Point each argument to the directory containing the corresponding `summary_cases.csv`.

## Strict validation

By default, the script stops before writing manuscript plots when any of the following is absent:

- one of the six method families;
- one of the four geometries;
- convergence data;
- finest-grid `L2`, `Linf`, near-boundary, or core error data.

This is intentional. It prevents a missing CSV, incorrect path, or incompatible column name from producing blank submission figures.

For diagnostic work only, `--allow-missing` permits partial plots and prints warnings. Do not use partial plots in the manuscript.

## Generated files

- `benchmark_geometries.pdf`
- `convergence_l2_circle.pdf`
- `convergence_l2_ellipse.pdf`
- `convergence_l2_star.pdf`
- `convergence_l2_annulus.pdf`
- `convergence_linf_circle.pdf`
- `convergence_linf_ellipse.pdf`
- `convergence_linf_star.pdf`
- `convergence_linf_annulus.pdf`
- `finest_boundary_core_l2_circle.pdf`
- `finest_boundary_core_l2_ellipse.pdf`
- `finest_boundary_core_l2_star.pdf`
- `finest_boundary_core_l2_annulus.pdf`
- `winner_summary_finest.pdf`
- `livegate_gain_finest.pdf`
- `sparse_activation_counts_finest.pdf`
- `sparse_selective_activation_finest.pdf`

## Notes

- Errors shown in convergence plots are medians over trials at each `h`.
- The finest spacing is detected separately for each geometry from frozen CCPFM data.
- Percent changes use `(variant - frozen) / frozen * 100`; negative values denote improvement.
- Selective activated-run statistics are paired using `family_seed`, then `trial` when the seed is unavailable.
- The winner summary is recomputed from measured errors and instability indicators. Its overall column uses a 1% margin-safe baseline-holding rule.
