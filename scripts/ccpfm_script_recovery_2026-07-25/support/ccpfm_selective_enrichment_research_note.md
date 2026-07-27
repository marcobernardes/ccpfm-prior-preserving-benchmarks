# CCPFm selective-enrichment branch: compact research note

## Executive diagnosis
The current run is a **successful diagnosis run**, but **not yet a true enrichment run**.

The decisive indicators are:
- `added frac = 0.000` on every geometry
- `added corr p90 = nan` on every geometry

That means the winning CCPFm graphs contained **no added rescue edges**. So the reported results are effectively from the boundary-split baseline with stronger diagnostics, not from an activated selective-enrichment mechanism.

## What the run establishes
1. **Patch consistency is no longer the bottleneck.**  
   CCPFm patch residuals are uniformly tight (roughly `1e-9` to `1e-10` on linear/quadratic patch tests), so weak convergence is no longer attributable to failed polynomial consistency.

2. **CCPFm currently wins on error constant, not on asymptotic rate.**  
   At the finest reported `h`, CCPFm beats WLS in median `L2` on all four domains, but its global median `L2` order is only about `1.30–1.81`, versus roughly `2.38–4.76` for WLS.

3. **The interior/core remains the dominant bottleneck.**  
   The boundary/core split confirms that the weaker convergence is primarily in the core interior, not in the boundary layer.

4. **The constrained correction is still doing heavy work on the original graph.**  
   `orig corr p90 ≈ 1.75–1.85` and `mean neg(lambda) ≈ 0.32–0.36` indicate that the consistency correction still strongly reshapes the original graph fluxes.

## Interpretation
The present branch says:

> Under the current selective-enrichment triggers and graph-selection logic, the best-scoring CCPFm graph is still the one with no added rescue edges.

So the right conclusion is **not** “selective enrichment helped” or “selective enrichment failed.” The correct conclusion is:

> The current enrichment rules are too conservative to engage on the winning graphs, and the remaining error mechanism is still tied to the corrected original interior graph.

## Patch objective
The patch attached with this note does two things:

1. **Force the enrichment path to activate** so the experiment actually tests rescue edges.
2. **Print direct enrichment diagnostics** so activation can no longer be hidden behind summary aggregates.

## What the patch changes
### 1) Deterministic fallback flagging
If the absolute triggers
- `cond > selective_cond_trigger`, or
- `sigma_min < selective_sigma_trigger`

flag too few deep-core nodes, the patch force-flags the worst-scoring core nodes based on the local score

`cond / max(sigma_min, 1e-16)`.

Default behavior in the patched script:
- force at least **2** core nodes per case

### 2) Relaxed partner search for flagged nodes
If a flagged node cannot find a valid rescue partner under the strict core-core distance cap, the patch progressively relaxes the cap and finally falls back to the nearest admissible interior node away from the immediate boundary layer.

This preserves the original philosophy (sparse, interior-focused enrichment) while ensuring that a force-flagged node actually receives a rescue edge.

### 3) Direct activation diagnostics
The patched script now reports, both per case and in aggregates:
- number of flagged enrichment nodes
- number of force-flagged nodes
- number of flagged nodes that actually received added edges
- number of flagged nodes that did not receive added edges
- total number of added edges
- mean added edges per flagged node
- selected graph-attempt index

## Smoke-test result on the patched script
A smoke test on the patched file showed active enrichment, e.g. for a circle case:
- `flags=2`
- `forced=2`
- `hit=2`
- `added=2`

So the patched branch now actually exercises the selective-enrichment mechanism instead of silently collapsing back to the no-added-edge baseline.

## Recommended next experiment
Run the patched branch first with the current penalties, then sweep the aggressiveness.

Suggested baseline command:

```bash
python ccpfm_poisson_validation_selective_enrichment_forced.py \
  --outdir /mnt/data/ccpfm_poisson_results_selective_forced \
  --force-min-flagged-core-nodes 2 \
  --force-relaxed-distance-factor 5.0
```

Then try:
- `--force-min-flagged-core-nodes 4`
- `--added-edge-penalty 25`
- `--added-edge-penalty 50`
- `--selective-cond-trigger 1e4`
- `--selective-sigma-trigger 1e-3`

## Scientific value of the patch
This patch does **not** claim that rescue-edge enrichment is beneficial. It does something more important first:

> it converts the branch into a valid experiment.

Without activation, the branch only re-measures the no-added-edge operator. With activation and direct diagnostics, you can finally answer the real research question:

> When a very sparse, heavily penalized rescue-edge mechanism is forced to engage, does it improve the CCPFm interior-core error trend, or does it merely shift correction burden without improving convergence?
