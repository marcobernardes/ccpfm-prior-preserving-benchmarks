# CCPFM Python Script Recovery

This package contains every CCPFM Poisson Python script whose source bytes could be recovered from the active conversation workspace.

## Recovered runnable scripts

- `ccpfm_poisson_validation_prior_preserving_core_reg.py` — 2076 lines, 92912 bytes, syntax check passed
- `ccpfm_poisson_validation_prior_preserving_core_reg_complete.py` — 2174 lines, 99395 bytes, syntax check passed
- `ccpfm_poisson_validation_prior_preserving_core_reg_live_boundary.py` — 2329 lines, 107928 bytes, syntax check passed
- `ccpfm_poisson_validation_prior_preserving_core_reg_live_boundary_gate2.py` — 2347 lines, 108902 bytes, syntax check passed
- `ccpfm_poisson_validation_selective_enrichment.py` — 1809 lines, 75155 bytes, syntax check passed
- `ccpfm_poisson_validation_selective_enrichment_forced.py` — 1954 lines, 84984 bytes, syntax check passed

## Important provenance notes

- `ccpfm_poisson_validation_selective_enrichment.py` was recovered exactly from the preserved forced-version patch by applying that patch in reverse. Its recovered size is 75,155 bytes, matching the indexed original upload size.
- The `live_boundary_gate2.py` file is directly recovered and included.
- Some earlier/later filenames are documented in `manifest.json` but could not be recovered byte-for-byte because the old attachment backing is unavailable or the file existed only at a prior local path.
- No unavailable script has been silently replaced or renamed as though it were exact.

## Validation

All included `.py` files passed `python3 -m compileall`.
Use `SHA256SUMS.txt` to verify file integrity.
