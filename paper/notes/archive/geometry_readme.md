# R1 geometric certificate

Main report: `R1_GEOMETRY_AND_COVERAGE.md` (Chinese).

This package is self-contained for the geometric compression: 101 continuous
middle-scale cells, high-scale global convexity, low-scale dominant-template
balls, wrong-basin entry certificates, and an intentionally coarse slow-tracking
corollary. It does not contain or revalidate the prior 64-call trajectory proof.

Run the proof checks using Python 3, with assertions enabled:

```bash
python3 verify_geometry.py
```

No third-party packages are required. This regenerates:

- `geometry_certificate.json`: certified bounds, coverage, source SHA-256 hashes.
- `sparse_cell_bounds.json`: continuous interval bounds for each of the 101 cells.

Core files:

- `interval_core.py`: 42-digit outward-rounded Decimal interval arithmetic and model.
- `geometry.py`: whitened distances, competition trace, explicit component centers,
  dominant-template ball conditions, correlated initialization distances.
- `sparse_branch.py`: continuous-segment residual and contraction checks.
- `sparse_candidates.json`: untrusted decimal candidates that the verifier checks.

Optional diagnostics and illustrations require NumPy, SciPy, and Matplotlib:

```bash
python3 check_geometry.py
python3 plot_geometry.py
```

`identity_checks.json` contains supplementary floating-point identity checks;
these are not a substitute for the interval certificate. PNG/SVG curves are
nominal illustrations, not plots of verified interval enclosures.

`make_sparse_candidates.py` regenerates candidates with numerical root-finding.
Candidate generation is not part of the trusted proof. Different package versions
may produce different candidates or counts; the provided candidates are fixed.

The source and analytic argument have not received independent review or formal
verification. Exact coverage and limitations are listed in the report.
