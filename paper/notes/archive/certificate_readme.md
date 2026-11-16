# R1 three-template certificate

Read `R1_THREE_TEMPLATE_THEOREM.md` for the statement, proof, limitations, and references.

Run the complete certificate with Python 3.12 (standard library only):

```bash
python3 verify.py
```

Expected result: `PASS`. Do not use `python -O`; assertions are the checks.
Runtime in the creation environment: about 20 seconds. The verifier checks a
full parameter box and continuous noise-scale slabs with outward-rounded
Decimal intervals. It does not sample random observations.

The certificate covers a rational, rounded construction near the earlier R1
example. It does not certify the earlier 100,000-sample experiment or noise
standard deviation 0.05.

The 64-call theorem permits a per-coordinate model perturbation of 0.0001,
measurement noise coordinates in [-0.001, 0.001], and internal annealing-scale
perturbations of 0.0001. Endpoints are fixed at 2 and 0.35. The negative-curvature
witness shows why a separate finite-trajectory proof is needed for this budget.

Files:

- `interval.py`: outward-rounded interval arithmetic and exact GMM formulas.
- `verify.py`: assertions for continuous branches, finite tubes, trapping balls,
  nondegeneracy, slow tracking, and probability/MSE constants.
- `candidates.json`: rational candidate centers; no solver output is trusted.
- `certificate_result.json`: complete successful verification summary and source hashes.
- `finite_tubes.json`: every verified finite-step radius and derivative bound.
- `check_implementation.py`: independent rational arithmetic/Taylor checks and
  supplemental derivative checks.
- `implementation_checks.json`: supplemental check results.
- `explore.py`, `make_candidates.py`: optional floating-point proposal generation.
- `make_figure.py`: optional nominal-parameter illustration.
- `R1_path_certificate.png` and `.pdf`: scientific figure, not proof evidence.
- `verification.log`: complete certificate run log.

NumPy and SciPy are only needed to regenerate candidates or run the supplemental
derivative checks. Matplotlib is only needed to regenerate the figure. The
shipped certificate can be verified without any of these packages.

This is a computer-assisted proof draft, not an independently reviewed or
formally verified theorem. The interval implementation and correspondence
between assertions and claims remain reviewable parts of the proof.
