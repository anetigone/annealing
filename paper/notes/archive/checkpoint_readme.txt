R1 feasibility checkpoint, 2026-10-28

Read R1_PROGRESS.md for the proof, definitions, empirical results and limitations.

Reproduce from this directory using Python 3.12:
  python3 r1_experiment.py --out results --samples 200000
  python3 r1_three_templates.py
  python3 r1_three_geometry.py

The implementation uses CPU/float64 and an exact Gaussian-mixture MMSE denoiser.
There are no learned network weights or external datasets.
The three-template example was selected by an exploratory synthetic search.
The two-template proposition does NOT prove the three-template numerical result.
All reported comparisons within a case use paired observations and initial states.
See SHA256SUMS.json for file checksums generated when this package was produced.
