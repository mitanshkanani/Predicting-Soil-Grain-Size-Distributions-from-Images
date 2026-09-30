# Model 7 / shared

Documents that govern more than one Model 7 approach. Experiment-specific contracts stay inside their
own `Approach X/` folder; only genuinely cross-approach material belongs here.

## Contents

- `docs/model7_preregistrations.md` - the Model 7 candidate register, approved 2026-09-29. It covers
  M7-A, M7-B, M7-C, M7-E and M7-T, and holds:
  - section 1a - why M7-T is suspended rather than killed by rule
  - section 1b - why M7-E collides with a cancelled line and the measured L1 null
  - section 1c - the M7-A boundary narrowing (the K=4 statistic set)
  - section 1d - the M7-C boundary correction: the measured 1.68 EMD oracle regret, fold-choice
    instability, and the exclusion of the adaptive estimator from the primary arm
  - section 3 - the M7-C pre-registration, thresholds fixed and unrun
  - section 4 - M7-A and M7-C side by side, plus the constraints both candidates share

  The M7-A text that once lived here was moved on 2026-09-29 to
  `Model 7/Approach A/model_spec_m7a.md`, which is the authoritative copy; this register keeps only
  sections 1 and 3.

## Known breakage from the move, reported rather than patched

This file used to sit at the repository root as `model7_preregistrations.md`. Two things still point
at the old location, and neither was edited during the organization pass:

1. `scratch/check_m7_prereg.py:15` builds `ROOT / "model7_preregistrations.md"`. On its next run that
   static checker will raise `FileNotFoundError` until the path is updated. It is a repo-root scratch
   diagnostic, not experiment code, and its previous run output is preserved in
   `scratch/_check_m7_prereg.log` and `scratch/_m7_prereg_checks.log`.
2. `Model 7/Approach A/model_spec_m7a.md:8` describes the register as living "at the repository root,
   unchanged". That sentence is now stale as to location. It was left untouched because the owner
   ruled that no scientific file may be edited for organization reasons; correcting it needs a
   documented amendment, the same way AMENDMENT A1 did.

Text citations elsewhere (`Model 7/instructions.txt`, `Approach A/instructions.txt:8`, the plan's
file list) name the file without a directory, so they still resolve.
