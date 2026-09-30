# Model 7 / Approach C - selector-only alpha replacement (SEL-1SE)

**Status: PRE-REGISTERED. NOT IMPLEMENTED. NEVER RUN. NO RESULTS.**

This folder exists so that the repository layout shows which Model 7 candidates are registered but
unstarted. It contains no code, no data, no results, and no evidence.

| | |
|---|---|
| Implemented? | No. There is no `code/`, no notebook, no generator, no test file here. |
| Run? | No. No CV arm, no bootstrap, no prediction, no submission has ever been produced for M7-C. |
| Results? | None. Anything appearing in this folder in the future must be produced by a registered run. |
| Eligible now? | Yes on sequence: the contract makes M7-C the next step because M7-A was rejected at its
  pre-gate (`Model 7/instructions.txt`, "VERDICT - M7-A : REFUTED-PRE-GATE"). Eligible is not
  authorised; the owner must approve the run. |

## Where the definition lives

The registration is **not copied here**. It stays in its single authoritative home:

- `Model 7/shared/docs/model7_preregistrations.md`
  - section 1d - the boundary correction found while verifying M7-C's mechanism (the 1.68 EMD oracle
    regret, the fold-instability measurements, and why the adaptive estimator was excluded from the
    primary arm)
  - section 3 - the M7-C pre-registration itself: hypothesis H7-C, the `SEL-1SE` rule, the frozen
    validation protocol, the five controls, gates G1-G6, verdict precedence and the submission rule
- `Model 7/instructions.txt` - "STATUS OF EVERY SHORTLIST CANDIDATE", the M7-C line

Gain is quoted in internal CV EMD only. `43.0217308796477` is the historical internal anchor and
`43.453225201811563` is the Model 7 internal nested-LOFO baseline; neither is a Kaggle score, and
internal CV is never evidence of Kaggle performance.

## What must not be inferred from this folder

The existence of this directory is not a decision to run M7-C, not a result, and not a claim that the
selector has any effect. Read the registration before treating it as anything else.
