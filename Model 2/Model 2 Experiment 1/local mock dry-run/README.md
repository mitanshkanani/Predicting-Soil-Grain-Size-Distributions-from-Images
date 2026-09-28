# THIS IS NOT A RESULT

These are the artifacts of the **local mock dry run** (`BACKEND=mock`, no torch on this
machine), kept for provenance and deliberately separated from the experiment record.

What this run proves, and what it must never be quoted as:

| it proves | it does not prove |
|---|---|
| the pipeline executes end to end without torch present | anything about DINOv2, or about learned representations |
| the control arm reproduces Experiment 3's feature matrix to 2.8e-14 | that mock's 87.82 is a score |
| the mock arm sits at 87.82, worse than the no-image floor 82.62, so the pipeline does not manufacture signal out of nothing | |
| the decision gate refuses to fire and **no submission is written** | |

`mock` is a fixed random linear map of per-image-standardised grayscale pixels. It contains
no learned content by construction. A mock arm that scored *well* would itself be evidence
of a leak, which is why this file exists.

The authoritative record is one level up: `../Experiment1.txt`, produced by the Kaggle run
with all three arms (M1 / R / D) present.
