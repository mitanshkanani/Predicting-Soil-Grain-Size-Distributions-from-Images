# Model 7 - repository layout

Model 7 is experiment-specific work: each registered approach gets its own folder containing its own
contract, code, data, results and scratch. Documents that govern more than one approach live in
`shared/`. Nothing in this file changes any scientific content; it only says where things are.

## What exists

| Folder | Candidate | State |
|---|---|---|
| `Approach A/` | M7-A within-tile spatial organisation features (A1-A4, K=4) | **Implemented, run through Task 6, CLOSED `REFUTED-PRE-GATE`** on 2026-09-30. The alias-separability pre-gate failed, so no CV arm was ever scored and no submission exists. |
| `Approach C/` | M7-C selector-only alpha replacement (`SEL-1SE`) | **Pre-registered, not implemented, never run.** Empty by design except its README. |
| *(no folder)* | M7-B monotone-by-construction CDF head | **Not registered.** Record says "NOT REVIVED, by owner instruction. Frozen on the shortlist." No folder is created for it, because a folder would imply a registration that does not exist. |
| *(no folder)* | M7-E tail-emphasising loss weighting | **Not registered.** Collides with a cancelled AGENT_BRIEF line and the measured L1 null; recommended for kill. |
| *(no folder)* | M7-T transductive geometry from the 35 unlabeled test images | **Suspended, not pre-registered, explicitly not permitted to run.** |

## Where the authoritative documents are

- `instructions.txt` (this folder) - the model-level record: candidate statuses, the M5 boundary
  ruling, the scales that are not interchangeable, the Task 5 amendment record, and the
  `VERDICT - M7-A : REFUTED-PRE-GATE` block.
- `shared/docs/model7_preregistrations.md` - boundary rulings for M7-A/B/C/E/T (section 1) and the
  M7-C pre-registration (section 3).
- `Approach A/instructions.txt` - the M7-A experiment contract, including its frozen FILE MAP
  (lines 125-134).
- `Approach A/model_spec_m7a.md` - the approved M7-A pre-registration, including AMENDMENT A1.
- `Approach A/model_plan_m7a.md` - the M7-A implementation plan.
- `../AGENT_BRIEF.md` and `../research_map_1_2_5_6.md` - project-wide handoff and the Models 1-6
  evidence baseline. They are not Model 7 files and were not moved.

## Why `Approach A/` has no `code/` subfolder

Its layout is load-bearing and partly frozen. Every module locates the repository root as
`HERE.parents[1]` and its artifact directory as `HERE / "data"` (`m7a_data.py:73-77,89`,
`m7a_eval.py:48-52`, `alias_gate.py:43-47`), and `Approach A/instructions.txt:125-134` registers the
file map itself. Moving the Python files into `code/` would break the `transfer_eval` imports and
redirect the writers to `code/data/`, orphaning the recorded artifacts and duplicating
`alias_pairs.csv`. Repairing that needs path edits plus verification runs, which were not
authorised. So `Approach A/` keeps its registered shape:

```
Model 7/Approach A/
    instructions.txt        the experiment contract (also holds the frozen file map)
    model_spec_m7a.md       the approved pre-registration + AMENDMENT A1
    model_plan_m7a.md       the implementation plan
    spatial_features.py     production: A1-A4 per-tile statistics
    m7a_data.py             production: tile assembly, aggregation, census, publish guard
    m7a_eval.py             production: design matrices, alpha selection, LOFO arms
    alias_gate.py           production: the frozen pair set + amendment records + the pre-gate
    check_spatial_features.py  \
    check_m7a_data.py            | contract tests (510 / 324 / 58 / 53 checks, 0 failed)
    check_m7a_g0.py              |
    check_alias_gate.py         /
    data/                   derived feature tables and the census (5 artifacts)
    results/                empty - M7-A never reached an arm-scoring run
    scratch/                probes, suite logs, hash manifests (never evidence)
    __pycache__/            regenerable bytecode
```

`check_alias_gate.py` asserts that `results/` is empty and that no submission file exists, so nothing
is added to `results/`, including placeholders or `.gitkeep`.

## Scratch convention

Debug and diagnostic material stays in the repository-root `scratch/` quarantine (and in
`Approach A/scratch/` for M7-A's own probes). Root-level `scratch/` files whose names carry `m7a` or
`m7c` were left in place because the recorded verdict, the ledger and the scripts themselves cite
those exact paths. A scratch number is never experiment evidence.

## Organization history of this folder

2026-09-30, owner-authorised pass, moves only and no content edits:
`Model 7/Approach A/scratch_run.txt` -> `Model 7/Approach A/scratch/scratch_run.txt`, and
`model7_preregistrations.md` (repo root) -> `Model 7/shared/docs/model7_preregistrations.md`.
No Model 7 experiment was run as part of it.
