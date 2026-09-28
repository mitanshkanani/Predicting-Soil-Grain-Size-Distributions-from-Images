# Model 2 / Experiment 1 - Kaggle run 1, full transcript recovered from the executed notebook

**Source:** `soil-competition_model_2_exp1_run1.ipynb`, the notebook as Kaggle saved it
after the run (39/39 code cells executed, 0 errors).

**Why this file exists:** the run's own output CSVs (`Experiment1.txt`, `cv_per_fold.csv`,
`paired_effects.csv`, ...) were overwritten locally on 2026-09-28 by a mock dry run writing
into the same folder. The numbers were never at risk on Kaggle, and the executed notebook
carries every printed table, so the evidence is intact - but the CSVs are not, and this
transcript stands in for them until the probe run regenerates them. The routing bug is fixed:
a mock run now writes only into `local mock dry-run/`.

**Headline:** the decision gate did NOT fire. D 44.33 vs M1 43.02 vs R 39.78 in nested
in-domain LOGO-CV. The premise that a learned representation extracts more grain-size signal
than Model 1's 12 hand-built features is **REFUTED**. No submission was written by this run.

---

## Cell 1 - Cell A1 - configuration. Every tunable lives here and nowhere else.

```
Model2/E1  seed=20260930  context=kaggle
backbone arms requested this run: ('vit_random', 'dinov2')
outputs -> /kaggle/working
```

## Cell 2 - Cell A2 - imports and environment capture.

```
  python       3.12.13
  platform     Linux-6.12.90+-x86_64-with-glibc2.35
  numpy        2.0.2
  pandas       2.3.3
  scipy        1.16.3
  sklearn      1.6.1
  matplotlib   3.10.0
  Pillow       11.3.0
```

## Cell 3 - Cell A3 - locate the processed dataset and the backbone interface.

```
INPUT_ROOT  /kaggle/input/datasets/mitanshkanani/soil-gsd-processed-final-fixed  (kaggle)
backbones   /kaggle/input/datasets/mitanshkanani/soil-gsd-processed-final-fixed/backbones.py
```

## Cell 4 - Cell A4 - pin the preprocessing version, the submission schema, and the seed.

```
config_hash 010f44c36c74 | pipeline 1.0.0 | 4.552516 px/mm
tile 256px = 56.23 mm square -> resized to 224px = 49.20 mm
one ViT-S/14 patch = 3.075 mm at the canonical scale
```

## Cell 5 - Cell B1 - load manifests and labels. Labels come from the SAMPLE manifest only.

```
images (162, 103) | tiles@256 (1976, 21) (train 1541 / test 435) | labels (24, 11)
D50 spans 0.0213 .. 6.126 mm

SCALE CAVEAT (measured here, stated up front, not discovered afterwards): one ViT-S/14 patch = 3.075 mm.
  12 of 24 training soils have D50 < 1.03 mm, so a single patch spans MORE THAN 3 median
  grains for those soils. 16 of 24 have a median grain smaller than one whole patch.
  10 of 24 have D50 < 0.11 mm, where the patch is 28x the median grain.
  For those soils the network sees fines as TEXTURE, never as particles. Cropping a smaller
  physical area and upsampling so patches land at sub-millimetre size is a DIFFERENT variable
  and belongs in Model 2 E2. If D fails here it may be magnification, not representation.
```

## Cell 6 - Cell B2 - the EMD implementation and its reconciliation against the page.

```
trivial baseline: ours 100.3132 | page formula 102.0200 | host publishes 100.31
METRIC CONFIRMED (as in E2/E3): the trapezoid matches the host's own number.
```

## Cell 7 - Cell B3 - reference baselines and the five external ground-truth scores.

```
reference EMD on the training set (lower is better):
  official trivial baseline (9.09%/bin)         100.31
  constant train MEAN curve                      85.39
  constant train MEDIAN curve (no images)        82.62

external ground truth so far:
    172.70  E1 16 feats, alpha 0.03 chosen by LOGO-CV
    102.37  no-image probe (train median curve)
     71.27  E2 14 feats, alpha 30 chosen by CAM+RES
     61.24  E3 12 feats, alpha 10 chosen by CAM+RES
     77.12  E4  5 feats, alpha 30 chosen by CAM+RES

no-image in-domain floor 82.62. The rank-3 representation ceiling measured in E1 was 7.35.
```

## Cell 8 - Cell C1 - locate and load Experiment 3's outputs.

```
E3_DIR /kaggle/input/datasets/mitanshkanani/soil-gsd-processed-final-fixed/Model 1 Experiment 3
  features_soil.csv (34, 18) | cv_families.csv (24, 5) | submission (10, 12)
```

## Cell 9 - Cell C2 - the plateau that motivated Model 2, and why in-domain is the right ruler here.

```
MODEL 1 TRAJECTORY

exp    in-domain LOGO-CV   CAM+RES    Kaggle
E1                 35.98    205.34    172.70
E2                 42.66     75.73     71.27
E3                 41.15     51.87     61.24
E4                 39.46     56.98     77.12

  The in-domain column did not move. Every external gain came from removing
  features and raising shrinkage, which is variance reduction, not new signal.
  Headroom to the rank-3 ceiling: 28.6 EMD in-domain.

WHY LOGO-CV IS ADMITTED HERE, AFTER E1 DEMOTED IT

  E1's finding was precise: LOGO-CV ranks configurations in the WRONG ORDER for
  cross-camera transfer (rank correlation -0.953 against CAM+RES). That is a claim
  about transfer. It is not a claim about how much curve signal a representation
  carries on devices already seen, and the question Model 2 E1 asks is exactly the
  second one: does a learned representation lower in-domain error at all?
  If it does not, Model 2 is refuted cheaply on a CPU metric with no submission spent.
  If it does, a submission has been earned to discover how the transfer penalty
  behaves under a new representation.

  CAM+RES is computed in section K for reporting and for new calibration points in
  the ruler's own failure analysis. It selects nothing in this experiment.
```

## Cell 10 - Cell D1 - what can actually run in this process.

```
  torch        2.10.0+cu128
  timm         1.0.26
  cuda         True 2

arms requested / arms executable:
  M1          dim=  12  torch=False   executable=True
  vit_random  dim= 384  torch=True    executable=True
  dinov2      dim= 384  torch=True    executable=True

no DINO_WEIGHTS set: timm will fetch pretrained weights from the hub, which needs internet ON.
  kaggle_setup.md gives the offline dataset alternative. vit_random needs no download at all, which is why it is run first.
embedding seed 20260930 | batch 64
```

## Cell 11 - Cell E1 - run the backbone contract test against the arms this notebook is about to use.

```
running: /usr/bin/python3 /kaggle/input/datasets/mitanshkanani/soil-gsd-processed-final-fixed/check_backbones.py vit_random dinov2 


torch/timm present: True

2. mock backend - the path that must run everywhere
  [PASS] mock: output shape is (N, 384)  (7, 384)
  [PASS] mock: float dtype  float64
  [PASS] mock: all values finite
  [PASS] mock: rows are L2-unit-norm  max |1-norm| = 1.11e-16
  [PASS] mock: not all-zero
  [PASS] mock: same input twice -> byte-identical
  [PASS] mock: different tiles -> different embeddings  d_same 0.00e+00 vs d_diff 0.0573
  [PASS] mock: per-image standardisation kills a uniform brightness shift  max |delta| = 2.15e-16

3. determinism of the mock random map
  [PASS] re-seeded mock differs

4. degenerate inputs do not produce NaN
  [PASS] constant-colour tiles give finite output  a zero-variance tile would divide by zero without the guard

5. input validation
  [PASS] 2-D input is rejected
  [PASS] unknown backend is rejected

6. the mock arm is UNINFORMATIVE by construction
  [PASS] mock embeddings do not predict an unrelated held-out target  LOO R^2 = -0.211

7. backbone arms
  --- vit_random (real) ---
  backbone construction accepted: {'img_size': 224, 'dynamic_img_size': True}
  [PASS] vit_random: output shape is (N, 384)  (7, 384)
  [PASS] vit_random: float dtype  float64
  [PASS] vit_random: all values finite
  [PASS] vit_random: rows are L2-unit-norm  max |1-norm| = 1.11e-16
  [PASS] vit_random: not all-zero
  [PASS] vit_random: same input twice -> byte-identical
  [PASS] vit_random: different tiles -> different embeddings  d_same 0.00e+00 vs d_diff 0.0000
  [INFO] vit_random: max |delta| under a +40 brightness shift = 0.0000 (reported only; a frozen network is not a photometric normaliser)
  [PASS] vit_random: batch_size smaller than N concatenates correctly
  --- dinov2 (real) ---
  [PASS] dinov2: output shape is (N, 384)  (7, 384)
  [PASS] dinov2: float dtype  float64
  [PASS] dinov2: all values finite
  [PASS] dinov2: rows are L2-unit-norm  max |1-norm| = 1.11e-16
  [PASS] dinov2: not all-zero
  [PASS] dinov2: same input twice -> byte-identical
  [PASS] dinov2: different tiles -> different embeddings  d_same 0.00e+00 vs d_diff 0.0060
  [INFO] dinov2: max |delta| under a +40 brightness shift = 0.0319 (reported only; a frozen network is not a photometric normaliser)
  [PASS] dinov2: batch_size smaller than N concatenates correctly
  [PASS] vit_random: two seeds give two different networks

========================================================================
check_backbones: ALL CHECKS PASSED
verified backends: mock, vit_random, dinov2. Unverified by this run: none.

PASS: backbone interface contract holds for vit_random, dinov2
```

## Cell 12 - Cell E2 - the hand-built per-tile feature function, reused verbatim from E1/E2/E3.

```
control features 12: ['e4', 'e8', 'e16', 'lum_sd', 'grad_mean', 'R', 'G', 'B', 'sat', 'lum_p10', 'lum_p50', 'lum_p90']
hand-built columns extracted: 16 (extra ones are carried for the mechanism table only, never into the head)
```

## Cell 13 - Cell E3 - resolve the hand-built tile cache, searching this experiment then E3 then E1.

```
resolving hand-built features for 1976 tiles ...
  hand features from /kaggle/working/.cache: (1976, 17)
```

## Cell 14 - Cell E4 - the resolution-matched blur, derived from the manifests exactly as E3 did.

```
  Motorola  -> iPhone 14    extra anti-alias sigma 0.8346 px = 0.1833 mm
  Motorola  -> iPhone 16    extra anti-alias sigma 1.2033 px = 0.2643 mm
  Samsung   -> iPhone 14    extra anti-alias sigma 0.8356 px = 0.1835 mm
  Samsung   -> iPhone 16    extra anti-alias sigma 1.2039 px = 0.2645 mm
resolving RES-MATCH view 'res14' (iPhone 14) for 1541 training tiles ...
  res14 from /kaggle/working/.cache
resolving RES-MATCH view 'res16' (iPhone 16) for 1541 training tiles ...
  res16 from /kaggle/working/.cache
```

## Cell 15 - Cell E5 - the embedding arms, cached as .npy next to the tile order they were built in.

```

extracting embeddings: backend=vit_random dim=384 over 1976 real tiles
extracting embeddings: backend=vit_random view=res14 over 1541 train tiles
extracting embeddings: backend=vit_random view=res16 over 1541 train tiles

extracting embeddings: backend=dinov2 dim=384 over 1976 real tiles
extracting embeddings: backend=dinov2 view=res14 over 1541 train tiles
extracting embeddings: backend=dinov2 view=res16 over 1541 train tiles

embedding tables built: [('dinov2', 'real'), ('dinov2', 'res14'), ('dinov2', 'res16'), ('vit_random', 'real'), ('vit_random', 'res14'), ('vit_random', 'res16')]
```

## Cell 16 - Cell F1 - the tile metadata table and the materialised-tile filter.

```
tiles 1976 -> materialised+soil-fraction>= 0.5: 1946 (98.5%)
split  camera_fam
test   iPhone        405
train  Motorola      716
       Samsung       825
```

## Cell 17 - Cell F2 - tile -> image -> soil, one code path for every arm.

```
soil-level tables built:
  ('M1', 'real')                 (34, 12)
  ('M1', 'res14')                (24, 12)
  ('M1', 'res16')                (24, 12)
  ('dinov2', 'real')             (34, 384)
  ('dinov2', 'res14')            (24, 384)
  ('dinov2', 'res16')            (24, 384)
  ('vit_random', 'real')         (34, 384)
  ('vit_random', 'res14')        (24, 384)
  ('vit_random', 'res16')        (24, 384)
```

## Cell 18 - Cell F3 - prove the control arm IS Experiment 3's matrix, to floating-point noise.

```
max absolute drift vs Experiment 3 across 12 features: 2.842e-14
worst feature: B
REPRODUCED: the control arm is byte-for-byte Model 1 E3's feature matrix.
```

## Cell 19 - Cell F4 - copy the CV families and assert they are E1/E2/E3's, unchanged.

```
16 families over 24 soils (6 multi-member, 10 singletons), copied unchanged from E1/E2/E3.
cluster-EMD cut 14.0, average linkage on the label curves - the approved protocol.
```

## Cell 20 - Cell G1 - the arm table, and the assertions that make them comparable.

```
        backend  n_feat               kind              pretrained  train_rows  test_rows  zero_var_cols
arm                                                                                                     
M1           M1      12         hand-built  n/a (Model 1 features)          24         10              0
D        dinov2     384  learned embedding     yes: DINOv2 lvd142m          24         10              0
R    vit_random     384  learned embedding         no: random init          24         10              0

  A p>n head (384 features on 24 training soils) is handled by ridge in closed
  form. No embedding-PCA step is added: that would be a SECOND change to the
  pipeline and would make the comparison to Model 1 uninterpretable.
  no zero-variance training columns in any arm.

  p / n per arm (features / training soils): M1=12/24, D=384/24, R=384/24
```

## Cell 21 - Cell G2 - the honest statement of what this run can and cannot conclude.

```
arms present: ['M1', 'D', 'R']

  All three gate arms are present: the decision in section L is evaluable.
```

## Cell 22 - Cell H1 - model primitives, byte-for-byte Model 1's, plus the leak assertion.

```
arms scored: ['M1', 'D', 'R']
alphas: 16   folds: 16   fits per arm: 256
leak assertion is inside logo_errors, so it runs on every fold of every arm.
```

## Cell 23 - Cell H2 - the structural error table: every arm, every alpha, every fold, once.

```
  M1    done (1s cumulative)
  D     done (2s cumulative)
  R     done (3s cumulative)

fold sizes: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 4]  (1 = singleton family)
Every soil appears in exactly one evaluation fold, so every one of the 24 training
soils is scored out-of-fold.
```

## Cell 24 - Cell H3 - the in-domain alpha sweep, oracle and nested, per arm.

```
     n_feat  oracle_alpha  oracle_emd  parsimony_alpha  parsimony_emd  grid_spread  parsimony_is_grid_max
arm                                                                                                      
M1       12          3.00       41.15             30.0          51.70        51.87                  False
D       384        300.00       43.19           1000.0          48.28        49.62                  False
R       384          0.03       39.78            100.0          52.18        53.08                  False

  Full sweep, in-domain LOGO-CV mean EMD by alpha:
               M1      D      R
0.03        53.80  54.04  39.78
0.10        48.35  54.02  39.83
0.30        43.92  53.94  39.99
1.00        41.28  53.68  40.46
3.00        41.15  52.92  41.40
10.00       45.65  51.05  43.31
30.00       51.70  48.01  46.51
100.00      64.87  44.26  52.18
300.00      78.19  43.19  58.00
1000.00     87.41  48.28  63.49
3000.00     91.00  60.83  71.37
10000.00    92.40  77.17  81.56
30000.00    92.82  86.57  88.11
100000.00   92.96  90.93  91.37
300000.00   93.00  92.31  92.46
1000000.00  93.02  92.81  92.85
```

## Cell 25 - Cell H4 - the nested procedure score per arm, and the per-fold record.

```
PRIMARY RESULT - nested in-domain LOGO-CV (lower is better):

  M1    n=24  mean  43.02  median  39.76  max 104.73   alphas used: [1.0, 3.0]
  D     n=24  mean  44.33  median  38.50  max 124.24   alphas used: [100.0, 300.0]
  R     n=24  mean  39.78  median  30.41  max  76.58   alphas used: [0.03]

  no-image floor 82.62 | rank-3 representation ceiling 7.35
  MOCK is expected to sit at or above the floor. If mock scores WELL, the pipeline is
  leaking and nothing in this notebook means anything - check_backbones' R^2 test and
  this line are the two independent guards against that.
```

## Cell 26 - Cell I1 - paired per-soil contrasts with a soil-level bootstrap.

```
paired on all 24 training soils, each scored out-of-fold

contrasts are in EMD units, lower is better, so a NEGATIVE estimate means the
first arm scores better:

  D - R  (pretraining, capacity held fixed)    +4.55  CI [   -7.02,  +16.78]  wins 11/24  not distinguishable from zero
  D - M1 (representation vs hand-built)    +1.31  CI [   -6.67,   +9.36]  wins 11/24  not distinguishable from zero
  R - M1 (capacity alone vs hand-built)    -3.24  CI [  -12.53,   +6.06]  wins 13/24  not distinguishable from zero
  MOCK - M1 (plumbing control, not a result)   --  skipped: an arm is not present in this run

  Reading rule fixed in advance: the claim 'pretraining helps' requires C_DR to be
  negative AND significant. If C_RM is also negative, dimensionality helps by itself,
  and that is the finding even if C_DR is null.
```

## Cell 27 - Cell J1 - camera shift and sharpness shift, in the same z convention E3/E4 used.

```
dual-camera soils available: 21
           n_feat  camera_shift_z  camera_shift_z_max  sharpness_shift_z  sharpness_shift_z_max
arm                                                                                            
M1             12           1.214               2.081              0.714                  2.998
M1:core         5           0.756               2.081              0.873                  2.998
M1:colour       7           1.541               2.000              0.600                  1.672
D             384           0.620               2.465              0.232                  1.250
R             384           1.262               2.182              0.321                  0.777

  reference from Model 1: colour block camera shift 1.54 z, texture core 0.85 z,
  absolute-frequency features 4.31 z under blur. Those are the numbers the
  embedding arms are being compared against.
```

## Cell 28 - Cell J2 - a scale-free version of the same question, for the embedding arms.

```
     camera_distance  soil_distance  camera_over_soil  blur_distance  blur_over_soil
arm                                                                                 
M1            0.0031         0.0028            1.1048         0.0023          0.8182
D             0.0392         0.1875            0.2090         0.0054          0.0286
R             0.0000         0.0000            1.3340         0.0000          0.1078

  camera_over_soil ~ 1 -> the device is as informative as the soil (bad).
  ~0 -> the representation ignores the camera (the property Model 1 never had).
  NOTE: cosine geometry is meaningful for L2-normalised embeddings; for M1's 12
  unnormalised features the same formula is computed for comparability but is a
  weaker summary, which is why the z-table above is the primary mechanism readout.
```

## Cell 29 - Cell K1 - the ruler conditions and the leak assertion, as in E3.

```
CAM+RES conditions (4): Mot>Sam:res14, Mot>Sam:res16, Sam>Mot:res14, Sam>Mot:res16
CAM conditions     (2): Mot>Sam:real, Sam>Mot:real
PASS: 21 dual-camera soils, no evaluated soil and no soil of its own
family appears in its own fit under any condition.
```

## Cell 30 - Cell K2 - CAM+RES at the alphas the in-domain procedure actually chose.

```
     n_feat  camres_at_parsimony  camres_at_logo_alpha  camres_best_alpha  camres_best  cam_at_parsimony  in_domain_nested
arm                                                                                                                       
M1       12                54.98                 55.16              30.00        54.98             55.40             43.02
D       384                54.01                 47.08             300.00        47.08             53.95             44.33
R       384                91.89                 78.55               0.03        78.55             99.71             39.78

  NOT USED TO SELECT. Read it as three new calibration points for the ruler's
  failure analysis: Model 1's four experiments produced factors 0.84 -> 1.35
  (Kaggle / CAM+RES) drifting monotonically, and one inverted pair (E2 vs E4).

  existing calibration points (CAM+RES, Kaggle, factor):
    E1:  205.34 -> 172.70   factor 0.84
    E2:   75.73 ->  71.27   factor 0.94
    E3:   51.87 ->  61.24   factor 1.18
    E4:   56.98 ->  77.12   factor 1.35
  A Model 2 arm would be the fifth point. If the learned arms score well on CAM+RES
  but the in-domain table says they are not better than R, the ruler is still wrong
  and the gate still does not fire.
```

## Cell 31 - Cell L1 - submit only if D beats the control AND beats the random-init ablation.

```
gate clauses:

  [FAIL] in_domain(D) 44.33 < in_domain(M1) 43.02
  [FAIL] in_domain(D) 44.33 < in_domain(R) 39.78
  [FAIL] paired CI of (D - R) excludes zero with D better   CI [-7.02,+16.78]

GATE -> NO SUBMISSION

  outcome table, fixed in the plan before the run:
   D beats control and beats random-init -> pretraining is real: submit, gain a
     fifth calibration point for the ruler.
   D beats control but not random-init   -> it is CAPACITY, not pretraining: do not
     submit; Model 2 E2 tests magnification instead.
   D does not beat control               -> representation is not the bottleneck: the
     34 EMD in-domain headroom is a data limit at n=24, and the remaining time goes
     to robustness of Model 1's 61.24 configuration, not to new representations.
```

## Cell 32 - Cell M1 - fit the submission arm and predict the 10 test soils.

```
PLUMBING CHECK on the control arm (no file will be written)
  arm M1: 12 features, alpha 30 (parsimony rule)
  columns pinned at 0/100: E1 59.1%  E3 13.6%  this 10.0% (real labels 27%)
  mean EMD of predictions vs the train mean curve: 44.25
```

## Cell 33 - Cell M2 - eight validation gates, then write only if the gate fired.

```
eight submission gates (plumbing check on the control arm - the file is NOT written):
  PASS  row count == 10
  PASS  ids exactly match sample_submission order
  PASS  columns exactly match
  PASS  no missing values
  PASS  values within [0,100]
  PASS  cumulative / non-decreasing
  PASS  200mm column == 100
  PASS  no extra columns

all 8 submission checks passed
NO SUBMISSION WRITTEN. The gates above validated the machinery only. Reason recorded in gate.json:
  NOT  in_domain(D) 44.33 < in_domain(M1) 43.02
  NOT  in_domain(D) 44.33 < in_domain(R) 39.78
  NOT  paired CI of (D - R) excludes zero with D better
```

## Cell 34 - Cell N1 - the in-domain CV error by soil, per arm.

```
<Figure size 1050x440 with 1 Axes>
```

## Cell 35 - Cell N2 - per-soil differences against the control, with family grouping.

```
<Figure size 1050x400 with 1 Axes>
  D     - M1: wins 11/24 soils, median   +1.27, mean   +1.31, worst soil G190  +43.44, best soil H366  -35.66
  R     - M1: wins 13/24 soils, median   -4.58, mean   -3.24, worst soil H493  +43.79, best soil F827  -46.58
```

## Cell 36 - Cell N3 - the submission curves, if one exists.

```
No submission curves: the gate did not fire, so nothing was written.
The control-arm plumbing predictions from section M are not plotted here - they
would be indistinguishable from E3 and would invite a comparison nobody asked for.
```

## Cell 37 - Cell O1 - the pre-registered predictions, evaluated mechanically.

```
predictions were written in the plan BEFORE implementation; the blind/confirmation
label is fixed and may not be edited afterwards.

  P1  [not evaluated ] mock in-domain LOGO-CV is worse than the no-image floor 82.62
      mock arm not executed in this run
  P2  [CONFIRMED     ] DINOv2 embeddings shift LESS between cameras than the colour block does
      D camera shift 0.620 z vs colour block 1.541 z (E4 reported 1.54). Diagnostic, not decisive.
  P3  [REFUTED       ] random-init arm R does NOT beat the Model 1 control in-domain
      R 39.78 vs M1 43.02. If R wins too, the effect is width, not pretraining.
  P4  [REFUTED       ] DINOv2 arm D beats the Model 1 control in-domain
      D 44.33 vs M1 43.02; paired D-M1 +1.31 EMD, CI [-6.67,+9.36], wins 11/24 soils, not distinguishable from zero
  P5  [REFUTED       ] D beats R by a paired margin whose CI excludes zero
      paired D-R +4.55 EMD, CI [-7.02,+16.78], wins 11/24 soils, not distinguishable from zero
  P6  [not applicable] if a submission is made it beats 61.24
      no submission this run (gate submit=False). E3's 61.24 stands.
```

## Cell 38 - Cell O2 - write Experiment1.txt from the live objects.

```
wrote /kaggle/working/Experiment1.txt (133 lines)
```

## Cell 39 - Cell O3 - summary.

```
========================================================================
Model2/E1  context=kaggle  arms=['M1', 'dinov2', 'vit_random']  torch=True

nested in-domain LOGO-CV (primary):
  M1      43.02
  D       44.33
  R       39.78
  no-image floor 82.62 | rank-3 ceiling 7.35

gate: evaluable=True submit=False
submission: NOT written

files written:
  ok   Experiment1.txt
  ok   arms_config.csv
  ok   cv_per_fold.csv
  ok   cv_per_soil.csv
  ok   alpha_selection_in_domain.csv
  ok   paired_effects.csv
  ok   mechanism_table.csv
  ok   embedding_separation.csv
  ok   camres_report_only.csv
  ok   predictions.csv
  ok   gate.json
  ok   features_soil_M1_control.csv
  ok   cv_families.csv
```
