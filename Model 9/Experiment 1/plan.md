# Model 9 / Experiment 1 - a camera-invariant CNN, blended with the 55.80591 head

Date: 2026-10-02
Status: PRE-REGISTERED. Written before any CNN is trained or any score is seen.

An internal CV mean on 24 labelled soils is NOT a Kaggle score. The two are never subtracted.
Models 1-8 and their files are untouched; the two frozen files here (`transfer_eval.py`,
`ruler.py`) are verbatim copies so this experiment stands alone.

--------------------------------------------------------------------------------
0. THE ONE-LINE CLAIM THIS EXPERIMENT TESTS
--------------------------------------------------------------------------------
A convolutional network, trained per-tile on the exact EMD metric with aggressive CAMERA
colour-domain randomisation, learns a representation that transfers to the unseen iPhone test
camera BETTER than the handcrafted colour features can - and even if it does not win alone, its
errors decorrelate enough from the handcrafted head that a blend of the two beats 55.80591.

This is a genuine BET, and the plan says exactly which part is measurable now and which part is
not. Honesty first: Model 8 Experiment 4 measured this project as DATA-LIMITED (learning curve
falling into a ~35.3 EMD floor, ~2.9 EMD per data doubling). Beating 55.80591 is uphill. This
experiment attacks the ONE axis that verdict explicitly could not measure - the camera axis.

--------------------------------------------------------------------------------
1. WHY THIS, AND WHY IT IS DIFFERENT FROM EVERYTHING ALREADY TRIED
--------------------------------------------------------------------------------
What the project has established, and this plan respects:

  - E2: feature-set choice moves the board 11.46 EMD; regularisation is already optimal (alpha 3).
  - E3: colour ENCODING is worth under 2 EMD; the handcrafted head is already insensitive to how
        colour is written down, and it SURVIVED the camera split where DINOv2 LOST by 4.76 EMD.
  - E4: DATA-LIMITED. Eight model-side levers on the handcrafted head failed internally. The
        learning curve explains why: 24 soils simply do not carry more localisable signal for
        THIS head.

The trap in all of that: every one of those experiments varied a model that consumes 12
handcrafted features. "More model doesn't help" was measured for ONE model class. The learning
curve in E4 even says so in its own LIMIT field: it measures transfer among unseen ANDROID
families and "says nothing about the camera axis - there are 0 iPhone training images."

So the untested direction is not "a bigger head on the same features". It is a DIFFERENT MODEL
CLASS that (a) sees the pixels the 12 features throw away, and (b) can be explicitly trained to
be camera-invariant by domain randomisation - the one thing the handcrafted pipeline's fixed
grey-world correction cannot adapt and the learning curve cannot score.

Why not just "DINOv2 again"? Because DINOv2 here was a FROZEN feature extractor feeding the same
ridge, and it lost by 4.76. The three things this experiment changes versus that:
  1. END-TO-END fine-tuning on the exact EMD metric (not frozen features + a separate head).
  2. CAMERA colour-domain randomisation in training (DINOv2 saw Android colour and had no reason
     to be invariant to it).
  3. A monotone-CDF output head that is correct BY CONSTRUCTION, so no capacity is spent learning
     that a cumulative curve goes up.

--------------------------------------------------------------------------------
2. WHAT THE LITERATURE SAYS (the "net" search)
--------------------------------------------------------------------------------
  - SediNet (Buscombe 2020, Earth Surface Processes & Landforms): a configurable CNN for optical
    granulometry, predicting 9 grain-size percentiles from ~205 sediment images. This is the
    direct precedent: small-sample, multi-output grain-size regression from photographs WORKS.
  - Soranzo "GRAI3" (2026, Computers & Geotechnics): predicts soil PSD from standardised
    SMARTPHONE images with a deep-learning pipeline. Almost exactly our task.
  - Domain/colour randomisation is the most consistently supported method for sensor/camera
    shift: randomising colour and texture to force invariant features, "domain randomisation to
    reduce overfitting to source-domain appearance", "avoid overfitting on the colour during
    training". This is precisely our camera-split problem and the lever we build on.
  - Tile/patch aggregation into one prediction (multiple-instance style) is standard and is how
    we turn 24 soils into 1541 training instances without inventing new soils.

Nothing in the literature claims a few-dozen-sample regressor is easy. The honest read is:
the method is right for the problem shape; the sample count is the headwind.

--------------------------------------------------------------------------------
3. THE DESIGN (all of it fixed before training)
--------------------------------------------------------------------------------
DATA (shipped as a self-contained Kaggle ZIP built from data/):
  - 1541 train t256 tiles + 435 test t256 tiles (the materialised, soil_fraction>=0.50 set that
    every prior upload used), the three manifests, sample_submission.csv.
  - handcrafted_oof_train.csv : the 55.80591 head's family-held-out OOF on the 24 train soils
    (alpha=3.0, reproduces the oracle anchor 41.1475 to 1e-14). For honest blend-weight choice.
  - train_labels_matrix.csv   : the 24x11 labels, so the notebook scores blends without manifests.
  - Submission_Model8_E2_E2A.csv : the 55.80591 TEST predictions, the blend's handcrafted side.

INSTANCE / LABEL:
  - One training instance = one 256px tile; its label = its soil's 11-point cumulative curve.
  - Backbone input 224px (pretrained-weight compatible).

MODEL:
  - Pretrained backbone (default resnet34; timm with a torchvision fallback), fine-tuned
    end-to-end. Global-average-pooled features -> a monotone-CDF head:
        logits(11) -> softmax -> cumsum x100  ==>  non-decreasing, ends at exactly 100 by
    construction. The 11 softmax masses are the fraction of grains in each diameter bin; the
    cumsum is the cumulative curve. Physically exact, zero capacity wasted on monotonicity.

LOSS = THE METRIC:
  - EMD = trapezoid(|pred - true|, log10(diameters)) over the 11 supports, implemented in torch
    as the identical trapezoid weighting. It is fully differentiable, so we train directly on
    what Kaggle scores. No surrogate.

CAMERA DOMAIN RANDOMISATION (the bet, applied only in training):
  - Per-channel multiplicative gain, log-uniform in [0.80, 1.20] per channel independently. The
    measured camera gap is exactly this shape: Android grey-world gains run r~0.90 / b~1.13-1.26
    versus iPhone r~0.98 / b~1.03. Randomising channel gains across and beyond that span forces
    the network to NOT key on the Android colour signature, so an iPhone tile at test time is
    inside the trained distribution rather than outside it.
  - Plus standard ColorJitter(brightness, contrast, saturation, hue), flips, 90-degree rotations
    (grain texture is orientation-free), and a mild random-resized-crop.
  - Rationale is randomisation, NOT a directed Android->iPhone recolour: we must not bake in a
    guessed direction. Broad randomisation is what the literature supports and what cannot overfit
    to a wrong assumption.

INFERENCE / AGGREGATION:
  - Predict every test tile; median the per-tile CDFs within a soil (median of monotone curves is
    monotone, and the last column is 100 for all, so the soil curve stays valid); project as a
    safety net. Optional flip-TTA.

MODEL SELECTION - family-held-out, exactly the registered protocol:
  - Epoch count is chosen by leave-one-FAMILY-out CV (16 families from cv_families.csv): at each
    epoch, average the held-out soil EMD across folds; pick the epoch with the minimum. The final
    model is then trained on ALL 24 soils' tiles for that many epochs (seed-ensembled x3 to cut
    variance). No test soil, and no held-out family, ever touches a fit that scores it.
  - This CV ALSO yields the CNN's family-held-out OOF on the 24 train soils - the quantity the
    blend weight is chosen on.

BLEND:
  - Blended CDF = w * CNN + (1-w) * handcrafted, projected. w is chosen on the 24 soils by
    minimising internal LOFO EMD of (w*CNN_oof + (1-w)*hand_oof). This is an HONEST internal
    selection: both OOFs are family-held-out, labels are the train labels, the test set is never
    consulted.
  - A blend helps only if the components' per-soil errors decorrelate. We will REPORT that
    correlation. E4 found handcrafted-vs-handcrafted blending dead at correlation 0.86-0.95; a
    CNN is a different model class and should correlate far less. If it does not, the blend
    collapses to w=0 (pure handcrafted) and we will say so rather than force it.

--------------------------------------------------------------------------------
4. WHAT IS MEASURABLE NOW, AND WHAT IS NOT (the honest split)
--------------------------------------------------------------------------------
MEASURABLE INTERNALLY, THIS RUN, NO SUBMISSION:
  - The CNN's family-held-out OOF mean EMD on 24 soils (is the CNN any good at Android transfer?).
  - The error correlation between CNN and handcrafted (does a blend even have a chance?).
  - The internally-optimal blend weight and its internal LOFO EMD (does the blend beat 41.1475
    internally? - note that is the oracle-alpha anchor, not a Kaggle number).

NOT MEASURABLE INTERNALLY - this is the part that genuinely needs a submission:
  - Whether the camera-domain-randomisation BUYS anything on real iPhone images. With 0 iPhone
    training images, the camera axis is unobservable internally by construction (E4's LIMIT). The
    internal CV measures ANDROID transfer only. The camera benefit can appear ONLY on the board.
  - This is stated now, before any score, so it cannot be retrofitted as an excuse later.

--------------------------------------------------------------------------------
5. GATES (must pass before any submission is trusted)
--------------------------------------------------------------------------------
  G0  handcrafted anchors reproduce (nested 43.4532, oracle alpha=3 -> 41.1475) bit-exactly.
      [already passing locally: gaps 0.0 and 7e-15]
  G1  config_hash of the mounted dataset equals 010f44c36c74 (right data, pinned).
  G2  EMD implementation reproduces the host trivial baseline 100.31 (metric is correct).
  G3  monotone-CDF head output is non-decreasing and ends at exactly 100 for every row.
  G4  submission contract: 10 rows, 11 diameter columns, ids == sample_submission ids after the
      manifest submission_id join, all values in [0,100], monotone, last column 100, 0 NaN.
  G5  family CV leaks nothing: no held-out family appears in any fold that scores it.

--------------------------------------------------------------------------------
6. PRE-REGISTERED OUTCOMES - decided now, not after
--------------------------------------------------------------------------------
Let H = 55.80591 (handcrafted, the control; deterministic, a known reference, not a resubmission).

  (a) Blend beats H by > ~1 EMD on the board            -> the camera-invariant CNN adds real,
                                                           transferable signal. The main claim holds.
  (b) Blend ties H (within the ~5 EMD public band)      -> indistinguishable publicly; the CNN did
                                                           not clearly help on 3 public soils.
                                                           Private (70%) may still differ; we do not
                                                           pretend a tie is a win.
  (c) Blend or CNN scores WORSE than H                   -> the camera bet did not pay. The CNN's
                                                           Android-learned representation did not
                                                           transfer, exactly as DINOv2 did not.
                                                           A clean, publishable negative: "a
                                                           fine-tuned, camera-randomised CNN still
                                                           does not beat 12 handcrafted features on
                                                           this camera split." That tightens E4.
  (d) CNN-only >> blend, or blend w*~0                   -> errors did not decorrelate; the CNN is
                                                           not a useful ensemble partner here.

No outcome is spun. A public tie is not a win. The only claim that counts as "beat 55" is (a).

--------------------------------------------------------------------------------
7. SUBMISSION BUDGET - the user has ~6 slots across accounts; spend the FEWEST that learn the most
--------------------------------------------------------------------------------
The notebook writes several candidate CSVs but the RECOMMENDED order banks on learning per slot:

  SLOT 1 (highest EV): the BLEND at the internally-chosen weight. It is the lowest-variance bet
          to beat H, because it keeps the proven handcrafted head and only ADDS the CNN where the
          CNN helps. If the blend beats H, outcome (a); the camera bet paid.
  SLOT 2 (diagnostic, only if slot 1 is interesting or ties): the CNN-ONLY submission. Comparing
          CNN-only vs blend vs H on the board tells us WHICH part carried: a CNN that alone beats
          H means the representation transfers; a CNN that alone is much worse but whose blend
          still beats H means decorrelation (not transfer) did the work. Different lessons.

  We do NOT submit multiple blend weights - the public board's ~5 EMD resolution cannot separate
  neighbouring weights, so that would be slots burned for numbers inside the noise, the exact
  pattern E4 warned against. The handcrafted control H is already known (55.80591) and is not
  re-submitted.

--------------------------------------------------------------------------------
8. FILES
--------------------------------------------------------------------------------
  plan.md                         this pre-registration
  transfer_eval.py, ruler.py      verbatim copies (frozen head, metric, family protocol)
  make_handcrafted_oof.py         -> handcrafted_oof_train.csv, train_labels_matrix.csv  [done]
  build_notebook.py               emits the Kaggle notebook; compile-checks every cell locally
  Model9_Experiment1_kaggle.ipynb the 2xT4 GPU notebook (trains CNN, blends, writes submissions)
  build_dataset_zip.py            assembles final_kaggle_upload_m9e1.zip from data/
  verify_local.py                 checks ZIP contents, manifest join, submission contract, blend
  README.txt                      upload-and-run instructions + which CSV to submit first

Local python has no torch/timm, so the CNN cannot be run here. Everything that does NOT need
torch is verified locally (anchors, metric, OOF, ZIP, manifest join, submission contract, blend
math, and a compile-check of every notebook cell). The CNN itself runs only on Kaggle's GPUs.
