================================================================================
 MODEL 9 / EXPERIMENT 1  -  camera-invariant CNN, blended with the 55.80591 head
 Upload-and-run instructions
================================================================================

WHAT THIS IS
  A pretrained CNN (default resnet34) fine-tuned end-to-end, per 256px tile, DIRECTLY on the
  competition EMD metric, with camera colour-domain randomisation - then blended with the frozen
  handcrafted head whose test predictions scored 55.80591. The one axis this attacks that nothing
  before could measure is the CAMERA axis (train = Android only, test = iPhone only).

  Read plan.md for the full pre-registration: the claim, why it differs from the DINOv2 attempt
  that lost by 4.76, what is measurable internally and what is NOT, the gates, and the
  pre-registered outcomes. The honest headline: this project measured DATA-LIMITED, so beating
  55.80591 is uphill; the camera bet is the reason it might still pay.

--------------------------------------------------------------------------------
STEP 1 - make the dataset
--------------------------------------------------------------------------------
  Upload  final_kaggle_upload_m9e1.zip  as a new Kaggle Dataset.
  It contains: 1976 t256 tiles (data/tiles/...), the 3 manifests, sample_submission.csv,
  handcrafted_oof_train.csv, train_labels_matrix.csv, Submission_Model8_E2_E2A.csv (the 55.8 CSV).
  Nothing else is needed - the notebook is self-contained.

--------------------------------------------------------------------------------
STEP 2 - make the notebook
--------------------------------------------------------------------------------
  New Notebook -> File -> Import Notebook -> Model9_Experiment1_kaggle.ipynb
  Settings:
     Accelerator = GPU T4 x2        (the notebook uses both via DataParallel when present;
                                     it also runs on 1 GPU, just slower)
     Internet    = ON               (so the pretrained backbone can download its weights)
  Add data -> your final_kaggle_upload_m9e1 dataset.
  Run All.

  The notebook self-checks as it goes and STOPS on any failed gate:
     G1 config_hash == 010f44c36c74      G2 EMD reproduces trivial 100.31
     G3 monotone-CDF head (ends at 100)  G5 no family leak in any CV fold
     G4 submission contract on every CSV it writes

--------------------------------------------------------------------------------
STEP 3 - which CSV to submit (budget: spend the fewest slots that learn the most)
--------------------------------------------------------------------------------
  The notebook writes, into /kaggle/working:
     Submission_Model9_E1_BLEND_w<NN>.csv   blend at the INTERNALLY-chosen weight  <-- SUBMIT 1st
     Submission_Model9_E1_CNN.csv           CNN alone                              <-- SUBMIT 2nd
     Submission_Model9_E1_BLEND_w50.csv     a fixed 50/50 fallback (do not spend a slot on this
                                            unless the chosen-w blend and the CNN disagree a lot)

  SLOT 1  the BLEND at the chosen weight. Lowest-variance bet to beat 55.80591: it keeps the
          proven handcrafted head and only ADDS the CNN where the CNN helps. If this beats
          55.80591 by more than ~1 EMD, the camera bet paid (pre-registered outcome (a)).
  SLOT 2  the CNN ALONE. Diagnostic: if the CNN alone also beats 55.80591 the representation
          itself transferred; if the CNN alone is worse but the blend still won, decorrelation
          did the work. Different lessons - see plan.md section 6.

  Do NOT submit several blend weights - the public board's ~5 EMD resolution cannot separate
  neighbouring weights. The handcrafted control is the KNOWN 55.80591 and is not re-submitted.

--------------------------------------------------------------------------------
READING THE RESULT HONESTLY
--------------------------------------------------------------------------------
  - The notebook's CNN OOF and blend EMD are INTERNAL numbers on 24 Android soils. They are NOT
    Kaggle scores and must never be subtracted from one.
  - The camera-augmentation benefit is UNMEASURABLE internally (0 iPhone training images). Whether
    it helps on real iPhone test images can only appear on the board. That is the whole bet, stated
    before any score was seen.
  - A public tie with 55.80591 (within the ~5 EMD public band) is NOT a win; the private split
    (70% of the test soils) may still differ. Only a clear beat is outcome (a).

--------------------------------------------------------------------------------
IF YOU WANT TO TURN KNOBS (all in CFG, cell A1)
--------------------------------------------------------------------------------
  backbone          "resnet34" default; "resnet50"/"resnet18" or any timm name also work.
  chan_gain_lo/hi   the camera randomisation span [0.80, 1.20]; this IS the lever under test.
  cv_epochs         how far CV probes for the best epoch (default 24).
  n_seed_ensemble   final models averaged to cut seed variance (default 3).
  do_cv=False       skips CV (uses epochs_fallback) for a fast smoke-run; then there is no CNN OOF
                    so the blend weight defaults to 0.50 - use only to check the pipeline runs.

  A full 2x T4 run (CV over 16 families x 24 epochs + 3 final models) is the long pole; expect it
  to dominate the notebook's wall-clock. Reduce cv_epochs or n_seed_ensemble to trade accuracy of
  the epoch pick / variance for time.
================================================================================
