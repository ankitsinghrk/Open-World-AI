OPEN-WORLD AI — E1 RANDOMNESS DECOMPOSITION
===========================================

PURPOSE
-------
The existing E1 ten-seed study changes BOTH NumPy sample selection and
PyTorch training randomness when --seed changes. This follow-up separates
those sources while preserving the E1 protocol.

FILES
-----
1. open_world_e1_randomness_decomposition.py
   New runner for the two controlled experiments.

2. Keep this existing helper script in the SAME experiments folder:
   open_world_static_100_baseline_v2.py

Do not replace or modify the existing E0/E1 scripts. The new runner imports
and reuses E1 v2's tested data loading, scaling, MLP, training, prediction,
metrics, and hashing functions.

EXPERIMENT A — SAMPLE-SELECTION EFFECT
--------------------------------------
Vary NumPy sampling seed; keep PyTorch training seed fixed at 42.
This measures the combined effect of which 100 rows are selected, the class
composition of those rows, and the scaler fitted on those selected rows,
while holding model initialization and minibatch-order randomness fixed.

Run from the project root in PowerShell:

python .\experiments\open_world_e1_randomness_decomposition.py --mode selection --seeds 42 43 44 45 46 47 48 49 50 51 --fixed-training-seed 42

Outputs:
experiments\results\e1_randomness_decomposition\selection_effect\

EXPERIMENT B — TRAINING-RANDOMNESS EFFECT
-----------------------------------------
Keep NumPy sampling seed fixed at 42; vary PyTorch training seed from 42 to 51.
The selected 100 rows and their fitted scaler remain identical. This measures
variation from model initialization and minibatch order controlled by PyTorch.

Run from the project root in PowerShell:

python .\experiments\open_world_e1_randomness_decomposition.py --mode training --seeds 42 43 44 45 46 47 48 49 50 51 --fixed-sampling-seed 42

Outputs:
experiments\results\e1_randomness_decomposition\training_effect\

LOCKED E1 SETTINGS
------------------
- Exactly 100 labels from Batch 1
- Uniform sampling without replacement, independent of labels
- MinMax scaler fit only on the selected 100 rows; no future clipping
- MLP: 128 -> 64 -> ReLU -> 6
- Adam, learning rate 0.001
- 60 epochs, batch size 32, CPU
- No validation set or checkpoint selection
- Batches 2–10 are evaluation-only; no model updates

OUTPUTS
-------
Each run folder contains:
- metrics.json
- metrics.csv
- selected_training_rows.csv
- model_and_scaler.pt

Each mode folder also contains summary.json and summary.csv.
The runner refuses to overwrite existing outputs. If a run is interrupted
or the destination already exists, choose a new --output-root, for example:

--output-root .\experiments\results\e1_randomness_decomposition_retry1

REFERENCE ANCHOR
----------------
The (sampling_seed=42, training_seed=42) run should reproduce the E1 seed 42
metrics under the same dataset, helper script and runtime. This anchor is a
useful consistency check for both modes.

INTERPRETATION LIMITS
--------------------
- Experiment A captures the full data-selection pathway, including the
  selected class composition and the scaler fit on those rows. It is not a
  pure class-count-only effect.
- Experiment B captures PyTorch model initialization and minibatch-order
  randomness. It does not separately estimate initialization versus order.
- These are controlled one-factor-at-a-time studies around seed 42, not a
  full 10x10 factorial interaction study.
- They do not establish that concept drift alone caused performance decline.
- Do not edit src/ for these experiments.

RECOMMENDED ORDER
-----------------
1. Run Experiment A and inspect its completed summary.
2. Run Experiment B and inspect its completed summary.
3. Upload both summary.json files (and anchor metrics.json files if requested)
   for combined analysis.
