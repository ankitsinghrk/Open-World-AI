# Reproducing Experiments

This guide records the current local workflow for the published Open-World AI experiment scripts. It is not yet a fully automated reproduction package: the dataset and generated checkpoints are not stored in this repository.

## 1. Dataset

The experiment loaders expect the original **UCI Gas Sensor Array Drift Dataset** in LIBSVM-style batch files. The E2 data audit recorded these ten batch sizes:

```text
[445, 1244, 1586, 161, 197, 2300, 3613, 294, 470, 3600]
Total: 13,910 observations
```

Download the original dataset from the [UCI Machine Learning Repository — Gas Sensor Array Drift Dataset (ID 224)](https://archive.ics.uci.edu/dataset/224/gas%2Bsensor%2Barray%2Bdrift%2Bdataset). Extract the ten files into this exact directory:

```text
data/
└── raw/
    └── gas_sensor_array_drift/
        └── Dataset/
            ├── batch1.dat
            ├── batch2.dat
            ├── batch3.dat
            ├── batch4.dat
            ├── batch5.dat
            ├── batch6.dat
            ├── batch7.dat
            ├── batch8.dat
            ├── batch9.dat
            └── batch10.dat
```

Use the original ID 224 batch format expected by the loader: an integer class label followed by 128 `index:value` feature entries on each row. Do not substitute a differently formatted concentration-augmented export unless the loader has first been adapted and tested for that format.

The dataset is not bundled with this repository. The UCI record describes 13,910 measurements, 128 features, and six gas classes. Cite the dataset when reporting results:

> Vergara, A. (2012). *Gas Sensor Array Drift Dataset*. UCI Machine Learning Repository. https://doi.org/10.24432/C5RP6W.

## 2. Python environment

The versions used in the recorded local validation were Python 3.14.10, NumPy 2.5.4, PyTorch 2.15.0+cpu, and pytest 9.2.0.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

The scripts are intended to be run from the repository root.

## 3. E1 — Seed sensitivity

The crossed 5 × 5 sampling/training-seed design uses five sampling seeds and five training seeds (25 combinations):

```powershell
python .\experiments\open_world_e1_crossed_5x5.py
```

The separate randomness-decomposition script has two modes:

```powershell
python .\experiments\open_world_e1_randomness_decomposition.py --mode selection
python .\experiments\open_world_e1_randomness_decomposition.py --mode training
```

The scripts use the ten batch files from the dataset directory above and write generated run artifacts under `experiments/results/` by default.

## 4. E2 → E3 → E4 — Temporal and replay baselines

Run these in order. The defaults use seeds 42–51.

**E2: static versus online no-replay baseline**

```powershell
python .\experiments\open_world_e2_temporal_baseline.py
```

**E3: fixed-capacity uniform reservoir replay**

```powershell
python .\experiments\open_world_e3_fixed_reservoir_replay.py
```

E3 reads the completed E2 reference results from:

```text
experiments/results/e2_temporal_baseline_seeds42-51/
```

**E4: reservoir memory-capacity sweep**

```powershell
python .\experiments\open_world_e4_memory_capacity_sweep.py
```

E4 requires both the completed E2 and E3 reference result directories. Its default capacities are 10, 20, 40, and 80 labelled records.

These runners refuse to overwrite non-empty output directories. For a fresh run, move or rename a previous output directory, or supply a new output path through the script's command-line options.

## 5. E5-B — Fixed compute-budget calibration

The script requires one of the supported compute-credit conditions: 4, 40, or 100. A bounded smoke run uses the default positive `--max-steps`; setting it to zero requests the full stream.

Example full-stream run for one condition and seed:

```powershell
python .\experiments\open_world_budget_calibration_e5b.py --compute-credits 4 --seed 42 --max-steps 0
```

Repeat with the other compute-credit values and predeclared seeds for the intended study. E5-B is a fixed-schedule calibration experiment; it is not a test proving that the adaptive DQN jointly allocates all three resources effectively.

## 6. DQN pilot — checkpoint prerequisite

The V2 DQN pilot expects a prediction checkpoint at:

```text
experiments/results/static_baseline_100_seed42/model_and_scaler.pt
```

Generate that checkpoint first using the static baseline:

```powershell
python .\experiments\open_world_static_100_baseline_v2.py --seed 42
```

Then run the pilot:

```powershell
python .\experiments\open_world_causal_dqn_pilot_v2.py --seed 42 --max-steps 0
```

The pilot also requires the ten dataset batch files. The generated checkpoint is not committed to the repository. Keep the pilot's smoke-run and full-stream configurations distinct when recording results.

## 7. Tests

Run the full test suite:

```powershell
python -m pytest -q
```

Two integration tests require the static-baseline checkpoint described above. In the recorded validation environment, where that checkpoint was not present, those two tests were deselected and the remaining suite reported:

```text
277 passed, 2 deselected, 59 subtests passed
```

The exact command used for that validation was:

```powershell
python -m pytest -q --deselect=tests/test_environment_prediction_checkpoint.py::TestRealPredictionStateIntegration::test_real_checkpoint_to_encoded_state --deselect=tests/test_real_prediction_dqn.py::TestRealPredictionDQN::test_real_checkpoint_to_dqn
```

This is a recorded local test result, not a guarantee for every platform or dependency combination. To exercise the two checkpoint-dependent tests, generate the required checkpoint first and rerun the full suite.

## 8. Generated files and version control

Generated experiment outputs and checkpoints belong under `experiments/results/` and should be reviewed before any intentional publication. This directory is excluded from Git by the repository's `.gitignore`; do not commit raw data, large checkpoints, or unreviewed result dumps by default.

The E1–E4 research reports already committed under `research/` are the curated reports. A clean-machine reproduction still requires the external dataset, the stated dependencies, and any prerequisite reference outputs or checkpoints for chained experiments.
