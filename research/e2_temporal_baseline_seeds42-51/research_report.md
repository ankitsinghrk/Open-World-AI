# E2-TEMP-01: Temporal Continual-Learning Baseline

## 1. Experiment identification
- Experiment ID: `E2-TEMP-01`
- Status: completed; this report is produced only after all requested seeds succeed.
- Output directory: `.\experiments\results\e2_temporal_baseline_seeds42-51`

## 2. Research objective
Compare a frozen static model with a no-replay online fine-tuning baseline on the chronological Gas Sensor Array Drift batch stream. Measure current-batch adaptation, final stream performance, and loss of previously acquired batch performance.

## 3. Hypotheses
- H1: Online no-replay updates may improve performance on newly arriving batches relative to a frozen model.
- H2: Updating only on the current batch may reduce performance on earlier batches (catastrophic-forgetting signal).
- H3: Equal label and training-exposure budgets may produce a trade-off between adaptation to recent batches and retention of earlier batches.

## 4. Experimental setup
- Dataset: UCI Gas Sensor Array Drift; raw batch sizes: `[445, 1244, 1586, 161, 197, 2300, 3613, 294, 470, 3600]`; total rows: 13,910.
- Seeds: `[42, 43, 44, 45, 46, 47, 48, 49, 50, 51]` (10 paired runs).
- Per-batch split: deterministic class-wise 80/20 train-pool/held-out test split; same split used for both strategies within each seed.
- Label budget: 100 per strategy. Static receives 100 from Batch 1; online receives 10 from each of Batches 1â€“10.
- Scaler: feature-wise min-max to [-1, 1], fitted on unlabeled features in Batch 1's 80% train pool only, then frozen; held-out and future batches are not used to fit it.
- Model: 128 â†’ 64 â†’ ReLU â†’ 6 MLP; Adam; learning rate 0.001; cross-entropy; CPU; batch size 10.
- Compute/exposure normalization: both strategies receive exactly 240 optimizer updates and 2,400 example presentations.
- Static schedule: 24 epochs over its 100 labels (10 updates per epoch); then frozen.
- Online schedule: 24 epochs over each current batch's 10 labels (24 updates per batch); no replay or access to prior labelled examples during later updates.
- No validation-based selection or checkpoint selection; final scheduled models evaluated.

## 5. Methodology
For each seed, each of the ten raw batches is split independently by class. A common scaler is fitted on Batch 1's training-pool features only. The static strategy samples 100 labels from Batch 1 and trains once. The online strategy samples 10 labels from each batch in chronological order and fine-tunes only on that batch's selected labels. After each online update, both models are evaluated on held-out test sets of all batches seen so far. At the end, both are evaluated on the concatenated held-out sets from all ten batches. Online forgetting for each of Batches 1â€“9 is defined as the maximum post-update accuracy seen after that batch's acquisition minus its final accuracy after Batch 10; Batch 10 is excluded because no later update can affect it.

## 6. Results
### 6.1 Across-seed final pooled held-out performance
| Strategy | Accuracy mean | SD | 95% CI | Macro-F1 mean | SD | 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| Static Batch 1 | 0.4239 | 0.0324 | [0.4008, 0.4471] | 0.3425 | 0.0267 | [0.3234, 0.3616] |
| Online no replay | 0.5443 | 0.0969 | [0.4750, 0.6137] | 0.4670 | 0.1059 | [0.3912, 0.5428] |

### 6.2 Paired online-minus-static differences
| Metric | Mean paired difference | SD | 95% CI |
|---|---:|---:|---:|
| Accuracy | 0.1204 | 0.0917 | [0.0548, 0.1860] |
| Macro-F1 | 0.1245 | 0.0968 | [0.0552, 0.1937] |

### 6.3 Mean current-batch post-update performance by stream position
| Batch | Static accuracy | Online accuracy | Static Macro-F1 | Online Macro-F1 |
|---:|---:|---:|---:|---:|
| 1 | 0.9067 | 0.4456 | 0.8549 | 0.3339 |
| 2 | 0.8386 | 0.7104 | 0.6677 | 0.3605 |
| 3 | 0.6710 | 0.6918 | 0.4626 | 0.4982 |
| 4 | 0.4969 | 0.7531 | 0.3020 | 0.4852 |
| 5 | 0.3800 | 0.7500 | 0.2803 | 0.5652 |
| 6 | 0.2863 | 0.6500 | 0.2061 | 0.4429 |
| 7 | 0.2830 | 0.5535 | 0.1885 | 0.4978 |
| 8 | 0.1867 | 0.5900 | 0.1665 | 0.3320 |
| 9 | 0.2936 | 0.4160 | 0.1781 | 0.2873 |
| 10 | 0.3769 | 0.4319 | 0.2967 | 0.3811 |

- Online mean forgetting across Batches 1â€“9: 0.1295 (SD 0.0613; 95% CI [0.0856, 0.1733]).
- Mean current-batch accuracy across ten positions: static 0.4720; online 0.5992.

## 7. Statistical analysis
The ten seeds are paired: each static and online comparison uses the same seed-specific train/test splits and matching model-initialization seed; the two strategies use separate, reproducible label-selection RNG streams. Means, sample standard deviations, and two-sided 95% Student-t confidence intervals are computed across seeds (df=9 for the default ten-seed protocol). Paired intervals are reported for online-minus-static differences. These intervals quantify variation across the specified random-seed procedure; they do not establish causal effects across other datasets or environments. No unpaired significance test or claim of statistical significance is made.

## 8. Observations and interpretation
Interpret the current-batch trajectory together with final pooled performance and the forgetting metric. A higher current-batch score alone does not show better continual learning if prior-batch performance falls. The frozen static model is a no-adaptation reference; online no-replay is a simple fine-tuning reference, not a memory-aware or resource-aware policy.

## 9. Limitations â€” what this experiment does not establish
- The ten batch files are coarse chronological groups, not equal-duration or strictly consecutive time windows.
- The task is six-class classification under temporal/domain drift; it is not class-incremental or open-set novel-class discovery.
- The frozen scaler uses unlabeled features from the full Batch 1 training pool. This is an unsupervised initial-batch preprocessing assumption; it is not a purely label-only protocol.
- The online baseline has no replay, memory selection, uncertainty estimation, active label acquisition, compute-tier allocation, or Open-World AI policy.
- The online learner receives 10 labels at every batch, including batches where some classes may be absent or sparse; selected class counts are recorded per run.
- The static baseline is intentionally front-loaded with all 100 labels from Batch 1, whereas online labels arrive progressively; this is the intended timing comparison, not identical information availability at every time point.
- Equal optimizer updates and example presentations control a simple compute proxy, not exact wall-clock energy or hardware FLOPs.
- Held-out samples are used only for evaluation; the results remain specific to this dataset, model, split rule, seed set, and training schedule.

## 10. Conclusion
Across the 10 paired seeds, online no-replay final pooled accuracy was higher than static by a mean of 0.1204 (95% CI [0.0548, 0.1860]); the paired Macro-F1 difference was 0.1245 (95% CI [0.0552, 0.1937]). The online mean forgetting estimate for Batches 1â€“9 was 0.1295 (95% CI [0.0856, 0.1733]). These are protocol-specific baseline findings, not evidence that the Open-World AI policy works.
This experiment does not validate the full Open-World AI architecture or its research novelty.

## 11. Reproducibility record
- Python: 3.14.10; NumPy: 2.5.4; PyTorch: 2.15.0+cpu.
- Platform: `Windows-11-10.0.26200-SP0`.
- Runner SHA-256: `e9625e425847daa4f8ec164b7149e592a7a160222e32ddda087b0c14744b615b`.
- Static-baseline helper SHA-256: `cea3a7f2754f84d64893fb7b50766f7c09a7471a191a79bcfaf875c0b94a46e3`.
- Data directory: `.\data\raw\gas_sensor_array_drift\Dataset`.
- Each seed directory contains `metrics.json`, `metrics.csv`, `selected_training_rows.csv`, `static_model_and_scaler.pt`, and `online_model_and_scaler.pt`.
- Combined files: `summary.json`, `summary.csv`, `stage_summary.csv`, and `research_report.md`.

## 12. Next experiment
If online no-replay exhibits forgetting, use this E2 result as the reference point for a controlled replay/memory baseline (for example, fixed-size reservoir replay) under the same label and compute budgets. Only after that should the project test utility-based memory selection or the learned resource-allocation policy. If online no-replay does not show clear adaptation or forgetting, first inspect per-batch metrics and verify the stream/split protocol before adding policy complexity.

---
Integrity note: all outcome values in this report are computed from per-seed saved run records. No metrics are fabricated or filled in for unsuccessful runs.
