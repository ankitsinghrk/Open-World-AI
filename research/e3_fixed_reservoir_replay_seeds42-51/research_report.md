# E3-RESERVOIR-01: Fixed-Capacity Reservoir Replay Baseline

## 1. Experiment identification
- Status: completed; completed seeds: 10.
- Output directory: `.\experiments\results\e3_fixed_reservoir_replay_seeds42-51`.

## 2. Research objective
Test whether a small fixed-size random reservoir of previously labelled examples reduces forgetting relative to E2 online fine-tuning without replay, under matched label selections, train/test splits, initialization seeds, optimizer updates, and total example presentations.

## 3. Hypotheses
- H1: Reservoir replay may reduce peak-to-final forgetting on earlier batches.
- H2: Replay may improve final pooled Macro-F1 and/or accuracy versus E2 no-replay.
- H3: Any retention benefit may come with a trade-off in current-batch adaptation because half of each post-warm-up minibatch is replay data.

## 4. Experimental setup
- Dataset: UCI Gas Sensor Array Drift; seeds: `[42, 43, 44, 45, 46, 47, 48, 49, 50, 51]`.
- Same deterministic 80/20 class-wise split, Batch 1 training-pool scaler, model, initialization seeds, and selected 10 labelled rows per batch as the E2 online baseline.
- Label budget: 100 total (10 per batch); reservoir capacity: 20 labelled examples.
- Replay: standard uniform reservoir sampling; after Batch 1 warm-up, each update uses 5 current + 5 replay examples.
- Batch 1 warm-up: 10 current examples per update because no past examples exist yet.
- Training: 24 updates per batch, 240 total updates, 2400 total example presentations; Adam, learning rate 0.001, batch size 10, CPU.
- No validation-based checkpoint selection; held-out test data are evaluation only.

## 5. Methodology
For each seed, reproduce E2's per-batch train/test partitions and exact online labelled-row selections. Initialize the same MLP with the same seed. Process batches chronologically. During Batch 1 warm-up, each of its 10 labels appears in all 24 updates because no past memory exists. After each batch has been trained, insert its labelled examples into a capacity-20 uniform reservoir. From Batch 2 onward, each optimizer update combines five current-batch examples with five examples sampled with replacement from the prior reservoir. The current-example schedule ensures every one of the ten current labels appears exactly 12 times in each later batch's 24 updates. Evaluate all seen held-out batches after each phase and calculate peak-to-final forgetting on Batches 1-9.

## 6. Results
### 6.1 Final pooled performance and paired replay-minus-no-replay differences
| Outcome | Reservoir replay mean | SD | 95% CI | Paired difference vs E2 | Paired 95% CI |
|---|---:|---:|---:|---:|---:|
| Accuracy | 0.6384 | 0.0795 | [0.5815, 0.6953] | 0.0940 | [0.0714, 0.1166] |
| Macro-F1 | 0.5825 | 0.0873 | [0.5200, 0.6449] | 0.1155 | [0.0806, 0.1504] |
| Forgetting (Batches 1-9) | 0.0776 | 0.0362 | [0.0517, 0.1036] | -0.0518 | [-0.0812, -0.0224] |

### 6.2 Current-batch accuracy by stream position
| Batch | Reservoir current-batch accuracy | Reservoir current-batch Macro-F1 |
|---:|---:|---:|
| 1 | 0.4456 | 0.3339 |
| 2 | 0.7004 | 0.4129 |
| 3 | 0.7123 | 0.4920 |
| 4 | 0.7906 | 0.6155 |
| 5 | 0.8575 | 0.6588 |
| 6 | 0.6950 | 0.5042 |
| 7 | 0.5914 | 0.5527 |
| 8 | 0.5650 | 0.3397 |
| 9 | 0.4489 | 0.3356 |
| 10 | 0.4851 | 0.4645 |

## 7. Statistical analysis
Means, sample standard deviations, and two-sided 95% Student-t confidence intervals are calculated across the ten paired seeds (df=9). Replay-minus-no-replay differences are paired by seed because E3 reuses E2's exact train/test partitions, selected labelled rows, and initialization seeds. Intervals describe variability under this seed protocol and are not evidence of generalization to other datasets or deployment environments.

## 8. Observations and interpretation
Interpret accuracy, Macro-F1, forgetting, and current-batch performance jointly. Lower forgetting alone is not sufficient if overall predictive performance collapses. This arm is a simple random-replay control, not a learned or utility-based memory policy.

## 9. Limitations â€” what this experiment does not establish
- The dataset batches are coarse chronological groups, not equal-duration consecutive time windows.
- This is six-class temporal/domain drift, not class-incremental or open-set novel-class discovery.
- The scaler uses unlabeled Batch 1 training-pool features, as in E2.
- A capacity-20 uniform reservoir is only one memory size and one policy; no capacity sweep or memory utility scoring is tested.
- The 5-current/5-replay ratio is fixed after Batch 1 and may not be optimal.
- Equal update and example-presentation counts are a compute proxy, not exact FLOPs, energy, or wall-clock parity.
- No active label acquisition, uncertainty estimation, compute-tier allocation, or complete Open-World AI policy is evaluated.

## 10. Conclusion
Reservoir replay versus E2 no-replay: paired accuracy difference 0.0940 (95% CI [0.0714, 0.1166]); paired Macro-F1 difference 0.1155 (95% CI [0.0806, 0.1504]); paired forgetting difference -0.0518 (95% CI [-0.0812, -0.0224]). Interpret support only after comparing these outcomes and current-batch trajectory; this does not validate the Open-World AI policy.

## 11. Reproducibility record
Each seed folder contains `metrics.json`, `metrics.csv`, `selected_training_rows.csv`, `memory_trace.csv`, and `reservoir_replay_model_and_memory.pt`. The top-level directory contains `summary.json`, `summary.csv`, `stage_summary.csv`, and this report. Each run records source and dataset SHA-256 hashes, environment versions, selected rows, memory history, and training exposure counts.

## 12. Next experiment
If reservoir replay reduces forgetting, test memory capacity and replay ratio under the same label and training-exposure budgets. If results are mixed, inspect per-batch retention and class composition in the reservoir before testing utility-based selection. Only then evaluate the project's memory-utility pipeline and resource-allocation policy.

Integrity note: this report is generated only after every requested seed completes and every run passes update, presentation, capacity, and E2 row-selection consistency checks.
