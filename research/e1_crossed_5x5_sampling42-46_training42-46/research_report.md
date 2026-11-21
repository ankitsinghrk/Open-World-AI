# E1-CROSS-01: E1 Crossed Randomness Experiment (5 Ã— 5)

## 1. Experiment identification
- Experiment ID: `E1-CROSS-01`
- Status: completed; all 25 cells executed successfully if this report exists
- Output directory: `.\experiments\results\e1_crossed_5x5_sampling42-46_training42-46`

## 2. Research objective
Quantify how the initial labelled-row sample, PyTorch training randomness, and their interaction relate to future-batch performance under the locked E1 static-baseline protocol.

## 3. Hypotheses
- H1: Changing the sampling seed changes selected Batch 1 rows, fitted scaler, and downstream performance.
- H2: Changing the training seed changes model initialization and minibatch order, and may change downstream performance while rows/scaler remain fixed.
- H3: The effect of training seed may depend on which labelled sample was selected (sampling Ã— training interaction).

## 4. Experimental setup
- Dataset: UCI Gas Sensor Array Drift, batches 1â€“10; 13,910 validated rows.
- Sampling seeds: `[42, 43, 44, 45, 46]`
- Training seeds: `[42, 43, 44, 45, 46]`
- Design: complete balanced 5 Ã— 5 factorial; one run per seed pair (25 runs).
- Label budget: 100; uniformly sampled without replacement from Batch 1.
- Model: 128 -> 64 -> ReLU -> 6 MLP (as defined by E1 baseline helper); CPU; Adam; learning rate 0.001; 60 epochs; batch size 32.
- Scaling: feature-wise minâ€“max to [-1, 1], fitted only on the selected 100 training rows; no future clipping.
- Future batches 2â€“10 are evaluation-only; no future updates, validation selection, or checkpoint selection.

## 5. Methodology
For each seed pair, NumPy's sampling generator selects 100 Batch 1 rows using the sampling seed. The existing E1 helper fits the scaler on these rows and invokes the fixed-schedule MLP trainer using the training seed. The final-epoch model is evaluated on unselected Batch 1 rows and batches 2â€“10. Pooled future accuracy and fixed-six-class macro-F1 are the primary outcomes.

## 6. Results
### 6.1 All 25 seed combinations
| Sampling seed | Training seed | Accuracy | Macro-F1 (6 classes) | Selected-row hash |
|---:|---:|---:|---:|---|
| 42 | 42 | 0.470628 | 0.346000 | `9b4a1b41f76eâ€¦` |
| 42 | 43 | 0.408318 | 0.305006 | `9b4a1b41f76eâ€¦` |
| 42 | 44 | 0.468177 | 0.314897 | `9b4a1b41f76eâ€¦` |
| 42 | 45 | 0.431489 | 0.318081 | `9b4a1b41f76eâ€¦` |
| 42 | 46 | 0.402228 | 0.274776 | `9b4a1b41f76eâ€¦` |
| 43 | 42 | 0.406535 | 0.335999 | `cbfe4cf494f0â€¦` |
| 43 | 43 | 0.398663 | 0.331250 | `cbfe4cf494f0â€¦` |
| 43 | 44 | 0.411957 | 0.335658 | `cbfe4cf494f0â€¦` |
| 43 | 45 | 0.432974 | 0.344186 | `cbfe4cf494f0â€¦` |
| 43 | 46 | 0.419235 | 0.340995 | `cbfe4cf494f0â€¦` |
| 44 | 42 | 0.411808 | 0.295993 | `73679be4d044â€¦` |
| 44 | 43 | 0.394430 | 0.277893 | `73679be4d044â€¦` |
| 44 | 44 | 0.395321 | 0.274471 | `73679be4d044â€¦` |
| 44 | 45 | 0.388489 | 0.266267 | `73679be4d044â€¦` |
| 44 | 46 | 0.395767 | 0.269411 | `73679be4d044â€¦` |
| 45 | 42 | 0.424211 | 0.344375 | `52017dfa947eâ€¦` |
| 45 | 43 | 0.448570 | 0.364333 | `52017dfa947eâ€¦` |
| 45 | 44 | 0.431192 | 0.343875 | `52017dfa947eâ€¦` |
| 45 | 45 | 0.459488 | 0.367727 | `52017dfa947eâ€¦` |
| 45 | 46 | 0.473821 | 0.383839 | `52017dfa947eâ€¦` |
| 46 | 42 | 0.542666 | 0.449008 | `ff2a3a656273â€¦` |
| 46 | 43 | 0.497066 | 0.399513 | `ff2a3a656273â€¦` |
| 46 | 44 | 0.522094 | 0.435944 | `ff2a3a656273â€¦` |
| 46 | 45 | 0.470999 | 0.391532 | `ff2a3a656273â€¦` |
| 46 | 46 | 0.540141 | 0.457116 | `ff2a3a656273â€¦` |

### 6.2 Overall outcome summaries
| Metric | Mean across 25 cells | Sample SD | Minimum | Maximum |
|---|---:|---:|---:|---:|
| Pooled future accuracy | 0.441851 | 0.046175 | 0.388489 | 0.542666 |
| Pooled future Macro-F1 | 0.342726 | 0.054542 | 0.266267 | 0.457116 |

### 6.3 Sampling-seed marginal means
| Sampling seed | Mean accuracy | Mean Macro-F1 |
|---:|---:|---:|
| 42 | 0.436168 | 0.311752 |
| 43 | 0.413873 | 0.337618 |
| 44 | 0.397163 | 0.276807 |
| 45 | 0.447456 | 0.360830 |
| 46 | 0.514593 | 0.426623 |

### 6.4 Training-seed marginal means
| Training seed | Mean accuracy | Mean Macro-F1 |
|---:|---:|---:|
| 42 | 0.451170 | 0.354275 |
| 43 | 0.429410 | 0.335599 |
| 44 | 0.445748 | 0.340969 |
| 45 | 0.436688 | 0.337559 |
| 46 | 0.446238 | 0.345227 |

## 7. Statistical analysis
### 7.1 Descriptive two-factor sums-of-squares decomposition
| Component | Accuracy SS share | Macro-F1 SS share |
|---|---:|---:|
| Sampling Seed | 79.49% | 88.92% |
| Training Seed | 2.96% | 1.54% |
| Sampling Training Interaction | 17.56% | 9.54% |

This is a descriptive decomposition over the specified 5 Ã— 5 seed grid, not a population-level variance-components estimate. There is one observation per cell, so the interaction term has no independent within-cell error estimate. No inferential F-tests, p-values, or statistical-significance claims are produced.

## 8. Observations and interpretation
Use the marginal means and interaction shares to describe the observed seed sensitivity in this exact grid. A larger share indicates more variation associated with that factor in this finite design; it does not prove that the factor dominates for other seed values, datasets, models, or environments.

## 9. Limitations â€” what this experiment does not establish
- This remains a static supervised baseline, not the Open-World AI agent or a continual-learning policy.
- It does not test online adaptation, label acquisition decisions, memory operations, compute allocation, open-set recognition, or forgetting control.
- The five seed values are a finite selected grid; results may not generalize to other seeds or datasets.
- One run per cell prevents an independent estimate of within-cell experimental error and formal significance testing of interaction.
- Batch chronology and class support are inherited from the dataset; pooled metrics can mask per-batch and class-specific behaviour. Inspect per-batch metrics in each run's `metrics.json`.

## 10. Conclusion
Conclusion is restricted to the measured 5 Ã— 5 seed grid. The report records observed descriptive differences only; it does not claim that either source of randomness is universally causal or that the research hypothesis for Open-World AI has been validated.

## 11. Reproducibility record
- Python: 3.14.10
- NumPy: 2.5.4
- PyTorch: 2.15.0+cpu
- Platform: Windows-11-10.0.26200-SP0
- Crossed-runner SHA-256: `4cf857bc302a99ad65bbfd2a0beb4368583cda0cb414f4ddd8cd2022111cb5a5`
- E1 randomness helper SHA-256: `191575c8ebdb19123256e9c93c5bdab9450b27edd39ee202bdfb0b95ac8e6929`
- Static baseline helper SHA-256: `cea3a7f2754f84d64893fb7b50766f7c09a7471a191a79bcfaf875c0b94a46e3`
- Dataset directory: `.\data\raw\gas_sensor_array_drift\Dataset`
- Each run folder contains `metrics.json`, `metrics.csv`, `selected_training_rows.csv`, and `model_and_scaler.pt`.
- Combined outputs: `crossed_results.csv`, `summary.json`, and this report.

## 12. Next experiment
After reviewing these results, consider a replicated crossed design with additional independent seed values and/or repeated execution of selected cells to estimate run-to-run reproducibility. Then proceed to a controlled continual-learning benchmark comparison; keep that separate from this static-seed audit.

---
**Integrity note:** This report is generated from the saved run metrics. No result is inserted unless its corresponding run completed and produced a metrics record.
