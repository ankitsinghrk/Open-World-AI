# E4-MEMORY-CAPACITY-01: Fixed-Ratio Reservoir Memory-Capacity Sweep

## 1. Experiment identification
- Status: completed; completed runs: 40/40.
- Seeds: `[42, 43, 44, 45, 46, 47, 48, 49, 50, 51]`; capacity arms: `[10, 20, 40, 80]` stored labelled records.
- Output directory: `.\experiments\results\e4_memory_capacity_sweep_seeds42-51`.

## 2. Research objective
Estimate how fixed reservoir capacity affects final predictive performance and forgetting when label selections, data splits, initialization, replay ratio, optimizer updates, and total example presentations are held constant.

## 3. Hypotheses
- H1: Increasing reservoir capacity may improve retention by broadening the set of historical labelled examples available for replay.
- H2: Capacity effects may show diminishing returns or non-monotonic behaviour; monotonic improvement is not assumed.
- H3: Current-batch adaptation may vary as the composition of the replay reservoir changes.

## 4. Experimental setup
- Dataset: UCI Gas Sensor Array Drift; 10 coarse chronological batches; 128 features; six shared classes.
- Seeds: `[42, 43, 44, 45, 46, 47, 48, 49, 50, 51]`; capacities: `[10, 20, 40, 80]` examples per reservoir.
- Each arm uses 100 labels (10 per batch), 240 optimizer updates, and 2,400 example presentations per seed.
- After Batch 1 warm-up, every update has five current examples and five replay examples; replay is sampled with replacement from a standard uniform reservoir.
- Same E2 train/test partitions, selected labels, Batch-1-only frozen scaler, model, optimizer, and initialization seeds are reused.
- E2 online-no-replay is the zero-memory reference; capacity 20 is repeated as an E3 replication check.

## 5. Methodology
For each seed and capacity, process batches 1-10 in order. Train with 10 current examples per update during Batch 1 because no prior memory exists. After training each batch, insert its 10 labelled examples into a uniform reservoir of the assigned capacity. From Batch 2 onward, use five examples from the current labelled batch and five samples with replacement from the prior reservoir per update. The total update and presentation budgets therefore remain constant across capacities. Evaluate all seen held-out batches after each phase and calculate peak-to-final forgetting on Batches 1-9.

## 6. Results
| Capacity | Accuracy mean (95% CI) | Macro-F1 mean (95% CI) | Forgetting mean (95% CI) |
|---:|---:|---:|---:|
| 10 | 0.6192 [0.5881, 0.6503] | 0.5437 [0.5093, 0.5781] | 0.0847 [0.0619, 0.1076] |
| 20 | 0.6384 [0.5815, 0.6953] | 0.5825 [0.5200, 0.6449] | 0.0776 [0.0517, 0.1036] |
| 40 | 0.6604 [0.5929, 0.7280] | 0.6157 [0.5378, 0.6936] | 0.0652 [0.0280, 0.1023] |
| 80 | 0.6722 [0.6234, 0.7211] | 0.6254 [0.5698, 0.6811] | 0.0507 [0.0256, 0.0759] |

### 6.1 Paired comparison against capacity 20
| Capacity | Accuracy difference (95% CI) | Macro-F1 difference (95% CI) | Forgetting difference (95% CI) |
|---:|---:|---:|---:|
| 10 | -0.0192 [-0.0575, 0.0192] | -0.0388 [-0.1023, 0.0247] | 0.0071 [-0.0227, 0.0369] |
| 40 | 0.0220 [-0.0209, 0.0650] | 0.0332 [-0.0361, 0.1025] | -0.0125 [-0.0429, 0.0180] |
| 80 | 0.0339 [0.0030, 0.0647] | 0.0430 [-0.0020, 0.0880] | -0.0269 [-0.0508, -0.0030] |

## 7. Statistical analysis
For each capacity, report means, sample standard deviations, and two-sided 95% Student-t confidence intervals across ten seeds (df=9). Capacity comparisons are paired by seed against the capacity-20 arm. These are exploratory multiple comparisons; no multiplicity adjustment is applied. Confidence intervals are conditional on this dataset and protocol, not population-wide guarantees.

## 8. Observations and interpretation
Interpret final pooled accuracy, Macro-F1, forgetting, and current-batch trajectory jointly. A larger record capacity does not automatically imply better results: reservoir composition, class distribution, drift, and the fixed replay ratio may interact. Memory capacity is counted in stored records, not bytes or measured RAM.

## 9. Limitations â€” what this experiment does not establish
- One dataset with coarse, non-uniform chronological batches is used.
- This is temporal/domain drift over six known classes, not class-incremental or open-set novel-class discovery.
- Only capacities in the requested sweep and one 5:5 replay ratio are tested.
- Uniform reservoir replay is not the project's future-utility memory selection policy.
- Capacity counts labelled records, not serialized bytes, RAM overhead, or energy use.
- Equal update and presentation counts are a compute proxy, not exact FLOPs or wall-clock parity.
- Ten seeds provide limited precision; further datasets and independent replications are required.
- Multiple paired capacity comparisons are exploratory and are not multiplicity-adjusted.

## 10. Conclusion
The capacity effect is supported only to the extent shown by the completed paired seed results. The capacity-20 arm must reproduce E3 within the strict numerical and memory-trace checks before the sweep is considered audited. No result from this experiment alone validates the complete Open-World AI agent or its utility-based memory policy.

## 11. Reproducibility record
Each capacity/seed folder contains `metrics.json`, `metrics.csv`, `selected_training_rows.csv`, `memory_trace.csv`, and `reservoir_replay_model_and_memory.pt`. Top-level artifacts are `summary.json`, `summary.csv`, `capacity_summary.csv`, `paired_comparisons_vs_capacity20.csv`, `stage_summary.csv`, and this report. Script, helper, reference-summary, and dataset SHA-256 hashes are recorded.

## 12. Next experiment
After identifying a useful capacity range, hold capacity fixed and test replay-ratio allocation, then compare uniform reservoir replay with the project's memory-utility ranking under identical budgets. Any utility-based policy must be tested as a distinct treatment, not inferred from this random-replay sweep.

Integrity note: capacity-20 replication checks passed for 10/10 seeds. Each capacity/seed run is accepted only after label, split, initialization, update, presentation, and memory-capacity checks pass.
