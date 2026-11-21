# Open-World AI: Resource-Bounded Open-World Continual Learning

**An independent research project by Ankit Kumar**

Investigating how learning systems can adapt to changing environments while managing limited annotation, memory, and computational resources.

---

## 1. Motivation

This research began with a simple question:

**Can an AI system continue learning when it has limited memory?**

That initial curiosity led to a broader research problem: AI systems are often trained on a limited set of data, yet they may encounter changing environments and unfamiliar concepts after deployment.

A continually learning system must not only learn from new observations, but also decide what is worth learning, what knowledge should be retained, and how to use limited resources over time.

This project investigates that problem through the lens of resource-bounded open-world continual learning.

## 2. Research Problem

The central research question is:

**How can a learning system coordinate limited annotation, memory, and computational resources while learning from a changing data stream?**

The project treats these as interconnected sequential decisions:

- **Annotation:** Which incoming observations should receive labels?
- **Memory:** Which previously observed examples should be retained, replaced, or discarded?
- **Computation:** How much computational effort should be allocated to learning at each stage?

The broader objective is to investigate whether coordinating these decisions through a state-dependent policy can improve long-term learning and resource-use trade-offs compared with fixed or independently designed strategies.

## 3. Long-Term Research Objectives

The intended system should investigate the ability to:

1. Recognize what it currently knows.
2. Detect observations that may fall outside its known knowledge.
3. Estimate which unfamiliar observations are worth learning.
4. Learn new concepts incrementally.
5. Retain previously acquired knowledge.
6. Detect when previous knowledge becomes unreliable.
7. Quantify and communicate uncertainty.
8. Operate under limited annotation, memory, and computational budgets.

These are research objectives, not claims that all capabilities have already been implemented or validated.

## 4. Open Research Questions — To Investigate

**Q1. Learning with Limited Labels**

Can a learning system acquire useful new knowledge from very few labelled examples?

**Q2. Adaptation to Changing Environments**

Can it adapt reliably when the environment or data distribution changes over time?

**Q3. Knowledge Retention**

Can it learn genuinely new classes while retaining previously acquired knowledge and minimizing catastrophic forgetting?

**Q4. Learning-Value Estimation**

Can it determine which unfamiliar observations are genuinely worth spending limited annotation, memory, and computational resources on?

**Q5. Reliability Under Uncertainty**

Can it respond appropriately when its confidence is incorrect or poorly calibrated?

## 5. Research Formulation and Approach

The problem is being developed as a resource-constrained sequential decision process.

The formulation considers:

- A state representation describing the current observation, prediction-related information, uncertainty indicators, and remaining resources.
- A joint action involving annotation, memory management, and computation allocation.
- Feasibility constraints imposed by the available budgets and memory capacity.
- A reward or utility formulation intended to account for learning performance and resource expenditure.

An initial feasibility-masked Deep Q-Network (DQN) implementation has been developed as a pilot for investigating sequential resource-allocation decisions.

The effectiveness of adaptive joint-resource allocation remains to be established experimentally.

## 6. Dataset and Experimental Setting

The preliminary experimental work uses the **UCI Gas Sensor Array Drift dataset**.

The dataset contains a sequential stream with 10 batches, 6 known classes, and 10,015 observations.

Current experimental constraints include:

- Limited annotation budget.
- Persistent memory capacity of 10 records in the pilot setup.
- Explicit computational budget constraints.
- Sequential evaluation using held-out data.
- Multiple random seeds for experimental consistency.

The current evaluation covers six known classes. Genuine unknown-class discovery has not yet been established in these experiments.

## 7. Experimental Progress

### E5-A — Progress-State Pilot

An experiment investigated adding an explicit stream-progress feature to the controller state.

The implementation and semantic checks passed, but the full seed-42 comparison produced identical action sequences and identical task-learner evaluation metrics between the baseline and progress-state conditions.

This did not establish an improvement from adding the progress feature.

### E5-B — Fixed Compute-Budget Calibration

A 10-seed calibration study compared three fixed compute-allocation conditions:

- 4 task-learner updates.
- 40 task-learner updates.
- 100 task-learner updates.

The study used the same seed-specific data split and initial task-model setup across the three conditions within each seed.

**Aggregate results across 10 seeds:**

| Fixed update condition | Mean accuracy | Mean macro-F1 |
|---|---:|---:|
| 4 updates | 0.220431 | 0.118591 |
| 40 updates | 0.310880 | 0.192687 |
| 100 updates | 0.382011 | 0.273306 |

The results showed increasing held-out performance across the tested fixed compute-dose and pacing conditions.

**Interpretation and limitations:**

- E5-B evaluated a fixed reference scheduler, not an adaptive DQN policy.
- The number of updates and their temporal distribution varied together.
- Therefore, this study does not isolate compute quantity from update timing.
- The results do not establish the benefit of joint adaptive resource allocation.
- The results are specific to the tested dataset, setup, and evaluation protocol.

The experiment is considered a calibration study, not proof of the central research hypothesis.

### Current Status

The work has progressed through initial mathematical formulation, feasibility-masked DQN pilot implementation, and fixed compute-budget calibration.

A comprehensive comparison of adaptive joint allocation against relevant baselines remains future work.

## 8. Current Limitations

The following limitations remain important:

- Genuine unknown-class discovery has not yet been evaluated.
- Reliable open-set recognition is not yet established.
- The effectiveness of an adaptive joint-resource allocation policy remains unverified.
- Catastrophic forgetting requires more comprehensive evaluation.
- Uncertainty calibration requires dedicated evaluation.
- Compute credits are provisional experimental proxies, not hardware-independent FLOPs, energy, or wall-clock measurements.
- Memory usage is currently measured in retained records rather than a complete byte-level or RAM-cost model.
- The current dataset contains six known classes in the reported evaluation.

These limitations define important parts of the ongoing research agenda.

## 9. Planned Research Directions

The next phase is intended to investigate:

1. Comparisons between fixed, random feasible, independent, and joint adaptive resource-allocation policies.
2. Multi-seed experimental evaluation and statistical analysis.
3. More comprehensive forgetting and knowledge-retention measurements.
4. Ablation studies to understand the role of individual resources.
5. Genuine unknown-class discovery and open-world evaluation.
6. Better uncertainty and reliability assessment.
7. More explicit and realistic resource-cost measurements.

These are planned investigations, not completed results.

## 10. Reproducibility and Project Materials

This repository is being organized as the public research home for the project.

Code, experiment configurations, analysis scripts, result artifacts, and documentation will be added after review and verification.

Until those materials are published, this README should be treated as a research overview rather than a complete reproducibility package.

## 11. Research Interests

Continual Learning · Open-World Learning · Active Learning · Reinforcement Learning · Uncertainty Estimation · Resource-Aware AI

---

**Researcher:** Ankit Kumar  
**Project:** Open-World AI  
**Status:** Ongoing independent research
