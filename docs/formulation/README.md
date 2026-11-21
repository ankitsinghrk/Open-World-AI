# Research Problem Formulation

This section documents the formal problem formulation for **Open-World AI: Resource-Bounded Open-World Continual Learning**.

## Core Research Problem

How can a learning system coordinate limited annotation, memory, and computational resources while learning from a changing data stream?

## Three Resource Decisions

1. **Annotation:** Decide which incoming observations should receive labels.
2. **Memory:** Decide which previously observed examples should be retained, replaced, or discarded.
3. **Computation:** Decide how much computational effort should be allocated to learning.

## Sequential Decision Perspective

The project formulates these resource decisions as interconnected sequential decisions under feasibility constraints.

The research investigates whether a state-dependent policy coordinating all three decisions can improve long-term learning and resource-use trade-offs.

## Mathematical Specification

The current V1.0 working specification is available here:

[**Chapter 11 – Mathematical Specification (PDF)**](Open_World_AI_Chapter_11_Mathematical_Specification_Updated.pdf)

It covers the state representation, joint action space, transition dynamics, reward and objective, resource constraints, feasibility-masked DQN formulation, optimization procedure, and evaluation criteria.

This is a formal design specification; it does not by itself establish the effectiveness of the adaptive policy.

## Current Status

The initial formulation and a feasibility-masked DQN pilot have been developed. The effectiveness of adaptive joint-resource allocation has not yet been established.

See the [main project overview](../../README.md) for experimental progress and current limitations.
