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

## Formal Specification

The detailed mathematical specification, including state representation, action space, transition assumptions, feasibility constraints, and objective function, will be published here after final review.

## Current Status

The initial formulation and a feasibility-masked DQN pilot have been developed. The effectiveness of adaptive joint-resource allocation has not yet been established.

See the [main project overview](../../README.md) for experimental progress and current limitations.
