# Sharded Dataflow Graph

---
id: sharded-dataflow-graph
paper: pathways-2022
section: "§2 (Design Motivation), §4.3 (Coordination implementation)"
tags:
  - dataflow
  - sharding
  - scalability
  - representation
related:
  - asynchronous_distributed_dataflow
  - plaque_coordination
  - single_controller_model
---

## Definition

A **sharded dataflow graph** is a compact representation of a distributed computation where a single node represents a computation across **all of its shards**, rather than materializing one node per shard. This prevents the graph from growing quadratically as the number of shards increases.

## The Problem It Solves

In a naive dataflow representation:
- An edge between an **M-way sharded** computation and an **N-way sharded** computation requires **M + N nodes** and **M × N edges**.
- At thousands of shards, this produces **millions of graph edges**, causing substantial overhead in graph serialization and execution.

TensorFlow v1 suffered from exactly this problem.

## Pathways Solution

In Pathways, a chained execution of two computations A and B with N shards each has only **4 nodes** in the dataflow representation:

```
Arg → Compute(A) → Compute(B) → Result
```

This is true **regardless of N**. Individual data tuples are tagged with destination shard indices and flow between nodes during execution.

## Key Properties

- **Compact IR**: The MLIR-based intermediate representation stays small regardless of shard count.
- **Shard-tagged tuples**: The PLAQUE runtime routes data tuples by shard tag rather than by explicit edges.
- **Progressive lowering**: The device-agnostic IR is lowered through compiler passes to include physical locations, making re-lowering cheap if device mappings change.

## Impact on Scalability

| Approach | Nodes for A→B (N shards) | Edges |
|----------|--------------------------|-------|
| Naive (TF v1) | 2N | N² |
| Sharded (Pathways) | 4 | 3 (logical) |

This difference is critical for scaling to **thousands of shards** without overwhelming the coordination system.

## Paper Reference

> "A naive dataflow graph representing an edge between an M-way sharded computation and an N-way sharded computation would require M + N nodes and M × N edges, rapidly becoming unwieldy." — §2
