# Asynchronous Distributed Dataflow

---
id: asynchronous-distributed-dataflow
paper: pathways-2022
section: "§4 (System Architecture)"
tags:
  - architecture
  - dataflow
  - execution-model
  - asynchronous
related:
  - sharded_dataflow_graph
  - parallel_asynchronous_dispatch
  - plaque_coordination
---

## Definition

Asynchronous distributed dataflow is the execution model at the core of Pathways. Programs are expressed as **directed acyclic graphs (DAGs)** of operators that **consume and produce futures**—opaque handles to values that will be computed in the future. This allows the control plane to proceed with scheduling and resource allocation before data-plane values are available.

## How It Works in Pathways

1. **Futures as edges**: Each edge in the dataflow graph carries a future. Nodes do not block waiting for predecessors to complete before the system begins scheduling successor nodes.
2. **Decoupled control and data planes**: The control plane (scheduling, resource allocation, coordination) executes **in parallel** across nodes, even when the data plane has true dependencies. This is possible because resource requirements of compiled functions are **statically known**.
3. **Sharded representation**: The dataflow graph uses a compact sharded representation where a single node represents a computation across N shards, avoiding the M×N edge explosion of naively materialized graphs.

## Contrast with Prior Systems

| System | Model | Limitation |
|--------|-------|------------|
| **TF v1** | Distributed dataflow, but sequential coordination | Cross-host dispatch latency accumulates for pipelines |
| **JAX/PyTorch** | Multi-controller, SPMD-only | No centralized coordination; data movement limited to collectives |
| **Pathways** | Async distributed dataflow with parallel dispatch | Control and data planes decoupled; scales to thousands of shards |

## Key Properties

- **Non-blocking scheduling**: Successor computations can have resources allocated and be partially prepared before predecessors finish.
- **Efficient pipelining**: Enables pipeline-parallel execution across many stages without accumulated dispatch latency.
- **Sparse data exchanges**: Supports dynamically chosen subsets of shard-to-shard communication.

## Paper Reference

> "PATHWAYS uses a sharded dataflow graph of asynchronous operators that consume and produce futures, and efficiently gang-schedules heterogeneous parallel computations on thousands of accelerators." — Abstract
