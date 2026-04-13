# Asynchronous Distributed Dataflow

---
id: asynchronous-distributed-dataflow
paper: pathways-2022
section: "§4 (System Architecture)"
tags:
  - architecture
  - dataflow
  - execution-model
related:
  - parallel_asynchronous_dispatch
  - sharded_dataflow_graph
  - plaque_coordination
---

## Definition

**Asynchronous Distributed Dataflow** is an execution model where computations are represented as a directed acyclic graph (DAG) of operators that consume and produce **futures** (or promises). Execution is triggered as soon as input futures are resolved, allowing the system to overlap computation, communication, and coordination.

## Key Characteristics

1. **Fire-and-forget dispatch**: The client enqueues multiple nodes of a computation graph without waiting for their results.
2. **Buffer futures**: Intermediate data remains on accelerators; hosts only exchange small "future" objects representing that data until the final result is needed.
3. **Pipelining**: Different parts of the graph run as soon as their specific dependencies are met.
4. **Dynamic routing**: Data can be sent to different downstream nodes based on runtime conditions (e.g., MoE experts).

## Pathways Implementation

Pathways implements this model using:
- **MLIR-based IR**: A custom dialect for expressing sharded computations.
- **PLAQUE**: The low-level substrate that manages the distributed execution of the dataflow graph.
- **Asynchronous RPCs**: Communication between the controller and workers uses an asynchronous messaging protocol.

## Advantages

- **Latent masking**: Overlaps the time spent sending commands over DCN with the time the accelerator spends executing previous kernels.
- **Host-compute overlap**: Allows the Python client to continue preparing future work while previous work is still enqueued or running.
- **Resource flexibility**: Since nodes are loosely coupled by futures, the system can choose where and when to run them more dynamically than a bulk-synchronous system.

## Paper Reference

> "PATHWAYS uses a sharded dataflow graph of asynchronous operators that consume and produce futures." — §1
