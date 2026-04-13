# Gang Scheduling

---
id: gang-scheduling
paper: pathways-2022
section: "§2 (Design Motivation), §4.4 (Gang-scheduled dynamic dispatch)"
tags:
  - scheduling
  - coordination
  - SPMD
  - multi-tenancy
related:
  - single_controller_model
  - parallel_asynchronous_dispatch
  - resource_manager
---

## Definition

**Gang scheduling** is the practice of ensuring that all shards of a communicating SPMD computation are enqueued for execution at the same time, in a **consistent order** across all devices.

## Why It Is Required

- **TPU necessity**: TPUs are single-threaded and run non-preemptible kernels. If communicating computations are not enqueued consistently, the system **deadlocks**.
- **GPU benefit**: Even for GPUs (which support concurrent execution), gang scheduling enables more **efficient collective execution** by ensuring all participants enter collectives simultaneously.

## Pathways Implementation

1. **Centralized per-island scheduler**: Each island has a single scheduler that determines execution order for all computations in that island.
2. **FIFO ordering**: The current implementation enqueues work in first-in-first-out order.
3. **Single-message subgraph submission**: When a subgraph can be scheduled statically, the system sends a single message describing the entire subgraph to the scheduler, minimizing network traffic.
4. **Interleaving**: Computations from different programs can still be interleaved between gang-scheduled batches.

## Multi-Tenancy

Gang scheduling enables **multi-tenancy** — multiple clients' programs are interleaved on the same accelerators with:
- **Zero context-switch overhead** (when resources fit in HBM).
- **Proportional-share fairness** (e.g., 1:2:4:8 ratios between clients).
- **Millisecond-scale** or finer interleaving.

## Performance Evidence

- With 4 independent clients, Pathways achieves **at least the same aggregate throughput** as dedicated JAX instances.
- Maximum throughput of Pathways **exceeds** JAX for very small computations because workers accept more remote computations than JAX can dispatch locally via Python.

## Paper Reference

> "Gang-scheduling is essential in the case of TPUs, since they are single-threaded and only run non-preemptible kernels, so the system will deadlock if communicating computations are not enqueued in a consistent order." — §2
