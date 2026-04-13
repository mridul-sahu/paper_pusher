# JAX / TensorFlow Comparison

---
id: jax-tf-comparison
paper: pathways-2022
section: "§2 (Design Motivation), §3 (Programming Model), §6 (Related Work)"
tags:
  - frameworks
  - comparison
  - programming-model
related:
  - single_controller_model
  - spmd_vs_mpmd
  - sharded_dataflow_graph
---

## Overview

Pathways is designed to bridge the gap between the performance of **JAX** (multi-controller) and the flexibility of **TensorFlow** (single-controller).

## Comparison Table

| Feature | JAX (Multi-Controller) | TF v1 (Single-Controller) | Pathways (Single-Controller) |
|---------|------------------------|---------------------------|------------------------------|
| **Control Plane** | Distributed (Every host runs Python) | Centralized (Coordinator) | Centralized (Coordinator) |
| **Flexibility** | Limited to SPMD | High (Arbitrary graphs) | High (Arbitrary graphs + SPMD) |
| **Dispatch Latency** | Low (local PCIe) | High (DCN round-trips) | Low (masked by Async Dispatch) |
| **Coordination** | User-managed collectives | Graph-based dataflow | Sharded Dataflow (PLAQUE) |
| **Graph Scaling** | Individual host graphs | $M \times N$ sharded edges | Compact sharded nodes |
| **Resource Mgmt** | User-defined / Static | System-managed / Dynamic | System-managed / Virtualized |

## Why Pathways is Better for Researchers

1. **MPMD Support**: Researchers trying to implement mixture-of-experts or pipelining in JAX must write complex distributed coordination. In Pathways, it's a natural part of the dataflow model.
2. **Resource Sharing**: In JAX, one job usually owns the whole TPU Pod. In Pathways, researchers can share the Pod for fine-tuning or small-scale experiments without reconfiguring hardware.
3. **Debugging and Inspection**: A single-controller model provides a centralized view of the entire computation, making it easier to monitor and debug distributed executions.

## Performance vs JAX

- **Throughput Parity**: For large language model training (e.g., T5-11B on 512 cores), Pathways achieves **identically matching throughput** to JAX.
- **Latency Masking**: Pathways' asynchronous dispatch effectively cancels out the DCN latency inherent in the single-controller model.

## Paper Reference

> "PATHWAYS combines the flexibility of single-controller frameworks like TensorFlow with the performance of multi-controller frameworks like JAX." — §2
