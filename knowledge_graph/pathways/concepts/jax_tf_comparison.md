# JAX / TensorFlow Comparison

---
id: jax-tf-comparison
paper: pathways-2022
section: "§2 (Design Motivation), §3 (Programming Model), §6 (Discussion)"
tags:
  - frameworks
  - comparison
  - programming-model
  - JAX
  - TensorFlow
  - PyTorch
  - Ray
related:
  - single_controller_model
  - spmd_vs_mpmd
  - accelerator_utilization_metrics
---

## Systems Compared

### Multi-Controller Systems

| System | Architecture | Dispatch | Communication |
|--------|-------------|----------|---------------|
| **JAX** | Multi-controller | Local PCIe (low latency) | XLA collectives over ICI |
| **PyTorch** | Multi-controller | Local PCIe (low latency) | NCCL collectives |
| **MPI** | Multi-controller | Local PCIe (low latency) | MPI collectives |

**Advantages**: Very low dispatch latency (communication only over fast PCIe links).

**Limitations**:
- Poor match for pipelining, MoE, and computational sparsity.
- Non-collective communication requires user-implemented coordination.
- Assumes exclusive hardware ownership; no multi-tenancy.
- JAX cannot scale beyond a single TPU pod (collectives limited to ICI).

### Single-Controller Systems

| System | Architecture | Dispatch | Communication |
|--------|-------------|----------|---------------|
| **TF v1** | Single-controller | Over DCN (high latency) | Send/recv ops over DCN |
| **Pathways** | Single-controller | Over DCN but **async parallel** | PLAQUE dataflow + ICI/DCN |

**TF v1 Limitations**:
- Dispatch latencies accumulate for pipelined models with many stages.
- Over-specialized for single, small, exclusively-owned islands.
- Materializes full sharded computation graph → millions of edges at scale.
- No centralized scheduler → cannot ensure consistent ordering across programs.

### Pathways vs JAX Performance

- **Identical throughput** for all realistic computation sizes across all tested models (T5-Base through T5-11B).
- Pathways **exceeds** JAX throughput for very small computations in multi-tenant scenarios.
- Pathways can scale JAX programs to **multiple TPU pods** (JAX is limited to one).

### Pathways vs TF v1

- Pathways uses **sharded** dataflow (compact) vs TF's **materialized** graph (M×N edges).
- Pathways has **parallel async dispatch** vs TF's **sequential** host-side work.
- Pathways has a **centralized gang scheduler** vs TF's **control-edge barriers**.

### Pathways vs Ray

- Both are single-controller, but Ray is general-purpose.
- Ray lacks an **HBM object store** and efficient GPU interconnect transfer primitives.
- Ray shows ~10x worse per-computation performance, but could potentially match with engineering (fast paths, on-GPU object stores).

## Pathways-JAX Integration

- Pathways serves as a **plug-in replacement** for the JAX backend.
- Unmodified JAX code runs on Pathways; SPMD computations gain access to all provisioned cores, not just locally connected ones.
- Pathways enables JAX programs to scale to **multiple TPU pods** for the first time.
- Optional `@pw.program` tracer generates multi-node dataflow programs from Python blocks.

## Paper Reference

> "PATHWAYS combines the flexibility of single-controller frameworks with the performance of multi-controllers." — §2
