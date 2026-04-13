# Single-Controller Model

---
id: single-controller-model
paper: pathways-2022
section: "§2 (Design Motivation), §4 (System Architecture)"
tags:
  - architecture
  - control-plane
  - design-pattern
related:
  - spmd_vs_mpmd
  - resource_manager
  - gang_scheduling
---

## Definition

In a **single-controller** system, a central coordinator manages the execution of programs across all devices on behalf of clients. This contrasts with **multi-controller** systems where identical copies of the user program run directly on every host.

## Multi-Controller vs Single-Controller

| Aspect | Multi-Controller (JAX, PyTorch) | Single-Controller (Pathways) |
|--------|-------------------------------|------------------------------|
| **Dispatch latency** | Low (local PCIe) | Higher (DCN), mitigated by async dispatch |
| **Programming model** | SPMD only | SPMD + MPMD + arbitrary DAGs |
| **Resource ownership** | Exclusive per-program | Shared, centrally managed |
| **Coordination** | User-implemented | System-provided |
| **Multi-tenancy** | Not supported | First-class support |
| **Virtualization** | Not supported | Virtual devices, migration |

## Why Pathways Chose Single-Controller

1. **Richer computation patterns**: MPMD, pipelining, MoE, and data-dependent routing are awkward or impossible in multi-controller.
2. **Centralized resource management**: Enables sharing, virtualization, elasticity, and fair scheduling.
3. **Easier user experience**: Users don't need to implement coordination primitives.

## How Pathways Overcomes Single-Controller Limitations

The historical problem with single-controller (e.g., TF v1) was **dispatch latency**—the round-trip over DCN between the controller and workers. Pathways solves this through:

1. **Parallel asynchronous dispatch**: Host-side work for multiple nodes runs concurrently.
2. **Sharded dataflow graph**: Compact representation avoids coordination overhead at scale.
3. **Program tracing**: Multiple compiled functions batched into a single program, reducing RPCs.

## Paper Reference

> "PATHWAYS combines the flexibility of single-controller frameworks with the performance of multi-controllers." — §2
