# SPMD vs MPMD

---
id: spmd-vs-mpmd
paper: pathways-2022
section: "§1 (Introduction), §2 (Design Motivation)"
tags:
  - parallelism
  - programming-model
  - computation
  - SPMD
  - MPMD
related:
  - single_controller_model
  - gang_scheduling
  - asynchronous_distributed_dataflow
---

## Definitions

![SPMD vs MPMD Comparison](file:///Users/mridulsahu/ai_stuff/paper_pusher/diagrams/papers/pathways/03-spmd-vs-mpmd.animated.svg)
*Figure: Comparison between SPMD (Homogeneous) and MPMD (Heterogeneous) models. [Edit Source](file:///Users/mridulsahu/ai_stuff/paper_pusher/diagrams/papers/pathways/03-spmd-vs-mpmd.excalidraw) | [**View Animation**](file:///Users/mridulsahu/ai_stuff/paper_pusher/diagrams/papers/pathways/03-spmd-vs-mpmd.animated.svg)*

- **SPMD (Single Program Multiple Data)**: All accelerators run the **same computation** in lockstep. Communication between accelerators uses collectives like AllReduce. Inspired by MPI.
- **MPMD (Multiple Program Multiple Data)**: Different accelerators can run **different computations**. Sub-parts of the overall computation are mapped to collections of smaller islands.

## Current State of the Art

Most state-of-the-art ML workloads use **SPMD**, implemented in multi-controller systems like JAX, PyTorch, and MPI. This works well for standard data-parallel training.

## Limitations of SPMD

1. **Pipelining**: Very large language models require pipeline parallelism (GPipe, PipeDream, Megatron-LM), which is heterogeneous and doesn't fit SPMD naturally.
2. **Mixture of Experts (MoE)**: Computational sparsity requires fine-grained control flow and heterogeneous computation—MoE routing is fundamentally MPMD.
3. **Multi-task learning**: Models serving multiple tasks benefit from different computations on different device subsets.
4. **Resource efficiency**: SPMD requires exclusive access to a large homogeneous island, which is wasteful.

## MPMD Advantages

- **Flexibility**: Map sub-computations to smaller, more readily available device islands.
- **Heterogeneity**: Natural support for models with different computation types on different accelerators.
- **Resource sharing**: Multiple programs can share the same devices via multiplexing.

## Pathways Approach

Pathways supports **both SPMD and MPMD** through its single-controller model:
- Unmodified JAX SPMD code runs as-is.
- Users can also express arbitrary DAGs where different nodes run on different device sets.
- The system handles data movement and resharding automatically.

## Examples of MPMD Workloads

| Workload | Why MPMD |
|----------|----------|
| Pipeline-parallel training | Different pipeline stages on different devices |
| Mixture of Experts | Data-dependent routing to different expert subsets |
| Neural Architecture Search | Different candidate architectures in parallel |
| Multi-modal multi-task | Different modalities/tasks on different device subsets |
| Foundation model fine-tuning | Shared base + task-specific heads on different devices |

## Paper Reference

> "Recently, researchers have begun to run into the limits of SPMD for ML computations." — §1
