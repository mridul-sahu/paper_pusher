# Pathways: Asynchronous Distributed Dataflow for ML

---
id: pathways-2022
title: "Pathways: Asynchronous Distributed Dataflow for ML"
authors:
  - Paul Barham
  - Aakanksha Chowdhery
  - Jeff Dean
  - Sanjay Ghemawat
  - Steven Hand
  - Dan Hurt
  - Michael Isard
  - Hyeontaek Lim
  - Ruoming Pang
  - Sudip Roy
  - Brennan Saeta
  - Parker Schuh
  - Ryan Sepassi
  - Laurent El Shafey
  - Chandramohan A. Thekkath
  - Yonghui Wu
year: 2022
venue: MLSys (5th Conference on Machine Learning and Systems)
arxiv: "2203.12533"
institution: Google
tags:
  - distributed-systems
  - ML-infrastructure
  - accelerators
  - TPU
  - dataflow
  - scheduling
---

## Abstract

Pathways is a large-scale orchestration layer for accelerators, explicitly designed to enable exploration of new systems and ML research ideas while retaining state-of-the-art performance for current models. It uses a **sharded dataflow graph** of asynchronous operators that consume and produce futures, and efficiently **gang-schedules** heterogeneous parallel computations on thousands of accelerators while coordinating data transfers over dedicated interconnects.

The system adopts a **single-controller model** made performant through a novel **parallel asynchronous dispatch** design that lets the control plane execute in parallel despite data-plane dependencies.

## Key Results

- **~100% accelerator utilization** for SPMD computations over 2048 TPUs.
- **Throughput parity** with multi-controller JAX for all realistic computation sizes.
- Pipelining across **16 stages** with throughput comparable to SPMD.
- Efficient training across **two islands** of accelerators over DCN, achieving **~97%** throughput versus a single large island.
- Multi-tenancy with **zero context-switch overhead** when resources fit in HBM.

## Motivation

1. **SPMD limitations**: Modern models (pipelining, MoE) exceed what MPI-style SPMD supports naturally.
2. **Heterogeneous clusters**: Exclusive access to large homogeneous islands is expensive and wasteful.
3. **Foundation models**: Shared, multi-task models benefit from resource multiplexing and state sharing.
4. **System co-evolution risk**: Systems over-specialized to current workloads fail to anticipate future needs.

## Core Contributions

1. A **single-controller architecture** that matches multi-controller performance via asynchronous dispatch.
2. A **sharded dataflow graph** representation that scales to thousands of shards without graph explosion.
3. **Gang-scheduled dynamic dispatch** with centralized per-island schedulers.
4. **Parallel asynchronous dispatch** exploiting statically known resource requirements of compiled functions.
5. A **resource manager** supporting virtual device abstraction and dynamic resource allocation.
6. Demonstrated **multi-tenancy**, **multi-island training**, and **pipeline parallelism** at scale.

## Evaluation Highlights

| Model | Params | TPU Cores | Throughput |
|-------|--------|-----------|------------|
| T5-Base | 270M | 32 | 618k tokens/s |
| T5-Large | 770M | 32 | 90.4k tokens/s |
| T5-3B | 3B | 512 | 282.8k tokens/s |
| T5-11B | 11B | 512 | 84.8k tokens/s |
| 3B Decoder (SPMD) | 3B | 128 | 125.7k tokens/s |
| 3B Decoder (Pipeline, S=16) | 3B | 128 | 131.4k tokens/s |

## Future Directions

- **Data-dependent vectorized control flow**: Fine-grained routing for MoE and routed capsule networks.
- **Advanced resource management**: Priorities, performance isolation, access control at millisecond timescales.
- **GPU support**: High-level architectural choices are believed valid for large-scale GPU systems.
- **Transparent suspend/resume**: Virtual-to-physical device remapping for migration without user cooperation.

## See Also

- [Architecture](architecture.md)
- [Concepts](concepts/)
- [Relationships (Turtle)](relationships.ttl)
