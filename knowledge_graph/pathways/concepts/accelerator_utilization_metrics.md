# Accelerator Utilization Metrics

---
id: accelerator-utilization-metrics
paper: pathways-2022
section: "§5 (Evaluation), Appendix A"
tags:
  - hardware
  - metrics
  - performance
related:
  - gang_scheduling
  - resource_manager
  - parallel_asynchronous_dispatch
---

## Definition

Accelerator utilization metrics measure how effectively an ML system keeps expensive hardware (TPUs/GPUs) busy with actual computation versus idling due to coordination, data transfer, or software overhead.

## Key Metrics in Pathways Evaluation

1. **Throughput (Tokens/sec or Steps/sec)**: The primary measure of end-to-end model performance.
2. **Weak Scaling**: Performance as both hardware and model size increase proportionally.
3. **Strong Scaling**: Performance as hardware increases for a fixed model size.
4. **Dispatch Latency**: The time from client command to kernel execution on the accelerator.
5. **Context-Switch Overhead**: The performance penalty for switching between different users' computations in a multi-tenant environment.

## Pathways Performance Benchmarks

- **SPMD Scaling**: Pathways matches multi-controller JAX performance for SPMD training on 2048 TPU cores (see §5.1).
- **Small-Scale Throughput**: For extremely small computations (0.33 ms), Pathways *exceeds* JAX throughput because its asynchronous design allows it to hide host-side overhead better than JAX can locally via Python.
- **Pipeline Parallelism**: Achieves near-equal throughput to an SPMD configuration (128 cores, 3B decoder model) with 2-16 pipeline stages.
- **Multi-Island Efficiency**: Training across two islands connected by DCN retains **97.2%** of the throughput of a single ICI-connected island of the same size.
- **Multi-Tenancy**: With 16 concurrent clients, Pathways achieves ~100% device utilization even if individual programs are too small to saturate the hardware.

## Why Pathways Metrics Matter

Historically, single-controller systems suffered from "hundreds of microseconds" or even "milliseconds" of dispatch latency over DCN. Pathways evaluates whether its **asynchronous dispatch** and **centralized scheduling** successfully move these delays off the critical path.

## Paper Reference

> "PATHWAYS scales to 2048 TPU cores with ~100% performance relative to an SPMD system like JAX." — §5.1
