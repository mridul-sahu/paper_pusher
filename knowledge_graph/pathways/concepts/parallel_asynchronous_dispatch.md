# Parallel Asynchronous Dispatch

---
id: parallel-asynchronous-dispatch
paper: pathways-2022
section: "§4.5 (Parallel asynchronous dispatch)"
tags:
  - optimization
  - dispatch
  - latency
  - compiled-functions
related:
  - asynchronous_distributed_dataflow
  - gang_scheduling
  - single_controller_model
---

## Definition

**Parallel asynchronous dispatch** is a novel optimization in Pathways that runs host-side work (scheduling, resource allocation, coordination) for multiple computation nodes **in parallel**, rather than serializing the work to happen after each predecessor has been enqueued.

## The Problem

In traditional asynchronous dispatch (Figure 4a in the paper):
1. Host A enqueues node A, receives a future, sends it to Host B.
2. Host B allocates inputs, prepares node B, waits for node A to complete.
3. Only then does node B start.

When **computation time is shorter than host-side scheduling time**, the asynchronous pipeline **stalls**—the host-side work becomes the critical bottleneck.

## The Solution

Because compiled functions are **regular** (their input/output shapes and resource requirements are known before execution), Pathways can:
1. Compute a successor node's input shapes **before the predecessor is even enqueued**.
2. Run host-side preparation for **all nodes in parallel**.
3. Send a **single message** describing an entire statically schedulable subgraph to the scheduler.

## Key Properties

- **Exploits static knowledge**: Only works for compiled functions with known resource requirements (the common case in ML workloads).
- **Graceful fallback**: For nodes with data-dependent control flow (where shapes aren't known until runtime), falls back to sequential dispatch.
- **Single-message subgraph**: Minimizes network traffic while still allowing interleaving with other programs' computations.

## Performance Impact

From the pipeline micro-benchmark (Figure 7):
- **Parallel dispatch** shows three phases as pipeline stages increase:
  1. Fixed client overhead amortized.
  2. Transfer costs begin to dominate.
  3. Fixed scheduling overhead amortized.
- **Sequential dispatch** is consistently slower, demonstrating the benefit of parallelizing host-side work.

The paper shows this is critical for pipelines with many stages and short per-stage computations.

## Compiled Functions

The concept of **compiled functions** (Appendix B) is central to parallel dispatch:
- Input/output types and tensor shapes known **before input data is computed**.
- Loop bounds known at scheduling time (or specified as maximum trip counts).
- Conditionals are "functional" with same output type for both branches.
- Resource requirements estimable in advance.

## Paper Reference

> "We therefore introduce a novel parallel asynchronous dispatch design, which exploits the statically known resource usage of regular compiled functions to run most of the host-side work for a computation's nodes in parallel." — §4.5
