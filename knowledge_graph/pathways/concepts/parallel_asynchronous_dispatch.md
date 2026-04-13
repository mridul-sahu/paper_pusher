# Parallel Asynchronous Dispatch

---
id: parallel-asynchronous-dispatch
paper: pathways-2022
section: "§4.5 (Parallel asynchronous dispatch)"
tags:
  - optimization
  - dispatch
  - latency
related:
  - asynchronous_distributed_dataflow
  - single_controller_model
  - gang_scheduling
---

## Definition

**Parallel Asynchronous Dispatch** is an architectural optimization in Pathways that allows the system to execute the host-side work for multiple computation nodes in parallel and ahead of time, even when there are data dependencies between them.

## The Problem

In traditional single-controller systems:
1. Client sends Node A to Worker.
2. Worker computes A.
3. Client waits for A to finish to get the output shape/type.
4. Client sends Node B to Worker.

The round-trip latency (DCN) between Client and Worker makes this **agonizingly slow**.

## The Solution

Pathways exploits the fact that most ML computations are **regular compiled functions** (Appendix B). For these functions:
- Output types and tensor shapes are known **before** any data is computed.
- Resource requirements (memory) can be estimated in advance.

Because of this, the Pathways client can:
1. Fire a "future" for the output of Node A.
2. **Immediately** begin preparing Node B, using the known metadata of A's output.
3. Batch the dispatch calls for many nodes into parallel threads.

## Implementation Detail

- When a `@pw.program` is called, Pathways traces the Python code to identify the sequence of compiled functions.
- It then initiates multiple dispatch tasks in a **thread pool**.
- Each task enqueues its work on the workers; the workers wait for input "buffer futures" to be resolved by previous computations.
- This effectively moves the "scheduling" logic onto the accelerators and keeps the host out of the critical path.

## Outcome

This optimization is what allows Pathways (a single-controller system) to match the performance of bare-metal multi-controller systems like JAX.

## Paper Reference

> "PATHWAYS exploits the statically known resource usage of regular compiled functions to run most of the host-side work for a computation's nodes in parallel." — §4.5
