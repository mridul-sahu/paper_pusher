# PLAQUE Coordination

---
id: plaque-coordination
paper: pathways-2022
section: "§4.3 (Coordination implementation)"
tags:
  - coordination
  - dataflow
  - infrastructure
related:
  - sharded_dataflow_graph
  - asynchronous_distributed_dataflow
---

## Definition

**PLAQUE** is a high-performance, sharded dataflow system developed at Google. Pathways uses it as the low-level substrate for managing all distributed coordination and communication between hosts.

## Core Features Used by Pathways

1. **Compact Dataflow Graph**: PLAQUE supports "sharded nodes," enabling a representation where a single node represents a computation on thousands of accelerators.
2. **Sparse Data-Plane Exchanges**: Efficiently routes messages between specific shards (e.g., in MoE models where results go to specific expert hosts).
3. **Dynamic Routing**: Supports sparse messages sent between subsets of shards chosen at runtime.
4. **Low-Latency Signaling**: Optimized for sending small, critical coordination messages (like futures or health checks) with very low latency.
5. **Progress Tracking**: PLAQUE uses standard progress tracking mechanisms to detect when all messages for a shard have been received, enabling efficient sparse communication.

## Role in Pathways

- When a researcher writes a program, it is lowered into a Pathways IR (Intermediate Representation).
- This IR is then converted directly into a **PLAQUE program**.
- PLAQUE handles the actual "running" of this program across the cluster: it manages message queues, host health, and data delivery.
- Background tasks (like distributing configuration or aggregating monitoring data) also run as PLAQUE sub-graphs.

## Why Not Vanilla TensorFlow?

Pathways moved away from the vanilla TensorFlow runtime for coordination because TF v1 materialized explicit edges between all shards ($M \times N$), which crashed at the scale of thousands of devices. PLAQUE's sharded dataflow model solves this.

## Paper Reference

> "PATHWAYS relies on PLAQUE for all cross-host coordination that uses DCN. PLAQUE is an existing (closed-source) production sharded dataflow system used at Google." — §4.3
