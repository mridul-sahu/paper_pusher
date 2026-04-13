# PLAQUE Coordination Substrate

---
id: plaque-coordination
paper: pathways-2022
section: "§4.3 (Coordination implementation)"
tags:
  - coordination
  - dataflow
  - infrastructure
  - DCN
related:
  - sharded_dataflow_graph
  - asynchronous_distributed_dataflow
  - gang_scheduling
---

## Definition

**PLAQUE** is a closed-source, production sharded dataflow system used at Google for many customer-facing services. Pathways uses PLAQUE as its **coordination substrate** for all cross-host communication over DCN.

## Role in Pathways

The low-level Pathways IR is converted **directly** into a PLAQUE program (a dataflow graph). PLAQUE then handles:
- Sending scheduling messages and data handles between hosts.
- Routing shard-tagged data tuples between computation nodes.
- Background housekeeping: configuration distribution, monitoring, cleanup, error delivery.

## Requirements (All Met by PLAQUE)

| Requirement | Description |
|-------------|-------------|
| **Compact sharded representation** | A single node per sharded computation, regardless of shard count |
| **Sparse data exchanges** | Messages between dynamically chosen subsets of shards |
| **Low-latency critical messages** | Scheduling messages on the critical path must have minimal delay |
| **High-throughput batching** | Messages destined for the same host are batched efficiently |
| **Extensibility** | General-purpose dataflow engine reused for housekeeping tasks |

## Could PLAQUE Be Replaced?

The paper suggests that the Pathways design could be re-implemented using other distributed frameworks:

> "We believe that it would be feasible to re-implement the full PATHWAYS design using other distributed frameworks such as **Ray** rather than PLAQUE."

However, some additions to Ray would be needed:
- An **HBM object store** (Ray lacks this).
- Primitives to efficiently **transfer remote objects over the GPU interconnect**.

## Progress Tracking

PLAQUE uses standard progress tracking mechanisms (from Akidau et al., 2013; Murray et al., 2013) to detect when all messages for a shard have been received, enabling efficient sparse communication.

## Paper Reference

> "PATHWAYS relies on PLAQUE for all cross-host coordination that uses DCN. PLAQUE is an existing (closed-source) production sharded dataflow system used at Google." — §4.3
