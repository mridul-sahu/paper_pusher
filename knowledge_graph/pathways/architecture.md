# Pathways System Architecture

---
id: pathways-architecture
paper: pathways-2022
tags:
  - architecture
  - system-design
  - distributed-systems
---

## Overview

![System Architecture](file:///Users/mridulsahu/ai_stuff/paper_pusher/diagrams/papers/pathways/01-system-architecture.animated.svg)
*Figure 1: Pathways Over-arching System Architecture. [Edit Source](file:///Users/mridulsahu/ai_stuff/paper_pusher/diagrams/papers/pathways/01-system-architecture.excalidraw) | [**View Animation**](file:///Users/mridulsahu/ai_stuff/paper_pusher/diagrams/papers/pathways/01-system-architecture.animated.svg)*

Pathways is a **client-server** system where the runtime executes programs on system-managed islands of compute on behalf of many clients. It builds on prior systems including **XLA** (for TPU computations), **TensorFlow** (for CPU-based distributed computations), and **JAX** / **TensorFlow** Python APIs.

```
┌─────────┐     ┌──────────────────────────────────────────────────┐
│  Client  │────▶│              Pathways Runtime                    │
│ (Python) │     │  ┌───────────────────┐  ┌───────────────────┐   │
│ JAX/TF   │     │  │ Resource Manager   │  │ PLAQUE Dataflow   │   │
└─────────┘     │  │ (global)           │  │ (coordination)    │   │
                │  └───────────────────┘  └───────────────────┘   │
                │  ┌──────────────┐ ┌──────────────┐              │
                │  │ Island       │ │ Island       │  ...         │
                │  │ ┌──────────┐ │ │ ┌──────────┐ │              │
                │  │ │Scheduler │ │ │ │Scheduler │ │              │
                │  │ └──────────┘ │ │ └──────────┘ │              │
                │  │ ┌─────┬─────┐│ │ ┌─────┬─────┐│              │
                │  │ │Exec │Exec ││ │ │Exec │Exec ││              │
                │  │ │(dev)│(dev)││ │ │(dev)│(dev)││              │
                │  │ └─────┴─────┘│ │ └─────┴─────┘│              │
                │  └──────────────┘ └──────────────┘              │
                └──────────────────────────────────────────────────┘
```

## Components

### 1. Resource Manager (Global)

- Centralized management of **all devices** across all islands.
- Clients request **virtual slices** with specific 2D/3D mesh shapes.
- Dynamically assigns **physical devices** to virtual devices.
- Supports dynamic addition/removal of backend compute resources.
- Enables future **transparent suspend/resume** and **migration**.

> See: [resource_manager.md](concepts/resource_manager.md)

### 2. Client

- Assigns virtual devices to computations and registers them with the Resource Manager.
- Constructs a **device-location-agnostic IR** using a custom **MLIR dialect**.
- The IR is progressively **lowered** via compiler passes to include physical device locations.
- Uses a **sharded buffer abstraction** to represent logical buffers distributed over multiple devices—amortizing bookkeeping cost at logical-buffer granularity instead of per-shard.
- Supports both **standalone programs** (one compiled function per RPC) and **traced programs** (a full dataflow DAG from a single Python block).

### 3. Coordination Substrate (PLAQUE)

- A closed-source **production sharded dataflow system** used at Google.
- Handles all cross-host coordination over DCN.
- The low-level Pathways IR is converted directly to a PLAQUE program (dataflow graph).
- Requirements met by PLAQUE:
  - **Compact representation**: Single node per sharded computation (N shards → still 4 IR nodes for A→B).
  - **Sparse data exchanges**: Messages between dynamically chosen subsets of shards.
  - **Low-latency critical messages** with high-throughput batching.
  - **Extensibility**: Used for background housekeeping (config distribution, monitoring, error delivery).

> See: [plaque_coordination.md](concepts/plaque_coordination.md)

### 4. Scheduler (Per-Island, Centralized)

- **One centralized scheduler per island** that consistently orders all computations.
- Implements **gang-scheduling**: ensures communicating computations are enqueued in consistent order.
- Current implementation uses **FIFO ordering**; future work may reorder by estimated execution time.
- When a subgraph can be scheduled statically, a **single message** describes the entire subgraph to minimize network traffic.

> See: [gang_scheduling.md](concepts/gang_scheduling.md)

### 5. Executor (Per-Device)

- Each device has an executor that:
  - Enqueues compiled functions with **buffer futures** as inputs.
  - Manages network sends of output buffer futures to destination shards.
  - Coordinates with the scheduler for execution ordering.

### 6. Sharded Object Store (Per-Host)

- Similar to **Ray's object stores** but extended to track **HBM buffers** at each shard.
- Client programs hold references via **opaque handles** that support migration.
- Intermediate values stored in object stores during transfers between computations.
- Objects tagged with **ownership labels** for garbage collection on failure.
- **Back-pressure** supported to stall computations when HBM is temporarily full.

> See: [sharded_object_store.md](concepts/sharded_object_store.md)

## Execution Flow

![Pathways Execution Flow](file:///Users/mridulsahu/ai_stuff/paper_pusher/diagrams/papers/pathways/02-execution-flow.animated.svg)
*Figure 2: The vertical execution path. [Edit Source](file:///Users/mridulsahu/ai_stuff/paper_pusher/diagrams/papers/pathways/02-execution-flow.excalidraw) | [**View Animation**](file:///Users/mridulsahu/ai_stuff/paper_pusher/diagrams/papers/pathways/02-execution-flow.animated.svg)*

```
Client Python Code
       │
       ▼
   Program Tracing (optional @pw.program decorator)
       │
       ▼
   Virtual Device Assignment
       │
       ▼
   MLIR IR Construction (device-agnostic)
       │
       ▼
   IR Lowering (compiler passes → physical locations)
       │
       ▼
   PLAQUE Dataflow Program
       │
       ├──▶ Scheduler (per-island gang-scheduling)
       │        │
       │        ▼
       │    Executor (per-device enqueue)
       │        │
       │        ▼
       │    XLA Compiled Function on TPU
       │
       └──▶ Data transfers (ICI intra-island / DCN inter-island)
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Single-controller** over multi-controller | Enables richer MPMD patterns, centralized resource management, multi-tenancy |
| **Sharded dataflow** representation | Avoids M×N edge explosion; scales to thousands of shards |
| **Parallel async dispatch** | Masks single-controller overhead by exploiting static resource knowledge |
| **Centralized per-island scheduler** | Required for TPU (non-preemptible); beneficial for GPU efficiency too |
| **Virtual device abstraction** | Enables resource virtualization, migration, and dynamic allocation |
| **PLAQUE for coordination** | Production-tested, low-latency, supports sparse exchanges |

## Hardware Topology

```
┌─────────────────────────────────┐
│         Island (TPU Pod)         │
│  ┌──────┐ ┌──────┐ ┌──────┐    │
│  │ Host │ │ Host │ │ Host │... │
│  │ 4 TPU│ │ 4 TPU│ │ 4 TPU│   │
│  └──┬───┘ └──┬───┘ └──┬───┘   │
│     └────ICI─┴───ICI──┘        │
└────────────────┬────────────────┘
                 │ DCN (Datacenter Network)
┌────────────────┴────────────────┐
│         Island (TPU Pod)         │
│  ┌──────┐ ┌──────┐ ┌──────┐    │
│  │ Host │ │ Host │ │ Host │... │
│  └──────┘ └──────┘ └──────┘    │
└─────────────────────────────────┘
```

- **ICI** (Inter-Chip Interconnect): High-bandwidth, low-latency within an island
- **DCN** (Datacenter Network): RDMA between islands, higher latency
- TPU islands can have **hundreds of hosts** with thousands of devices connected all-to-all

## See Also

- [Summary](summary.md)
- [Concepts](concepts/)
- [Relationships (Turtle)](relationships.ttl)
