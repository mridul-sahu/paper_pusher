# Part 4c: System Architecture — Plaque: The Distributed Coordination Layer

> *"Plaque is a sharded dataflow system of asynchronous operators that consume and produce futures."*
> — §4.3, Pathways paper

---

## The Coordination Problem

Consider the challenge: a single client has dispatched a computation graph to the Pathways runtime. The graph specifies that `function_A` runs on 512 TPUs in Island 1, `function_B` runs on 512 TPUs in Island 2, and there's a data transfer between them. How does the system ensure that:

1. All 512 TPUs in Island 1 start `function_A` at the **same time** (gang-scheduling)?
2. The data transfer is **initiated** only after `function_A` completes?
3. All 512 TPUs in Island 2 start `function_B` at the **same time**, and only after the transfer arrives?
4. All of this happens with **microsecond overhead**, not the milliseconds that naive RPC coordination would introduce?

This is the problem **Plaque** solves.

---

## What Is Plaque?

Plaque is Pathways' **distributed execution engine**. It's a dataflow system whose operators are sharded across workers, and whose edges carry futures (not materialized data). Plaque replaces the need for:

- A centralized coordinator that becomes a bottleneck.
- Direct client-to-worker RPCs that scale as O(num_workers).
- Global barriers that introduce unnecessary synchronization.

### The Key Design Principles

**1. Sharded Operators, Not Sharded Graphs**

Unlike TF v1 (which materialized per-device subgraphs), Plaque represents each compiled function as a **single sharded operator** that spans all the devices it executes on. The operator is "logically one thing" but "physically distributed" — just like the data it processes.

**2. Futures All the Way Down**

Every output of every operator is a **future** — a promise that a value will eventually exist on a set of accelerators. When Plaque encounters a dependency (operator B depends on operator A's output), it doesn't wait for A to complete. Instead, it immediately **enqueues** B, passing A's future as B's input. B's execution will **block on the accelerator** until A's future resolves, but Plaque's coordination logic doesn't wait.

This is critical for **pipelining**: Plaque can enqueue hundreds of operations in rapid succession, building up a deep pipeline of work on the accelerators. The accelerators execute operations as their inputs become ready, without any back-and-forth with the coordination layer.

**3. Hierarchical Coordination**

Plaque uses a **two-level hierarchy**:

- **Per-island coordinators** handle scheduling within a single TPU island (hundreds of chips on the same ICI mesh). This coordinator has low-latency access to all devices in the island.
- **Cross-island coordination** is handled by the Plaque runtime, which sets up data transfers via DCN and ensures that the per-island coordinators on both sides are ready.

```
┌─────────────────────────────────────────────────────────┐
│                     Pathways Client                      │
│              (dispatches sharded dataflow)               │
└──────────────┬────────────────────┬──────────────────────┘
               │                    │
               ▼                    ▼
┌──────────────────────┐  ┌──────────────────────┐
│  Island 1 Coordinator│  │  Island 2 Coordinator│
│  (512 TPUs on ICI)   │  │  (512 TPUs on ICI)   │
│                      │◄─┤                      │
│  Gang-schedules      │  │  Gang-schedules      │
│  within this island  │  │  within this island  │
└──────────────────────┘  └──────────────────────┘
         ▲                         ▲
         │         DCN             │
         └─────────────────────────┘
            Cross-island transfers
```

![Coordination Architecture](./assets/04c_coord_v1_1776015081977.png)

---

## How Execution Flows Through Plaque

Let's trace a concrete example: a single training step of a model that is pipeline-parallel across two islands.

### Step 1: Client Dispatches

The client constructs the sharded dataflow:

```
[forward_pass]  ──data──→  [transfer]  ──data──→  [backward_pass]
   Island 1                  DCN                    Island 2
```

Each node is a sharded operator. The client sends this graph to Plaque in a single, compact message.

### Step 2: Plaque Distributes

Plaque's runtime receives the graph and:
1. **Sends** `forward_pass` to Island 1's coordinator.
2. **Sends** `backward_pass` to Island 2's coordinator.
3. **Sets up** the transfer between islands.

Both islands receive their work **simultaneously** — there's no sequential dispatch.

### Step 3: Island 1 Executes

Island 1's coordinator:
1. Resolves `forward_pass` to specific XLA programs for each of its 512 TPUs.
2. **Gang-schedules** all 512 shards (see [Part 4d](04d_system_architecture_gang_scheduling.md)).
3. The TPUs execute the forward pass.
4. Upon completion, the output tensors are gathered and sent via DCN.

### Step 4: Data Transfer

The transfer operator:
1. **Within Island 1:** Uses ICI to gather output tensors from all 512 TPUs to the DCN-connected hosts.
2. **Across DCN:** Transfers the gathered data to Island 2's hosts.
3. **Within Island 2:** Uses ICI to scatter the received data to all 512 TPUs.

### Step 5: Island 2 Executes

Island 2's coordinator:
1. Sees that `backward_pass`'s input future has resolved (data has arrived).
2. **Gang-schedules** `backward_pass` across all 512 TPUs.
3. Execution proceeds.

### The Critical Observation

In a naive implementation, Steps 1–5 would be **sequential** — each step waits for the previous one. With Plaque's futures model, **Steps 1 and 2 happen almost instantly** (just structural dispatch), and **Steps 3–5 are pipelined**:

- While Island 1 is executing the forward pass for step N, the client is already dispatching step N+1.
- While data is transferring from Island 1 to Island 2 for step N, Island 1 is starting step N+1's forward pass.
- The only true blocking point is at the accelerator level — a TPU waits for its input future to resolve.

---

## Plaque's Execution Model: Asynchronous Operators

Each Plaque operator has a simple lifecycle:

```
PENDING → READY → EXECUTING → COMPLETE
```

- **PENDING**: The operator is registered but its input futures haven't resolved yet.
- **READY**: All input futures have resolved. The operator is eligible for gang-scheduling.
- **EXECUTING**: The operator's XLA program is running on accelerators.
- **COMPLETE**: Output futures are resolved, unblocking downstream operators.

The transition from READY to EXECUTING is controlled by the **gang-scheduler** (detailed in [Part 4d](04d_system_architecture_gang_scheduling.md)). This is where Plaque enforces the **consistent ordering** constraint that prevents deadlocks on TPUs.

---

## Why Not Just Use gRPC?

A natural question: why build Plaque at all? Why not just have the client send gRPC calls to each worker?

The answer is **scale and ordering**:

1. **Scale**: With 2048 TPUs across 128+ hosts, the client would need to maintain 128+ RPC connections and coordinate ordering across all of them. Any jitter in RPC latency would cause accelerator stalls.

2. **Ordering**: TPUs require that communicating computations are enqueued in a **globally consistent order**. If TPU A is doing an `AllReduce` with TPU B, both must enqueue the `AllReduce` before either enqueues any subsequent work. Naive per-worker RPCs provide no ordering guarantees.

3. **Data locality**: gRPC would route all data through the client. Plaque ensures data flows **directly between accelerators** (or between workers via DCN), never touching the client.

Plaque provides all three: scalable dispatch, consistent ordering, and accelerator-to-accelerator data flow.

---

## Performance Characteristics

The paper reports that Plaque's coordination overhead is **negligible** compared to computation time:

- For a 64B Transformer training step (~100ms of computation), cross-island transfer via DCN adds only **~3% overhead** — meaning the coordination layer introduces virtually no pipeline bubbles.
- Gang-scheduling overhead (time between when an operator becomes READY and when it starts EXECUTING) is on the order of **microseconds**, well below the millisecond-scale computation times.

---

*Next up: [Part 4d — Gang Scheduling: Preventing Deadlocks at Thousands-of-Chips Scale →](04d_system_architecture_gang_scheduling.md)*
