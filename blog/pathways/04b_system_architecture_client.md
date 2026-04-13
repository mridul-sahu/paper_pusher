# Part 4b: System Architecture — The Client

> *"The client library runs within the user's Python process and manages the lifecycle of computations."*
> — §4.2, Pathways paper

---

## The Single-Controller's Achilles' Heel

The deepest fear with a single-controller architecture is that the **client becomes a bottleneck**. If thousands of accelerators are waiting for one Python process to tell them what to do, any latency in the client—Python GIL, garbage collection pauses, serialization overhead—multiplies into devastating idle time across the entire cluster.

The Pathways client library is engineered specifically to avoid this. It achieves this through three mechanisms: **sharded dataflow representation**, **eager asynchronous dispatch**, and **a futures-based execution model**.

---

## What the Client Actually Does

The Pathways client library runs **inside the user's Python process** as a native extension. When the user calls a JIT-compiled JAX function, the client:

1. **Receives** the compiled XLA HLO program and its sharding annotations.
2. **Constructs** a **sharded dataflow graph** — the Pathways computation representation.
3. **Dispatches** the graph to the Pathways runtime **asynchronously**.
4. **Returns futures** immediately, allowing the user's Python code to continue.

### The Critical Insight: Separation of Compilation and Dispatch

In JAX's standard multi-controller model, compilation and dispatch are tightly coupled. The client JIT-compiles a function, then immediately dispatches it to the local accelerator. If compilation takes 500ms, the accelerator idles for 500ms.

In Pathways, compilation and dispatch are **decoupled**:

- **Compilation** happens once per unique function signature. The compiled XLA program is **cached** and reused. Since ML training loops are repetitive (the same `train_step` function is called millions of times), compilation is amortized to essentially zero cost.
- **Dispatch** is the act of telling the runtime "run this cached program on these devices with this data." Dispatch is designed to be **blazingly fast** — microseconds, not milliseconds.

---

## Sharded Dataflow: The Compact Representation

This is perhaps the most under-appreciated innovation in the Pathways paper.

### The Problem with TF v1

In TensorFlow v1's single-controller model, the computation graph was **fully materialized** for every device. If you had a 2048-way sharded `AllReduce`, the graph contained:

- 2048 computation nodes (one per device)
- 2048 × 2048 = **4+ million** communication edges

This O(N²) explosion in graph edges was a primary reason TF v1 struggled at scale.

### Pathways' Solution: Sharded Nodes and Edges

Pathways represents the computation as a **sharded dataflow** with three abstractions:

**1. Nodes** represent compiled functions. A single node in the Pathways graph represents a function that is **simultaneously executed** on all devices in a virtual mesh. There's one node regardless of whether the function runs on 1 device or 2048.

**2. Edges** represent data dependencies. A single edge connects a producing node to a consuming node. The edge carries **N data tuples** (one per shard), but is represented as a **single logical edge** regardless of shard count.

**3. Buffers** represent distributed data. A `PathwaysBuffer` is a handle to data that may be **sharded across hundreds or thousands of devices**. The client only holds the handle, not the data itself. Data never flows through the client.

```
┌──────────────┐                    ┌──────────────┐
│  train_step  │  ── single edge ── │  apply_grads │
│  (2048-way   │     (2048 data     │  (2048-way   │
│   sharded)   │      tuples)       │   sharded)   │
└──────────────┘                    └──────────────┘

vs. TF v1:

┌──────┐ ┌──────┐ ┌──────┐         ┌──────┐ ┌──────┐
│ ts_0 │ │ ts_1 │ │ ts_2 │  ...    │ ag_0 │ │ ag_1 │ ...
└──┬───┘ └──┬───┘ └──┬───┘         └──────┘ └──────┘
   │        │        │    × 2048         ▲        ▲
   └────────┤────────┤───────────────────┴────────┘
            └────────┴──── 4M+ edges ────────┘
```

This compact representation is critical for both **client memory** (the graph stays small) and **dispatch speed** (the client sends a compact descriptor, not a massive graph).

---

## The Futures Model

Every operation in Pathways returns a **`PathwaysFuture`** — a handle that represents a value that will exist *at some point in the future* on some set of accelerators. The client dispatches work without waiting for it to complete:

```python
# This returns immediately — future_1 is a handle, not data
future_1 = pathways.run(forward_fn, input_batch)

# This also returns immediately — depends on future_1
future_2 = pathways.run(backward_fn, future_1)

# This also returns immediately — depends on future_2
future_3 = pathways.run(optimizer_fn, future_2)

# Only blocks when we actually need a result on the host
loss_value = future_1.result()  # NOW we wait
```

The key property is that **data flows directly between accelerators** — from the output of `forward_fn` to the input of `backward_fn` — **without ever touching the client**. The client only orchestrates the graph structure. The actual tensor data (potentially gigabytes) stays on the accelerators.

This is the fundamental mechanism that **eliminates the single-controller bottleneck**. The client's job is to emit graph structure fast enough to keep the accelerator pipeline full. Since graph nodes are tiny (just metadata: function ID, device assignment, edge connections), the client can emit them orders of magnitude faster than accelerators can consume them.

---

## Buffer Management

The Pathways client uses **reference-counted distributed buffers**:

- Each `PathwaysBuffer` is a handle stored on the client that references sharded data distributed across accelerators.
- When the user's Python code drops its reference to a buffer (or it goes out of scope), the client **asynchronously** tells the runtime to free the corresponding accelerator memory.
- Buffers can be **donated** — when a function consumes a buffer and the client knows no other references exist, the buffer's accelerator memory can be reused in-place by the output, avoiding a copy.

The buffer donation optimization is particularly important for training loops, where the model parameters are updated in every step. Without donation, every step would allocate new HBM for the updated parameters and then free the old ones — a continuous churn of expensive HBM allocations.

---

## Handling Cross-Island Data Transfers

When a computation spans multiple accelerator islands (connected via DCN), data transfers between islands must be explicitly managed. The Pathways client represents these as **transfer nodes** in the dataflow graph:

```
Island A                          Island B
┌────────────┐                    ┌────────────┐
│ compute_fn │ ──transfer_node──→ │ reduce_fn  │
│ (512 TPUs) │   (via DCN)       │ (512 TPUs) │
└────────────┘                    └────────────┘
```

Transfer nodes are first-class citizens in the graph. They:
- **Participate in gang-scheduling** — the transfer is coordinated with the computations on both islands.
- **Use the ICI mesh within each island** to first gather data to the DCN-connected hosts, then use DCN for cross-island transfer, then scatter across the destination island's ICI mesh.
- **Overlap with computation** — the system pipelines transfers with ongoing computation to hide DCN latency.

---

## Performance: How Fast Is the Client?

The key metric is **dispatch overhead per step**. For the client to not be a bottleneck, it must dispatch one step's worth of computation faster than the accelerators can execute the previous step.

For a typical large model training step that takes 10–100ms on accelerators, the client's dispatch (graph construction + RPC to runtime) takes **microseconds**. This means the client can pipeline **hundreds** of steps ahead of the accelerators, ensuring the accelerator pipeline never starves.

The paper demonstrates this concretely: for a 6.8B-parameter Transformer on 2048 TPU v4 chips, Pathways achieves **within 2%** of the peak performance of a hand-tuned multi-controller baseline (Table 1), proving that single-controller dispatch overhead is negligible at scale.

---

*Next up: [Part 4c — Plaque: Coordinating Execution Across Thousands of Chips →](04c_system_architecture_coordination.md)*
