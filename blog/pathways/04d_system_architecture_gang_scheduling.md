# Part 4d: System Architecture — Gang Scheduling

> "The gang scheduler... ensures that all coordinating computations are enqueued in a globally consistent order."
> — §4, Pathways paper

---

## The Deadlock Dilemma

To understand gang scheduling, we must first understand the unique constraint of **TPUs** (though the same logic applies to specialized GPU interconnects like NVLink).

TPUs use a high-speed, direct Inter-Chip Interconnect (**ICI**). Key properties:
1. TPU A can write directly into TPU B's memory without involving B's host CPU.
2. This is extremely fast (~TB/s aggregate bandwidth), but…
3. It requires **both** TPU A and TPU B to be in a **compatible state**. If A is sending data that B isn't expecting (because B is running a different program), the result is at best corrupted data and at worst a **system deadlock**.

Concretely: if TPU 0 and TPU 1 are both part of an `AllReduce`, and the system enqueues a *different* computation on TPU 1 before the `AllReduce`, then:

- TPU 0 enters the `AllReduce` and tries to RDMA data to TPU 1.
- TPU 1 is running the wrong program and doesn't reciprocate.
- TPU 0 blocks waiting for TPU 1.
- TPU 1 blocks waiting for its program's own collective.
- **Both TPUs are deadlocked forever.**

This is not a theoretical risk. It's a **mathematical certainty** for any system that runs multiple programs on shared TPU hardware without centralized scheduling.

---

## What Is Gang Scheduling?

Gang scheduling is a technique from the HPC world where **all processes that need to communicate are scheduled simultaneously**. "Gang" refers to the group of processes that must run together.

In Pathways, a "gang" is **all the accelerator shards of a single compiled function**. For a 512-way sharded `train_step`, all 512 TPU shards must be enqueued and start executing in a **consistent order** relative to any other computations that use the same devices.

### The Consistency Requirement

The key invariant is not that all shards start at exactly the same clock time (that's impossible with distributed systems). The requirement is **consistent ordering**: if computation A is scheduled before computation B on TPU 0, then A must also be scheduled before B on TPU 1, TPU 2, …, TPU 511.

```
CORRECT (consistent ordering):
TPU 0:  [... A ...] [... B ...]
TPU 1:  [... A ...] [... B ...]
TPU 2:  [... A ...] [... B ...]

DEADLOCK (inconsistent ordering):
TPU 0:  [... A ...] [... B ...]
TPU 1:  [... B ...] [... A ...]  ← A and B both deadlock
TPU 2:  [... A ...] [... B ...]
```

---

## Pathways' Gang-Scheduling Design

Pathways implements gang-scheduling with a **per-island centralized scheduler**:

### Per-Island FIFO Queues

Each accelerator island has a **single scheduler** that maintains a **FIFO (First-In, First-Out) queue** of ready computations. When a compiled function's input futures all resolve (i.e., it transitions from PENDING to READY in Plaque), it enters the FIFO queue.

The FIFO discipline guarantees consistent ordering: if function A enters the queue before function B, then A is dispatched to **all** devices in the island before B is dispatched to **any** of them.

```
Island Scheduler FIFO Queue:
┌─────────┬─────────┬─────────┬─────────┐
│ train_0 │ train_1 │ eval_0  │ train_2 │ → dispatched in order
└─────────┴─────────┴─────────┴─────────┘
  ↓ dispatched to all 512 TPUs simultaneously
```

### Why Centralized Per-Island (Not Global)?

A natural question: why not have a **single global scheduler** for the entire cluster?

1. **Latency**: A global scheduler would need to coordinate across DCN, adding milliseconds of latency per scheduling decision. Per-island schedulers operate within the ICI mesh, with microsecond-scale coordination.

2. **Independence**: Different islands often run independent computations. There's no need to synchronize Island 1's scheduler with Island 2's unless they share a computation.

3. **Scalability**: Per-island schedulers process scheduling decisions in parallel. A global scheduler would be a serial bottleneck.

For **cross-island** computations (data-parallel training across two islands), the Plaque coordination layer ensures that both islands' schedulers receive and process the computation in a compatible order.

![Gang Scheduling Visualization](./assets/04d_gang_v1_1776015142292.png)

---

## Multi-Tenancy Through Gang Scheduling

Gang-scheduling is what makes Pathways' multi-tenancy actually work. Without it, sharing TPUs between programs is impossible (on TPUs) or extremely inefficient (on GPUs).

With gang-scheduling, the system can **interleave** programs at millisecond timescales:

```
TPU 0:  [Program A: 2ms] [Program B: 1ms] [Program A: 2ms] [Program C: 0.5ms]
TPU 1:  [Program A: 2ms] [Program B: 1ms] [Program A: 2ms] [Program C: 0.5ms]
...
TPU 511: [Program A: 2ms] [Program B: 1ms] [Program A: 2ms] [Program C: 0.5ms]
```

All TPUs execute the same sequence of programs. Context-switching between programs is **near-zero overhead** because:

1. Each program is a pre-compiled XLA function — there's no "loading" step.
2. The program's buffers are pre-allocated in HBM (managed by the resource manager).
3. The switch is just: stop executing program A's function, start executing program B's function.

### The Multi-Tenancy Experiment (§5.2)

The paper demonstrates this with a controlled experiment using a synthetic workload:

- **Setup**: 8 TPU v3 chips, programs with 0.33ms compute time each.
- **Variables**: 1, 4, 8, and 16 concurrent clients.
- **Result**: With 1 client, the 0.33ms compute time is too small to saturate the 8 chips (there's more dispatch overhead than compute). With **16 concurrent clients**, device utilization reaches **~100%** — the gang-scheduler seamlessly interleaves 16 programs at sub-millisecond granularity.

This is the mechanism that would allow, for example, 16 different researchers to fine-tune different LoRA adapters on the same base model simultaneously, each getting their fair share of accelerator time without interfering with each other.

---

## Gang Scheduling on GPUs

The paper makes an important observation (Appendix A.5): while GPUs *can* execute concurrent programs without centralized scheduling (they support hardware-level preemption and concurrent kernel execution), gang-scheduling is **still beneficial** for performance:

> "For a cluster prioritizing ML training workloads, where throughput is more important than latency, it is more efficient to dedicate an entire GPU, or a static fraction of a GPU, to a single carefully sized computation at a time, than to allow the GPU driver and hardware runtime to dynamically multiplex its computational resources across competing concurrent computations."

This means Pathways' gang-scheduling approach is **not TPU-specific** — it's a genuinely better scheduling strategy for ML workloads on any accelerator architecture.

---

## Scheduling Policy: Beyond FIFO

The current implementation uses FIFO as the scheduling policy, but the paper notes this is **extensible**. Alternative policies could include:

- **Priority-based scheduling** — high-priority training jobs preempt low-priority evaluation jobs.
- **Fair-share scheduling** — each tenant gets a proportional share of accelerator time.
- **Deadline-aware scheduling** — latency-sensitive serving workloads get preferential scheduling over batch training.

The per-island scheduler is designed as a pluggable component, allowing different policies to be deployed without changing the rest of the system.

---

## The Gang-Scheduling Invariant

To summarize, Pathways' gang-scheduling provides a single critical invariant:

> **All devices participating in a computation will execute it in the same relative order with respect to all other computations on those devices.**

This invariant:
- **Prevents deadlocks** on TPUs (where it's mandatory).
- **Maximizes throughput** on GPUs (where it's beneficial).
- **Enables multi-tenancy** on shared hardware (where it's transformative).
- **Costs almost nothing** — the scheduling decision is O(1) FIFO lookup per computation.

---

*Next up: [Part 4e — Asynchronous Dispatch: Hiding the Single-Controller's Latency →](04e_system_architecture_asynchronous_dispatch.md)*
