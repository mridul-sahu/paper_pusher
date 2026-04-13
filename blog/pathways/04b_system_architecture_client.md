# Part 4b: System Architecture — The Client

> "The PATHWAYS client assigns virtual devices to computations and then registers these sharded computations with the resource manager."
> — §4.2, Pathways paper

---

## Summary

In a single-controller architecture, the **client** is the central point of failure for performance. If the client machine is slow, the entire 2048-TPU Pod sits idle. This section explains the engineering behind the Pathways client — specifically how it manages to stay out of the "critical path" of the accelerators through a process called **progressive lowering**.

---

## The Client's Burden

The client (the researcher's local machine or a dedicated management VM) has to:
1. Run the user's Python code.
2. Trace that code into a graph of computations.
3. Assign those computations to thousands of virtual devices.
4. "Lower" that high-level graph into physical machine instructions.
5. Dispatch those instructions over the network.

If naive, this process adds hundreds of milliseconds of overhead. Pathways solves this by splitting the lowering process into two phases: **Device-Agnostic** and **Device-Specific**.

---

## Phase 1: Device-Agnostic IR

When the user first calls a Pathways function, the client builds an **Intermediate Representation (IR)** that knows *how much* work is being done, but not *where* it is being done.

- It knows: "We are running a 128-way sharded Matrix Multiply."
- It doesn't know: "Device #4 is at IP `10.0.0.5`."

This phase happens **once** during the initial execution of the Python program. The result is a compact, sharded representation. This is where Pathways beats TF v1—the IR is "shardedly compact," meaning one node in the IR represents work across all 2048 chips.

---

## Phase 2: Device-Specific Lowering

Once the Resource Manager (see [Part 4a](04a_system_architecture_resource_manager.md)) provides the physical device list, the client can finish the job.

It runs a set of **compiler passes** that:
1. Injects the actual physical IP addresses of the worker hosts.
2. Allocates specific memory buffers (HBM addresses) for the tensors.
3. Produces a final **Plaque program** (dataflow graph) for the workers.

### Why This Splitting Matters

Because the first phase is device-agnostic, Pathways can **cache** the result. If you run the same training step 1,000,000 times, the client only has to do the hard work of IR construction once. The second phase is much faster—it’s just "filling in the blanks" of physical locations.

---

## The Secret Weapon: Sharded Buffers

The client also manages **sharded buffers**.

Usually, if a model has a 1GB weight tensor sharded over 100 devices, the system would track 100 individual "futures" or handles. Tracking 1,000,000 tensors over 1,000 devices becomes a massive bookkeeping overhead for the client's CPU.

Pathways introduces a **Logical Sharded Buffer** abstraction. The client tracks **one** handle for the entire 1GB weight. The runtime handles the mapping of that logical handle to the 100 physical shards. This reduces the client's bookkeeping effort by an order of magnitude.

---

## Asynchronous Interaction

Most importantly, the client interacts with the workers **asynchronously**.

- The client sends a command: "Run Computation A and give me a future for out_A."
- The client does **not wait**.
- It immediately starts preparing the command for Computation B, using the future for `out_A` as an input.

This allows the client to build up a deep "queue" of work on the accelerators. Even if the client machine takes 10ms to prepare a command, and the accelerator takes only 5ms to run it, the accelerators don't idle because the client started preparing work 5 minutes ago!

---

## Summary Table: Client Optimization

| Innovation | Purpose |
|------------|---------|
| **Progressive Lowering** | Separates compute-heavy IR building from fast physical address injection. |
| **Sharded Buffers** | Amortizes bookkeeping costs by tracking logical tensors instead of physical shards. |
| **Future-based API** | Enables the client to "fire and forget" commands, masking its own latency. |

---

*Next up: [Part 4c — Coordination: How Plaque Executes the Distributed Dataflow →](04c_system_architecture_coordination.md)*
