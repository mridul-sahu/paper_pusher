# Part 1: Introduction — Why SPMD Is Breaking

> "The recent breakthroughs in large-scale machine learning have been powered by a combination of new model architectures, massive datasets, and the availability of large quantities of specialized accelerator hardware."
> — §1, Pathways paper

---

## Summary

The Pathways paper introduces a system designed to solve a fundamental scalability problem in high-performance machine learning. At its core, Pathways is an **orchestration layer** that manages thousands of accelerators (TPUs and GPUs) using a **single-controller architecture** that manages to match the performance of bare-metal multi-controller systems like JAX.

Wait—why do we need this? Wasn't JAX working fine? To understand Pathways, we first have to understand the **SPMD** (Single Program Multiple Data) model and why it’s starting to fail.

---

## The SPMD Legacy

Modern distributed ML is largely built on the **SPMD model**, popularized by MPI (Message Passing Interface) and implemented in systems like **JAX** and **PyTorch DDP**.

In SPMD:
- Every host runs the **exact same program**.
- All accelerators execute computations in **lockstep**.
- Communication happens via **collectives** (e.g., `AllReduce`) where all participants must join at the same time.

This model is extremely efficient for homogeneous workloads like standard data-parallel training (e.g., training a BERT or ResNet model). It allows programmers to write code as if it's running on one giant machine while the hardware handles the messy details of synchronization.

---

## The Boiling Point

The paper identifies several emerging research trends that are fundamentally incompatible with the SPMD model:

### 1. Pipelining & Heterogeneous Computation
As models grow larger than a single accelerator's memory, researchers use **pipeline parallelism**. This involves splitting a model across stages. Stage 1 executes on Device A, Stage 2 on Device B, and so on.

In a pure SPMD system, this is awkward: every device is forced to run the same code, so you end up with complex `if (stage == 1)` logic. More importantly, those different stages don't compute in lockstep—they depend on each other sequentially.

### 2. Mixture of Experts (MoE)
MoE models (like **Switch Transformer** or **GLaM**) route inputs to only a fraction of "experts" in each layer. This creates **computational sparsity**: different accelerators end up running different expert kernels on different amounts of data.

SPMD's lockstep requirement becomes a massive bottleneck here. If one expert takes 10ms and another takes 2ms, all participants wait for the 10ms expert to finish before they can move to the next step.

### 3. Foundation Models & Multi-Task Learning
The industry is moving toward "generalist" models that can perform many tasks (vision, language, robotics) at once. Researchers want to route different parts of a request to different specialized sub-graphs. In SPMD, coordinating this across thousands of devices is a distributed systems nightmare.

---

## Enter: MPMD (Multiple Program Multiple Data)

Pathways shifts the paradigm from SPMD to **MPMD**.

In an MPMD world:
- Different accelerators can run **different programs**.
- Computations can be mapped to smaller, specialized "islands" of hardware.
- High-level orchestrators handle the routing of data between these islands.

![SPMD vs MPMD Lecture](./assets/01-introduction-lecture.png)

---

## The Pathways Solution

The paper proposes a new architecture characterized by three things:

1. **A Single-Controller Model**: One central coordinator manages all devices. This provides the flexibility of a global view, making it easier to implement MPMD patterns.
2. **Asynchronous Dispatch**: The coordinator doesn't wait for one computation to finish before sending the next one. This allows it to "mask" the latency of the datacenter network.
3. **Gang-Scheduling**: It ensures that communicating computations are enqueued in a consistent order across devices, preventing deadlocks (especially critical on non-preemptible hardware like TPUs).

By combining these, Pathways enables researchers to explore complex, heterogeneous research ideas without sacrificing the near-100% hardware efficiency they've come to expect from SPMD.

---

## Why Should You Care?

If you're training a 100M parameter model on a single GPU, you don't need Pathways. But if you’re building the next **Gemini**, **PaLM**, or **GPT-5**—models that span thousands of chips and use advanced routing—Pathways provides the blueprint for the infrastructure that makes it possible.

---

*Next up: [Part 2 — Design Motivation: The Brutal Choice Between Fast and Flexible →](02_design_motivation.md)*
