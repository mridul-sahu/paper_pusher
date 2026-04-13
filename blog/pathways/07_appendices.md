# Part 7: Appendices — The Hardware Deep-Dive

> *"Hardware acceleration is critical to modern deep learning; unfortunately, achieving high performance with accelerators is a non-trivial systems exercise."*
> — Appendix A, Pathways paper

---

## Why This Appendix Matters

You cannot understand Pathways' design decisions without understanding the **hardware constraints** that shaped them. Every architectural choice — gang-scheduling, asynchronous dispatch, static buffer allocation, the compiled function abstraction — is a direct response to the physical realities of accelerator hardware.

This appendix covers the five key hardware considerations from the paper (Appendix A), plus additional context on TPU vs GPU differences and the structure of ML programs.

---

## A.1: Batching — Why Accelerators Need Fat Workloads

### The End of Dennard Scaling

Since ~2006, individual transistors no longer get faster with each manufacturing generation (the end of Dennard scaling). Instead, chips get **more parallel** — more cores, wider SIMD units, larger systolic arrays. This means:

- **Arithmetic is cheap** — modern accelerators can perform trillions of floating-point operations per second.
- **Memory bandwidth is the bottleneck** — reading/writing data from/to memory is orders of magnitude slower than computing on it.

### The Arithmetic Intensity Problem

For an accelerator to be fully utilized, each byte read from memory must fuel **many** arithmetic operations. This ratio is called **arithmetic intensity** (AI = FLOPs / bytes transferred).

| Operation | Arithmetic Intensity | Accelerator Utilization |
|-----------|---------------------:|-----------------------:|
| Element-wise add | 1 FLOP / 2 bytes | ~0.1% |
| Matrix-vector multiply | N FLOPs / N bytes | ~1-10% |
| Matrix-matrix multiply | N² FLOPs / N bytes | ~50-90% |
| Batched matmul (batch B) | B×N² FLOPs / N bytes | ~90-99% |

**Batching** increases arithmetic intensity by amortizing memory reads across multiple data points. This is why deep learning uses mini-batches — it's not just a statistical technique, it's a **hardware necessity**.

### The Batching Dilemma

But batching has limits:
- **HBM capacity**: Larger batches require more memory for activations. TPU v4 has 32GB HBM per chip — this limits batch size for large models.
- **Convergence**: Very large batch sizes can slow or destabilize model convergence (Shallue et al., 2018).
- **Unified memory pitfall**: GPUs support transparent paging between HBM and host DRAM, but if HBM overflows, the computation drops from **HBM bandwidth** (~2 TB/s) to **PCIe bandwidth** (~32 GB/s) — a **60× slowdown** (Lim et al., 2021).

This is why Pathways' **static buffer allocation** is critical: by knowing exactly how much HBM each compiled function needs, the system prevents overflow and maintains full HBM bandwidth.

---

## A.2: Asynchronous Programming — Why Synchronous Dispatch Is Death

Accelerators use an **asynchronous programming model** for good reason. A synchronous model (dispatch → wait → dispatch → wait) wastes enormous amounts of accelerator time:

```
Synchronous:
Host:   [dispatch]───[wait]───[dispatch]───[wait]───
Accel:       [compute]────────[idle][compute]────────

Asynchronous:
Host:   [dispatch][dispatch][dispatch]...
Accel:       [compute][compute][compute]...
```

In the synchronous model, the accelerator is idle while the host processes the result and dispatches the next operation. The idle time includes:
- **PCIe round-trip latency** (~1-10 μs).
- **Host-side Python/C++ processing** (~10-100 μs).
- **Interrupt and context-switch overhead** (~1-10 μs).

For small operations (e.g., a 100μs matrix multiply), synchronous dispatch can waste **50%+ of accelerator time** on idle cycles.

The solution is **enqueuing** computations on a **stream** — a FIFO queue of operations that the accelerator processes sequentially without host involvement. This is exactly what Pathways extends to the distributed setting: the client enqueues entire **computation graphs** asynchronously, building up a deep pipeline of work.

---

## A.3: High-Performance Interconnects — ICI, NVLink, and DCN

### The Interconnect Hierarchy

Modern ML clusters have a three-level interconnect hierarchy:

```
Level 1: On-chip (registers, on-chip SRAM)
         Bandwidth: ~100 TB/s    Latency: ~1 ns

Level 2: Accelerator interconnect (ICI for TPU, NVLink for GPU)
         Bandwidth: ~1-10 TB/s   Latency: ~1-10 μs

Level 3: Datacenter network (DCN — Ethernet, InfiniBand)
         Bandwidth: ~10-100 GB/s  Latency: ~10-100 μs
```

Each level is **10-100× slower** than the one above it. The system must minimize communication at slow levels and maximize it at fast levels.

### TPU Inter-Chip Interconnect (ICI)

TPU chips are connected by a **custom mesh network** built directly into the silicon. Key properties:

- **Direct chip-to-chip communication** — no host CPU involvement needed.
- **2D or 3D torus topology** — each chip connects to its neighbors in a mesh.
- **Hundreds of chips per island** — a TPU v4 pod has ~4096 chips on a single ICI mesh.
- **Collective operations run on-chip** — the TPU's scalar core can execute `AllReduce` without any software coordination.

This is why gang-scheduling is **mandatory** for TPUs: the ICI collective operations assume all participating chips are executing the same program. If they're not, the RDMA transfers corrupt memory or deadlock.

### GPU NVLink

GPU systems use **NVLink** for high-speed communication within a single node (typically 4-8 GPUs). Key differences from ICI:

- **Smaller "islands"** — NVLink typically connects 8 GPUs on one host, vs. 1000+ TPUs on ICI.
- **Host-mediated collectives** — GPU collectives (via NCCL) are initiated by the host CPU.
- **Dynamic shapes** — GPU kernels are pre-compiled for dynamic input shapes, supporting more flexible computation patterns.

Between GPU nodes, communication uses **InfiniBand** or **RDMA-capable Ethernet** (GPUDirect RDMA).

### Datacenter Network (DCN)

DCN connects accelerator islands at datacenter scale:
- **Lower bandwidth** than ICI/NVLink (~10-100 GB/s per link).
- **Higher latency** (~100μs+).
- **Standard networking equipment** — Ethernet switches, not custom silicon.

Pathways' **three-stage transfer protocol** (ICI gather → DCN transfer → ICI scatter) is designed to minimize DCN usage while leveraging fast ICI within each island.

---

## A.4: Single-Tenancy — The Utilization Crisis

The conventional approach to accelerator management is **single-tenancy**: one program gets exclusive access to a set of accelerators for its entire lifetime.

### Why Single-Tenancy Exists

1. **HBM capacity**: ML models often consume all available HBM, leaving no room for another program's data.
2. **Context-switch cost**: Moving data in/out of HBM (32 GB at ~2 TB/s = ~16ms) is expensive relative to computation time.
3. **Interconnect interference**: Two concurrent programs competing for ICI/NVLink bandwidth degrade each other's performance.

### Why Single-Tenancy Is Wasteful

Real ML workloads are **bursty**:
- Training steps have compute-heavy phases (forward/backward pass) and communication-heavy phases (`AllReduce`). During communication, compute units are idle.
- MoE models activate only a fraction of experts per step — most compute capacity is unused.
- Interactive workloads (model development, debugging) have long idle periods between operations.

The paper estimates that typical ML clusters achieve **30-50% utilization** under single-tenancy. Pathways' gang-scheduling is designed to push this toward **100%** by interleaving multiple programs at the timescale of individual computations (~milliseconds).

---

## A.5: TPUs vs GPUs — A Systems Perspective

### TPU Design Philosophy

| Property | TPU | GPU |
|----------|-----|-----|
| Programming unit | Large XLA programs (ms–s) | Small kernels (μs–ms) |
| Shape support | Static shapes preferred | Dynamic shapes standard |
| ICI scope | 1000+ chips per mesh | 4–8 chips per NVLink domain |
| Collective initiation | On-chip (no host involvement) | Host-initiated (NCCL) |
| Multi-tenancy | Not supported (single program) | Hardware-level preemption |
| Compilation | JIT, heavy optimization | Pre-compiled, lighter optimization |
| Memory management | Static buffer assignment | Dynamic allocation |

### Key Implication for Pathways

TPUs' **single-program, non-preemptible** nature is what makes centralized gang-scheduling **mandatory**. But the paper argues this is actually an advantage:

> "Even though GPUs can execute concurrent programs without centralized scheduling, there is still a benefit from using a design like Pathways to make more efficient use of resources."

The reasoning: GPU hardware's ability to multiplex programs is not *efficient* — it just *works*. The GPU's hardware scheduler has no knowledge of ML workload structure and makes decisions that are locally optimal (maximize individual kernel throughput) but globally suboptimal (create interference between competing programs).

Pathways' centralized scheduling, by contrast, has complete visibility into all workloads and can make globally optimal decisions.

---

## Appendix B: The Structure of ML Programs

The paper formalizes what ML programs look like from the system's perspective:

### The 99% Case

Almost all computation in ML training consists of:
1. A **training loop** (Python `for`/`while` loop).
2. That calls a **small number of compiled functions** (forward, backward, optimizer update).
3. In a **fixed, predictable order**.
4. With **statically known shapes** and resource requirements.

Data-dependent control flow (e.g., early stopping, gradient clipping thresholds) is **rare** and typically occurs at boundaries between compiled functions, not inside them.

### Why This Matters for Pathways

This structure enables:
1. **Deep asynchronous pipelining** — the client can predict far ahead what the accelerators need.
2. **Compiled function caching** — the same XLA programs are reused millions of times.
3. **Static resource allocation** — the system knows exactly how much HBM each step needs.

Without this structure — if ML programs were arbitrarily complex, dynamic, unpredictable —  Pathways' asynchronous dispatch wouldn't work. The system **co-evolved with ML workloads**, and its design exploits the specific properties of how modern neural networks are trained.

---

## Appendix C: Input Data Processing

A practical concern: where does the **training data** come from?

JAX deliberately doesn't implement data loading pipelines. In practice, `tensorflow/datasets` is used for JAX programs. Pathways leverages this by:

1. Running a **CPU-based TensorFlow executor** on each Pathways worker host.
2. Allowing user programs to serialize data pipelines into TensorFlow graphs.
3. Distributing data loading across workers, decoupling input processing from expensive TPU-connected hosts.

Future plans include **streaming data protocols** that run input processing on independently managed servers, fully separating the CPU-bound data pipeline from the accelerator-bound training pipeline.

---

## Summary: Hardware Informs Architecture

Every Pathways design decision maps directly to a hardware constraint:

| Hardware Constraint | Pathways Design Decision |
|--------------------|--------------------------|
| Memory bandwidth bottleneck | Batching + static buffer allocation |
| Dispatch latency | Asynchronous programming + deep pipelining |
| ICI requires gang execution | Centralized per-island gang-scheduler |
| DCN is slower than ICI | Three-stage transfer protocol + pipelining |
| Single-tenancy waste | Multi-tenancy via millisecond gang-scheduling |
| TPU non-preemptibility | Consistent ordering invariant in scheduler |
| Compiled functions are predictable | Future-based dispatch + function caching |

Understanding these constraints is essential for evaluating not just Pathways, but any future distributed ML system. The hardware sets the rules; the system must play by them.

---

*← Back to [Series Index](index.md)*
