# Part 4f: System Architecture — Data Management

> *"Data flows directly between accelerators without touching the client."*
> — §4, Pathways paper

---

## The Data Locality Challenge

In a system spanning thousands of accelerators across multiple physical islands, data management is not just about storage — it's about **minimizing data movement**. Every byte that moves between devices consumes:

1. **Bandwidth** — a finite resource, especially over DCN.
2. **Time** — during which accelerators may be idle waiting for inputs.
3. **Energy** — data movement is the dominant power cost in modern ML systems.

Pathways' data management layer is designed to keep data **where it's needed, for as long as it's needed, and to move it as little as possible**.

---

## The Buffer Hierarchy

Pathways manages data at three levels:

### 1. On-Device Buffers (HBM)

The lowest level. Each accelerator (TPU/GPU) has **High-Bandwidth Memory (HBM)** — fast but limited capacity (e.g., 32GB per TPU v4 chip). This is where model parameters, activations, gradients, and optimizer states live during execution.

Pathways manages HBM through **static buffer allocation**: because compiled functions have known input/output shapes (the compiled function abstraction from §3), the system pre-allocates exactly the right amount of HBM before execution begins. This eliminates:
- **Runtime allocation overhead** (no `malloc` on the critical path).
- **Memory fragmentation** (all buffers fit perfectly into pre-assigned slots).
- **Over-allocation waste** (no need for padding or dynamic growth margins).

### 2. Distributed Sharded Buffers

The abstraction exposed to user code. A `PathwaysBuffer` represents a **logical tensor that is physically sharded** across many devices:

```
Logical Tensor: [batch=1024, hidden=4096]

Physical Layout (4-way data sharded):
  TPU 0: [batch=256, hidden=4096]  ← shard 0
  TPU 1: [batch=256, hidden=4096]  ← shard 1
  TPU 2: [batch=256, hidden=4096]  ← shard 2
  TPU 3: [batch=256, hidden=4096]  ← shard 3
```

The client holds a **handle** (metadata + reference count) to the sharded buffer. The actual data exists only on the accelerators.

### 3. Transfer Buffers

When data must move between accelerator islands (over DCN), Pathways uses a multi-stage transfer protocol:

```
Source Island                          Destination Island
┌──────────────────┐                   ┌──────────────────┐
│  TPU HBM         │                   │  TPU HBM         │
│  (source data)   │                   │  (output data)   │
│       │          │                   │       ▲          │
│       ▼  (ICI)   │                   │       │  (ICI)   │
│  Host DRAM       │                   │  Host DRAM       │
│  (staging buffer)│                   │  (staging buffer)│
│       │          │                   │       ▲          │
└───────┼──────────┘                   └───────┼──────────┘
        │          ========DCN========         │
        └──────────────────────────────────────┘
```

1. **ICI gather**: Data is collected from TPU HBM across the island to host DRAM staging buffers.
2. **DCN transfer**: Staging buffers are transferred between hosts over the datacenter network.
3. **ICI scatter**: Data is distributed from host DRAM into TPU HBM across the destination island.

This three-stage protocol ensures that the expensive DCN transfer only happens once (host-to-host), while the fast ICI links handle the fan-in (gather) and fan-out (scatter).

---

## Buffer Lifecycle and Reference Counting

Pathways uses **distributed reference counting** to manage buffer lifetimes:

1. **Creation**: When a compiled function produces output, the system allocates HBM on each participating device and creates a `PathwaysBuffer` handle.

2. **Consumption**: When a downstream function takes the buffer as input, it **increments** the reference count. When it's done, it **decrements** the count.

3. **Donation**: If a function is the **sole consumer** of a buffer (reference count = 1) and the output has the same shape, the buffer's HBM can be **reused in-place**. This avoids both allocation and deallocation costs.

4. **Deletion**: When the reference count reaches zero, the HBM is freed asynchronously.

### Buffer Donation in Training Loops

Buffer donation is particularly powerful for the parameter update step in training:

```python
# Without donation:
old_params = ...          # 10GB in HBM
gradients = backward(old_params)
new_params = update(old_params, gradients)  # allocates 10GB MORE
del old_params            # frees 10GB
# Peak HBM: 20GB for params alone

# With donation:
old_params = ...          # 10GB in HBM
gradients = backward(old_params)
new_params = update(old_params, gradients)  # reuses old_params' HBM
# Peak HBM: 10GB for params
```

For a 32B-parameter model in bfloat16, this saves **64GB of HBM** — the entire capacity of two TPU v4 chips. At the scale of thousands of chips, donation is the difference between "fits in memory" and "doesn't."

---

## Cross-Program Data Sharing

A key capability enabled by Pathways' data management is **cross-program data sharing**. In a multi-tenant system:

- **Program A** (pre-training) produces model weights that are stored in sharded buffers.
- **Program B** (fine-tuning) can access those **same buffers** without copying them.
- **Program C** (serving) can read the weights while Program A continues training.

This is only possible because:
1. Buffers are managed by the **runtime** (not owned by individual programs).
2. The **resource manager** knows which physical devices hold which buffers.
3. The **reference counting** system prevents premature deallocation.

---

## Data Transfer Performance (§5.3)

The paper provides concrete performance numbers for cross-island data transfers:

### 64B Transformer: Two Islands of 512 TPUs

For the largest benchmark — data-parallel training over 1024 TPUs in two islands:

- Each training step computes gradients independently on each island.
- Gradients (hundreds of GB) are transferred between islands via DCN.
- The system achieves **97.2% throughput** compared to the single-island SPMD baseline.

This means the three-stage transfer protocol (ICI gather → DCN transfer → ICI scatter) adds only **2.8% overhead** — despite moving hundreds of gigabytes of gradient data between physical locations in a datacenter.

### Why the Overhead Is So Small

Three factors:
1. **Pipelining**: Gradient transfers for step N overlap with forward pass computation for step N+1.
2. **Efficient staging**: The ICI gather/scatter is extremely fast (TB/s bandwidth within an island).
3. **Minimal DCN hops**: The transfer involves only the hosts directly attached to the accelerators — data doesn't traverse multiple network switches.

---

## Comparison with Multi-Controller Data Management

| Feature | Multi-Controller | Pathways |
|---------|-----------------|----------|
| Buffer ownership | Per-process | System-managed |
| Cross-program sharing | Copy required | Zero-copy sharing |
| Buffer donation | Limited | Full lifecycle optimization |
| Cross-island transfer | Manual NCCL/MPI | Automatic 3-stage protocol |
| Memory allocation | Dynamic | Static pre-allocation |

---

*Next up: [Part 5 — Evaluation: Proving It Works at Scale →](05_evaluation.md)*
