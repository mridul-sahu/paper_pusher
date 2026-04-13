# Accelerator Utilization Metrics

---
id: accelerator-utilization-metrics
paper: pathways-2022
section: "§5 (Evaluation), Appendix A"
tags:
  - hardware
  - metrics
  - performance
  - TPU
  - GPU
related:
  - gang_scheduling
  - parallel_asynchronous_dispatch
  - jax_tf_comparison
---

## Key Metrics from the Paper

### Dispatch Overhead (§5.1)

- **Pathways matches JAX throughput** for computation sizes ≥ 2.3 ms (16 hosts, 128 TPUs) and ≥ 35 ms (512 hosts, 2048 TPUs).
- Pathways **outperforms** TF and Ray single-controller systems across all configurations.
- In **Fused** mode, Pathways matches JAX up to **1000 TPU cores**.
- In **Chained** mode, Pathways outperforms JAX OpByOp up to **256 cores**.

### Multi-Tenancy (§5.2)

- **Zero context-switch overhead** when multiplexing concurrent programs (resources fit in HBM).
- Pathways **exceeds** JAX's maximum throughput for very small computations.
- Gang-scheduling interleaves programs at **millisecond scale** or less.

### Large-Scale Model Training (§5.3)

| Model | Params | TPU Cores | JAX Throughput | Pathways Throughput |
|-------|--------|-----------|----------------|---------------------|
| T5-Base | 270M | 32 | 618k tok/s | 618k tok/s |
| T5-Large | 770M | 32 | 90.4k tok/s | 90.4k tok/s |
| T5-3B | 3B | 512 | 282.8k tok/s | 282.8k tok/s |
| T5-11B | 11B | 512 | 84.8k tok/s | 84.8k tok/s |

### Pipeline Parallel Performance

| Configuration | TPU Cores | Throughput |
|---------------|-----------|------------|
| SPMD | 128 | 125.7k tok/s |
| Pipeline, S=4, M=16 | 128 | 133.7k tok/s |
| Pipeline, S=8, M=32 | 128 | 132.7k tok/s |
| Pipeline, S=16, M=64 | 128 | 131.4k tok/s |
| Pipeline, S=16, M=64 | 512 | 507.8k tok/s |

### Multi-Island Training

- 64B and 136B parameter models trained across **two islands** of accelerators.
- Achieves **~97%** throughput compared to a single island with equivalent total devices.
- DCN transfers incur **minimal overhead** even at 128 hosts per island.

## Hardware Considerations (Appendix A)

### Batching (A.1)
- Unlocks parallelism and memory re-use but pressures limited HBM capacity.
- Very large batch sizes can slow model convergence.

### Asynchronous Programming (A.2)
- Accelerators use async APIs to mask dispatch latency.
- Synchronous abstractions waste too many cycles between PCIe latency and kernel scheduling.

### Interconnects (A.3)
- **GPU**: NVLink (intra-host) + InfiniBand/RDMA (inter-host).
- **TPU**: Custom mesh network (ICI) built into chips; direct chip-to-chip without host involvement.

### Single-Tenancy (A.4)
- Accelerators not traditionally shared between programs.
- Fine-grained context-switching is expensive due to HBM-to-DRAM paging over PCIe.
- Stranded resources when a program doesn't fully utilize its devices.

## Paper Reference

> "We demonstrate that PATHWAYS can achieve performance parity (~100% accelerator utilization) with state-of-the-art systems when running SPMD computations over 2048 TPUs." — Abstract
