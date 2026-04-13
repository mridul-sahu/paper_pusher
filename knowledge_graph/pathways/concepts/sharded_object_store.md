# Sharded Object Store

---
id: sharded-object-store
paper: pathways-2022
section: "§4.6 (Data management)"
tags:
  - data-management
  - memory
  - HBM
  - object-store
related:
  - resource_manager
  - asynchronous_distributed_dataflow
  - accelerator_utilization_metrics
---

## Definition

The **Sharded Object Store** is a per-host data management layer in Pathways, similar to Ray's object stores but extended to also track buffers held in accelerator **HBM (High-Bandwidth Memory)** at each shard.

## Key Features

1. **HBM-aware**: Unlike Ray's object store (which only manages host DRAM), Pathways tracks buffers in both host memory and device HBM.
2. **Opaque handles**: Client programs hold references via opaque handles that allow the system to **migrate** objects transparently.
3. **Intermediate value storage**: Objects are stored while waiting to be transferred between accelerators or passed to subsequent computations.
4. **Ownership labels**: Objects are tagged with ownership labels for **garbage collection on failure** of program or client.
5. **Back-pressure**: Computations are stalled (via simple back-pressure) when they cannot allocate memory because other computations' buffers temporarily occupy HBM.
6. **Sharded buffer abstraction**: The client uses a logical buffer abstraction that may be distributed over multiple devices, amortizing bookkeeping (including reference counting) at logical-buffer granularity instead of per-shard.

## Contrast with Ray

| Feature | Ray Object Store | Pathways Object Store |
|---------|-----------------|----------------------|
| Host DRAM tracking | ✓ | ✓ |
| HBM buffer tracking | ✗ | ✓ |
| Transparent migration | Limited | ✓ (via opaque handles) |
| Back-pressure on HBM | ✗ | ✓ |
| Sharded buffer abstraction | ✗ | ✓ |

## Why HBM Tracking Matters

- PCIe bandwidth is **much smaller** than HBM or accelerator interconnect bandwidth.
- Moving data between HBM and host DRAM (context-switching) wastes significant accelerator cycles.
- By tracking HBM buffers, the system can keep intermediate values **on-device** and route them directly via ICI, avoiding unnecessary PCIe transfers.
- TF and Ray suffer performance penalties from lacking this store (TF transfers data back to client; Ray transfers GPU→DRAM before returning handles).

## Paper Reference

> "Each host manages a sharded object store that is similar to Ray's object stores, but extended to also track buffers held in accelerator HBM at each shard." — §4.6
