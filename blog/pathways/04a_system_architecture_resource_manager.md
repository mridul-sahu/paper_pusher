# Part 4a: System Architecture — The Resource Manager

> "The resource manager... is responsible for the centralized management of devices across all of the islands."
> — §4.1, Pathways paper

---

## Summary

The **Resource Manager** is the "brain" of the Pathways cluster. In this section, the paper describes how Pathways moves away from the traditional model (where programs own hardware exclusively) and toward a **virtualized, shared resource pool**. It is the component that makes multi-tenancy and high utilization possible at the scale of thousands of chips.

---

## The Islands of Compute

Pathways manages hardware organized into **islands**. An island is typically a collection of accelerators (like a TPU Pod or a GPU Cluster) that are connected by a high-speed, low-latency interconnect like **ICI** (Inter-Chip Interconnect) or **NVLink**.

Computations *within* an island are incredibly fast. Computations *between* islands happen over the standard **Datacenter Network (DCN)**, which is slower and higher latency. The Resource Manager's job is to intelligently map a user's program onto these islands to minimize DCN usage and maximize ICI performance.

---

## The Request: Virtual Device Sets

When a client wants to run a program, it doesn't ask for specific physical cores. Instead, it requests a **Virtual Device Set**.

```python
# The client requests a logical resource group
virtual_devices = pw.make_virtual_device_set()
slice_a = virtual_devices.add_slice(tpu_devices=256, shape=(16, 16))
```

This request specifies the **count** and **topology** (the shape of the mesh) required for the computation.

---

## The Mapping: Virtual to Physical

The Resource Manager takes those virtual requests and performs a **matching algorithm** against the cluster's current state.

1. **Topology Awareness**: If a researcher asks for a 16x16 mesh, the Resource Manager looks for a contiguous physical block of TPUs that supports that mesh shape with minimal communication hops.
2. **Load Balancing**: It spreads work across available islands to avoid hotspots.
3. **Elasticity**: It can add or remove backend compute resources dynamically. If an island's capacity increases, the Resource Manager can immediately start assigning more virtual slices to it.

Once the mapping is determined, the Resource Manager provides the client with a set of **physical device identifiers**.

---

## The "Transparent" Future

The paper hints at a future (now realized in systems like Cloud TPU) where this mapping is **transparently re-allocatable**.

- **Migration**: If a chip in Island A is failing, the Resource Manager should be able to "pause" the computation and "resume" it on Island B by re-mapping the virtual devices to new physical ones.
- **Suspend/Resume**: If a high-priority job arrives, the Resource Manager could temporarily reclaim an island from a lower-priority job, suspend it to disk, and resume it later when resources are free.

---

## Why Centralization Wins

By centralizing resource management, Pathways avoids the **"fragmentation" problem** of multi-controller systems.

In the old world:
- User A takes 512 TPUs.
- User B takes 512 TPUs.
- A 1024-TPU slot is "fragmented" even if neither user is currently using their TPUs.

In the Pathways world:
- The Resource Manager sees both users' needs.
- Because of **gang-scheduling** (see [Part 4d](04d_system_architecture_gang_scheduling.md)), it can interleave their computations on the same 512 TPUs with millisecond precision, ensuring the expensive hardware never sits idle between training steps.

![Resource Manager Lecture](./assets/04-system-architecture-lecture.png)

---

## Summary Table: Resource Manager's Responsibilities

| Responsibility | Description |
|----------------|-------------|
| **Device Inventory** | Keeps track of all physical accelerators and their interconnect topology. |
| **Virtualization** | Translates logical mesh requests into specific physical device lists. |
| **Backend Elasticity** | Handles the dynamic addition/removal of compute resources from the pool. |
| **Multi-Tenancy** | Coordinates how many different clients can share the same physical island. |

---

*Next up: [Part 4b — The Client: How One Machine Dispatches Work to Thousands →](04b_system_architecture_client.md)*
