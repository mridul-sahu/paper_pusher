# Pathways: Asynchronous Distributed Dataflow for ML
### A Deep-Dive Blog Series

> Based on the paper by Barham et al. (2022) — Google Brain

---

This series is a comprehensive, section-by-section technical breakdown of the **Pathways** paper — the system that replaced Google's multi-controller ML infrastructure with a single-controller architecture capable of gang-scheduling heterogeneous computations across thousands of accelerators.

Each post is grounded in the paper's text and includes hand-drawn lecture-style diagrams.

---

## The Series

| # | Post | Paper Section | Key Question |
|---|------|:------------:|-------------|
| 1 | [Introduction](01_introduction.md) | §1 | Why was SPMD breaking? |
| 2 | [Design Motivation](02_design_motivation.md) | §2 | Multi-controller vs. single-controller? |
| 3 | [Programming Model](03_programming_model.md) | §3, App. B/C | How does the user interact with Pathways? |
| 4a | [Resource Manager](04a_system_architecture_resource_manager.md) | §4.1 | How is hardware managed and virtualized? |
| 4b | [Client Architecture](04b_system_architecture_client.md) | §4.2 | How does the client avoid being a bottleneck? |
| 4c | [Plaque Coordination](04c_system_architecture_coordination.md) | §4.3 | How is execution coordinated at scale? |
| 4d | [Gang Scheduling](04d_system_architecture_gang_scheduling.md) | §4, App. A.5 | How are deadlocks prevented? |
| 4e | [Asynchronous Dispatch](04e_system_architecture_asynchronous_dispatch.md) | §4, App. B | How is dispatch latency eliminated? |
| 4f | [Data Management](04f_system_architecture_data_management.md) | §4 | How are terabytes moved efficiently? |
| 5 | [Evaluation](05_evaluation.md) | §5 | Does it actually work? The numbers. |
| 6 | [Related Work & Future](06_related_work_and_future_directions.md) | §6–7 | What comes next? |
| 7 | [Appendices: Hardware](07_appendices.md) | App. A–C | TPU vs GPU, ICI vs DCN |

---

## Visual Assets

All hand-drawn lecture diagrams are in [`assets/`](./assets/).

---

*Start reading: [Part 1 — Introduction →](01_introduction.md)*
