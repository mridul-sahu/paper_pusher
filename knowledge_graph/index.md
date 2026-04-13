# Knowledge Graph — Master Index

> A structured, indexed, and searchable knowledge base of research papers.

---

## Papers

| ID | Title | Authors | Year | Venue | Directory |
|----|-------|---------|------|-------|-----------|
| `pathways-2022` | Pathways: Asynchronous Distributed Dataflow for ML | Barham, Chowdhery, Dean, Ghemawat, Hand, Hurt, Isard, Lim, Pang, Roy, Saeta, Schuh, Sepassi, El Shafey, Thekkath, Wu | 2022 | MLSys | [pathways/](pathways/) |

---

## Global Concept Index

Concepts are organized alphabetically. Each links to the paper-specific concept file.

| Concept | Papers | Tags |
|---------|--------|------|
| [Accelerator Utilization Metrics](pathways/concepts/accelerator_utilization_metrics.md) | pathways-2022 | `hardware`, `metrics`, `performance` |
| [Asynchronous Distributed Dataflow](pathways/concepts/asynchronous_distributed_dataflow.md) | pathways-2022 | `architecture`, `dataflow`, `execution-model` |
| [Gang Scheduling](pathways/concepts/gang_scheduling.md) | pathways-2022 | `scheduling`, `coordination`, `SPMD` |
| [JAX / TensorFlow Comparison](pathways/concepts/jax_tf_comparison.md) | pathways-2022 | `frameworks`, `comparison`, `programming-model` |
| [Parallel Asynchronous Dispatch](pathways/concepts/parallel_asynchronous_dispatch.md) | pathways-2022 | `optimization`, `dispatch`, `latency` |
| [PLAQUE Coordination](pathways/concepts/plaque_coordination.md) | pathways-2022 | `coordination`, `dataflow`, `infrastructure` |
| [Resource Manager](pathways/concepts/resource_manager.md) | pathways-2022 | `resource-management`, `virtualization`, `scheduling` |
| [Sharded Dataflow Graph](pathways/concepts/sharded_dataflow_graph.md) | pathways-2022 | `dataflow`, `sharding`, `scalability` |
| [Sharded Object Store](pathways/concepts/sharded_object_store.md) | pathways-2022 | `data-management`, `memory`, `HBM` |
| [Single-Controller Model](pathways/concepts/single_controller_model.md) | pathways-2022 | `architecture`, `control-plane`, `design-pattern` |
| [SPMD vs MPMD](pathways/concepts/spmd_vs_mpmd.md) | pathways-2022 | `parallelism`, `programming-model`, `computation` |

---

## Relationship Graph

Machine-readable relationships between concepts are encoded in Turtle format:
- [pathways/relationships.ttl](pathways/relationships.ttl)

---

## How to Search

1. **By keyword**: Use `grep -ri "<keyword>" knowledge_graph/` to search across all files.
2. **By tag**: Tags are listed in the concept index above and embedded in each concept file's YAML frontmatter.
3. **By relationship**: Parse `relationships.ttl` for typed edges between concepts.
4. **By paper**: Navigate into the paper's directory for all related content.
