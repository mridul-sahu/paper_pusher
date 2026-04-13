# Part 3: The Programming Model — Write Python, Scale to Thousands of TPUs

> "The programming model... uses a simple Python API that will be familiar to users of current frameworks such as JAX or TensorFlow, while also supporting MPMD programs."
> — §3, Pathways paper

---

## Summary

This section of the paper details how a researcher actually "talks" to Pathways. The goal was to provide a familiar Python interface that hides the complexity of managing thousands of distributed accelerators, virtualizing hardware, and coordinating heterogeneous computations.

---

## Everything Is a Sharded Compiled Function

In Pathways, the fundamental unit of computation is a **compiled function** (usually from XLA). When a user writes a program, they aren't managing individual devices; they are managing **sharded computations**.

A sharded computation:
1. Runs on a set of **virtual devices**.
2. Expects inputs sharded in a specific way (e.g., partitioned by batch).
3. Produces outputs sharded in a specific way.

Pathways' magic is that it handles the **resharding** automatically. If Computation A produces data sharded one way, and Computation B (which depends on A) needs it sharded another way, the Pathways runtime inserts "transfer" nodes into its internal dataflow graph to move and shuffle the data between accelerators.

---

## The Python API

The paper describes a simple decorator-based approach:

```python
# 1. Define a standard compiled function (e.g., using JAX or TF)
@jax.jit
def train_step(state, batch):
    # Standard training logic here...
    return new_state

# 2. Assign it to virtual devices in a Pathways program
@pw.program
def distributed_training(dataset):
    # Initialize state on a virtual device set (e.g., a 2D mesh of 32 TPUs)
    state = pw.init_state(train_step, mesh_shape=(8, 4))
    
    for batch in dataset:
        # The runtime handles the sharding/dispatch of this call
        # across all 32 TPUs in parallel.
        state = train_step(state, batch)
```

### Key Differences from JAX

In multi-controller JAX, you would run this script on **every host**, and they would manually stay in sync. In Pathways, you run this script **once**, on a single client machine. The client machine traces the `@pw.program` block, builds a dataflow graph, and ships it to the Pathways runtime for execution.

---

## Virtual Device Virtualization

One of the most powerful features of the Pathways programming model is **virtual device sets**.

Researchers don't request "TPU core #45." Instead, they request high-level topologies:
- "Give me a 1D slice of 128 TPUs."
- "Give me a 2D mesh of 512 TPUs with a specific aspect ratio."

The **Resource Manager** (see [Part 4a](04a_system_architecture_resource_manager.md)) maps these virtual devices to physical ones. This allows the system to move computations around if a chip fails or if another user needs more resources, without the researcher changing a single line of Python code.

---

## From Python to MLIR

Under the hood, Pathways converts the Python `@pw.program` into a low-level **IR (Intermediate Representation)** using the **MLIR** framework.

This IR goes through several compiler passes:
1. **Lowering**: Converting high-level sharded ops into specific physical data transfers.
2. **Buffer Allocation**: Deciding exactly where in the TPU's memory (HBM) each tensor will live.
3. **Plaque Conversion**: Converting the IR into a dataflow graph that the coordination substrate (Plaque) can execute.

![Programming Model Lecture](./assets/03-programming-model-lecture.png)

---

## Why This Wins

By virtualizing the hardware and automating the coordination, the Pathways programming model does three things:

1. **Enables MPMD**: You can call `model_a()` on one set of 128 devices and `model_b()` on another set of 128 devices in the same program. Pathways handles the communication between them.
2. **Simplifies Research**: Researchers focus on the model logic, not the `AllToAll` synchronization logic.
3. **Improves Portability**: You can write a program for a generic "512-TPU mesh" and run it on a v3 pod today and a v4 pod tomorrow without changes.

---

*Next up: [Part 4a — The Resource Manager: Virtualizing the World's Most Powerful Hardware →](04a_system_architecture_resource_manager.md)*
