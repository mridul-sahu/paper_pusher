# Resource Manager

---
id: resource-manager
paper: pathways-2022
section: "§4.1 (Resource Manager)"
tags:
  - resource-management
  - virtualization
  - scheduling
  - virtual-devices
related:
  - single_controller_model
  - gang_scheduling
  - accelerator_utilization_metrics
---

## Definition

The Pathways **Resource Manager** is a global, centralized component responsible for managing all accelerator devices across all islands. It provides a virtual device abstraction that decouples user programs from physical hardware.

## Capabilities

1. **Virtual slices**: Clients request subsets of an island's accelerators with specific 2D or 3D mesh shapes to suit their communication patterns.
2. **Virtual-to-physical mapping**: The resource manager dynamically assigns physical devices to virtual devices, satisfying constraints on interconnect topology and memory capacity.
3. **Dynamic scaling**: Backend compute resources can be added and removed dynamically.
4. **Load balancing**: Current implementation uses a simple heuristic that statically balances load by spreading computations across available devices.

## Virtual Device Abstraction

```python
def get_devices(n):
    """Allocates n virtual TPU devices on an island."""
    device_set = pw.make_virtual_device_set()
    return device_set.add_slice(tpu_devices=n).tpus
```

- Users express computations in terms of **virtual devices**.
- The system handles all **data movement and resharding** between dependent computations automatically.
- Programs can be **re-lowered** if the resource manager changes the virtual-to-physical mapping.

## Future Capabilities

- **Transparent suspend/resume**: Client's virtual devices temporarily reclaimed without user cooperation.
- **Migration**: Virtual devices reassigned to different physical devices.
- **Sophisticated allocation**: Algorithms considering all client resource requirements and system state.
- **Diverse resource types**: Device memory, host memory, ICI/DCN/PCIe bandwidth.
- **Multi-tenancy policies**: Priorities, performance isolation, access control, resource accounting at millisecond timescales.

## Paper Reference

> "PATHWAYS has a 'resource manager' which is responsible for the centralized management of devices across all of the islands." — §4.1
