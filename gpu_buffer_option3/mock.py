import uuid
import numpy as cp
import ray
import uuid
from ray import ObjectRef


# Locality for GpuObjects?
# 1. Datasets only - we can extend Datasets actor pool strategy to take in locality hints from previous stages.
# 2. more general Ray core solution - given an ObjectRef, we can tell that it contains a GpuObjectRef, and we can get the actor handle that created the GPU buffer.
#    a. We use ray.get to figure out that it's a GpuObjectRef and then extract a handle to the actor.
#    b. Given any ObjectRef, Ray is aware that it's actually a GpuObjectRef without having to call ray.get.


# Options for GPU buffer passing:
# 1. Transfer through plasma.
# 2. Keep the GPU buffer local to the actor - requires scheduling dependent task on the same actor that created the buffer.
# 3. Keep the buffer on the same GPU, but send to a different actor via IPC.
# 4. Send to a different actor via NCCL/collectives.


class GpuObjectRef:
    """Presents a reference to GPU buffer."""

    def __init__(self, id: ObjectRef, group: uuid.UUID):
        self.id = id
        self.group = group


class GpuActorBase:
    def __init__(self):
        self.group = None

    def _setup(self, group):
        assert self.group is None, "Gpu actor could only belong to one group."
        self.group = group

    def put_gpu_buffer(self, object) -> GpuObjectRef:
        ref = GpuObjectRef(ray.put(object), self.group)
        return ref

    def get_gpu_buffer(self, ref: GpuObjectRef):
        assert self.group == ref.group, f"{self.group} is different from {ref.group}"
        numpy_obj = ray.get(ref.id)
        return numpy_obj


def setup_transfer_group(actors):
    group_uuid = uuid.uuid4()
    ray.get([actor._setup.remote(group_uuid) for actor in actors])
