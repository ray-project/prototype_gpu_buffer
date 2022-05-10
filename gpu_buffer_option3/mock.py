import uuid
import numpy as cp
import ray
import uuid
from ray import ObjectRef


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
