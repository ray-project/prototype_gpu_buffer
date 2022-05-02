import cupy as cp

import ray
import ray.util.collective as collective


class GPUObjectRef:
    def __init__(self, uuid, src_rank, shape, dtype):
        self.uuid = uuid
        self.src_rank = src_rank
        self.shape = shape
        self.dtype = dtype


class GPUObjectManager:
    """Coordinate a list of GPUDeviceActor(s)"""

    def __init__(self, collective_group_name = "default"):
        self.buffers = {}
        self.send_tasks = {}
        self.recv_tasks = {}

        self.object_ref_to_device = {}
        self.workers = []
        self.collective_group_name = collective_group_name

    def init_gpu_actor_group(self):
        num_gpus = int(ray.cluster_resources()["GPU"])
        self.workers.append(GPUDeviceActor.remote(self.collective_group_name))

        collective.create_collective_group(
            [self.workers],
            num_gpus,
            list(range(num_gpus)),
            group_name="send_recv",
        )

    def put(self, gpu_object):
        pass

    def get(self, gpu_object_ref):
        pass

    def _pattern_match(self):
        pass


class GPUDeviceActor:
    def __init__(self, collective_group_name):
        self.data = None
        self.collective_group_name = collective_group_name

    def send(self, target_rank):
        collective.send(
            self.data, target_rank, group_name=self.collective_group_name
        )

    def recv(self, src_rank):
        collective.recv(
            self.data, src_rank, group_name=self.collective_group_name
        )

SIZE = 100

gpu_obj_manager = GPUObjectManager(collective_group_name="send_recv")
gpu_obj_manager.init_gpu_actor_group()

tensor = cp.random.rand(SIZE, SIZE, dtype=cp.float32)

# @ray.remote
# def gpu_task():

# This can be pushed to ray level and become ray.put(tensor)
obj_ref = gpu_obj_manager.put(tensor)
gpu_obj_manager.get(obj_ref)
