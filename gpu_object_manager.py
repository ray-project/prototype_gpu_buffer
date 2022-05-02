import uuid
import cupy as cp
import time

import ray
import ray.util.collective as collective


class GPUObjectRef:
    def __init__(self, uuid, location_rank, shape, size, dtype):
        self.uuid = uuid
        self.location_rank = location_rank
        self.shape = shape
        self.size = size
        self.dtype = dtype

    def __str__(self):
        return f"{self.__dict__}"

@ray.remote(num_gpus=0.1, num_cpus=1)
class GPUObjectManager:
    """Coordinate a list of GPUDeviceActor(s)"""

    def __init__(self, collective_group_name = "default"):
        self.buffers = {}
        self.send_tasks = {}
        self.recv_tasks = {}

        self.object_ref_to_device = {}
        self.device_to_object_size = {}

        self.workers = []
        self.collective_group_name = collective_group_name

    def init_gpu_actor_group(self):
        num_gpus = int(ray.cluster_resources()["GPU"])
        for _ in range(num_gpus):
            self.workers.append(GPUDeviceActor.remote(self.collective_group_name))
        print(f">>>> num_gpus: {num_gpus}")
        print(f">>>> self.workers: {self.workers}")
        collective.create_collective_group(
            self.workers,
            num_gpus,
            list(range(num_gpus)),
            group_name="send_recv",
        )

    def put(self, gpu_object: cp.ndarray):
        # Base case: Put new tensor on a GPU device
        # TODO(jiao): This should be done without a cpu -> gpu copy later
        worker_rank = 0
        ray.get(self.workers[worker_rank].init_data.remote(gpu_object))

        # Transfer case: Send a tensor to another GPU

        gpu_obj_ref = GPUObjectRef(
            str(uuid.uuid4()),
            worker_rank,
            gpu_object.shape,
            gpu_object.size,
            gpu_object.dtype
        )
        return gpu_obj_ref

    def get(self, gpu_object_ref):
        print(f">>>>> Calling get on gpu_object_ref: {gpu_object_ref}")
        src_worker_rank = gpu_object_ref.location_rank
        dst_worker_rank = 1
        print(f">>>>> src_worker_rank: {src_worker_rank}")
        print(f">>>>> dst_worker_rank: {dst_worker_rank}")
        return ray.get([
            self.workers[src_worker_rank].send.remote(dst_worker_rank),
            self.workers[dst_worker_rank].recv.remote(src_worker_rank)
        ])


    def collective_pattern_match(self):
        """Identify and map to advanced collective patterns, ex: AllReduce
        """
        pass

    def _find_slot(self):
        """Given a new object to be placed, find a device to place it."""
        pass


@ray.remote(num_gpus=1)
class GPUDeviceActor:
    def __init__(self, collective_group_name):
        # Put some dummy data
        self.data = cp.random.rand(1, 1, dtype=cp.float32)
        self.data_buffer = cp.random.rand(1, 1, dtype=cp.float32)
        self.collective_group_name = collective_group_name

    def init_data(self, gpu_object):
        self.data = gpu_object
        self.data_buffer = gpu_object

    def send(self, target_rank):
        collective.send(
            self.data, target_rank, group_name=self.collective_group_name
        )

    def recv(self, src_rank):
        collective.recv(
            self.data_buffer, src_rank, group_name=self.collective_group_name
        )

TENSOR_SIZE = 20000

gpu_obj_manager = GPUObjectManager.remote(collective_group_name="send_recv")
ray.get(gpu_obj_manager.init_gpu_actor_group.remote())

tensor = cp.random.rand(TENSOR_SIZE, TENSOR_SIZE, dtype=cp.float32)

# @ray.remote
# def gpu_task():

# This can be pushed to ray level and become ray.put(tensor) / ray.get(ref)
obj_ref = gpu_obj_manager.put.remote(tensor)

start = time.time()
ray.get(gpu_obj_manager.get.remote(obj_ref))
time_diff_ms = (time.time() - start) * 1000
print(f"2D Tensor dim: {TENSOR_SIZE}, mean_ms: {time_diff_ms}")


# 1 - Create a tensor and put it on a GPU, return ref
# 2 - Pass that ref to another worker that needs it
# 3 - Ensure it uses NCCL's send/recv