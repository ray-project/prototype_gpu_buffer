import time

import cupy as cp
from cupy.cuda import Device

import ray
import ray.util.collective as collective


@ray.remote(num_gpus=2)
class Worker:
    def __init__(self, size=4):
        with Device(0):
            self.send1 = cp.ones((size, size), dtype=cp.float32)
            print(f"Init send1: {self.send1}, , gpu_ids: {ray.get_gpu_ids()}")
        with Device(1):
            self.recv2 = cp.ones((size, size), dtype=cp.float32) * 4
            print(f"Init recv2: {self.recv2}, gpu_ids: {ray.get_gpu_ids()}")

    def setup(self, world_size, rank):
        self.rank = rank
        collective.init_collective_group(
            world_size, rank, backend="nccl", group_name="177"
        )
        return True

    def p2p_call(self):
        if self.rank == 0:
            print(f"Before send1: {self.send1 * 2}, rank: {self.rank}, gpu_ids: {ray.get_gpu_ids()}")
            collective.send_multigpu(self.send1 * 2, 1, 1, group_name="177")
            print(f"After send1: {self.send1 * 2}, rank: {self.rank}, gpu_ids: {ray.get_gpu_ids()}")
        else:
            print(f"Before recv2: {self.recv2}, rank: {self.rank}, gpu_ids: {ray.get_gpu_ids()}")
            collective.recv_multigpu(self.recv2, 0, 0, group_name="177")
            print(f"After recv2: {self.recv2}, rank: {self.rank}, gpu_ids: {ray.get_gpu_ids()}")
        return self.recv2

# Note that the world size is 2 but there are 4 GPUs.
num_workers = 2
workers = []
init_rets = []
for i in range(num_workers):
    w = Worker.remote(size=4)
    workers.append(w)
    init_rets.append(w.setup.remote(num_workers, i))
a = ray.get(init_rets)
print(f"Init results: {a}")

start = time.time()
results = ray.get([w.p2p_call.remote() for w in workers])
print(f"p2p_call time: {(time.time() - start) * 1000} ms.")
print(results)
print(f"result length: {len(results)}")
