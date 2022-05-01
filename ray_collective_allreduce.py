import time

import cupy as cp
from cupy.cuda import Device

import ray
import ray.util.collective as collective


@ray.remote(num_gpus=4)
class Worker:
    def __init__(self, size=4):
        with Device(0):
            self.one = cp.ones((size, size), dtype=cp.float32)
        with Device(1):
            self.two = cp.ones((size, size), dtype=cp.float32) * 2
        with Device(2):
            self.three = cp.ones((size, size), dtype=cp.float32) * 3
        with Device(3):
            self.four = cp.ones((size, size), dtype=cp.float32) * 4

    def setup(self, world_size, rank):
        self.rank = rank
        collective.init_collective_group(
            world_size, rank, backend="nccl", group_name="177"
        )
        return True

    def allreduce_call(self):
        start = time.time()
        collective.allreduce_multigpu(
            [self.one, self.two, self.three, self.four], group_name="177"
        )
        print(f"Ray.collective allreduce_call time: {(time.time() - start) * 1000} ms.")
        # print(self.one)


# Note that the world size is 1 but there are 4 GPUs.
num_workers = 1
workers = []
init_rets = []
for i in range(num_workers):
    w = Worker.remote(size=10)
    workers.append(w)
    init_rets.append(w.setup.remote(num_workers, i))
a = ray.get(init_rets)
print(f"Init results: {a}")

start = time.time()
results = ray.get([w.allreduce_call.remote() for w in workers])
print(f"Ray.get with allreduce_call time: {(time.time() - start) * 1000} ms.")
# print(f"resulsts: {results}")

# 10 x 10 tensor    - 1270.2579498291016 ms.
# 10k x 10k tensor  - 1279.3364524841309 ms. (+9ms)
