
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
       with Device(1):
           self.send2 = cp.ones((size, size), dtype=cp.float32) * 2
       with Device(0):
           self.recv1 = cp.ones((size, size), dtype=cp.float32)
       with Device(1):
           self.recv2 = cp.ones((size, size), dtype=cp.float32) * 2

   def setup(self, world_size, rank):
       self.rank = rank
       collective.init_collective_group(world_size, rank, backend="nccl", group_name="177")
       return True

   def allreduce_call(self):
       collective.allreduce_multigpu([self.send1, self.send2], group_name="177")
       return [self.send1, self.send2]

   def p2p_call(self):
       if self.rank == 0:
          collective.send_multigpu(self.send1 * 2, 1, 1, group_name="177")
       else:
          collective.recv_multigpu(self.recv2, 0, 0, group_name="177")
       return self.recv2

# Note that the world size is 2 but there are 4 GPUs.
num_workers = 2
workers = []
init_rets = []
for i in range(num_workers):
   w = Worker.remote(size=5000)
   workers.append(w)
   init_rets.append(w.setup.remote(num_workers, i))
a = ray.get(init_rets)
print(a)

start = time.time()
results = ray.get([w.allreduce_call.remote() for w in workers])
print(f"allreduce_call time: {(time.time() - start) * 1000} ms.")

start = time.time()
results = ray.get([w.p2p_call.remote() for w in workers])
print(f"p2p_call time: {(time.time() - start) * 1000} ms.")
