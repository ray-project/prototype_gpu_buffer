import ray
import ray.util.collective as collective

import cupy as cp
from cupy.cuda import Device


@ray.remote(num_gpus=2)
class Worker:
   def __init__(self):
       with Device(0):
           self.send1 = cp.ones((4, ), dtype=cp.float32)
       with Device(1):
           self.send2 = cp.ones((4, ), dtype=cp.float32) * 2
       with Device(0):
           self.recv1 = cp.ones((4, ), dtype=cp.float32)
       with Device(1):
           self.recv2 = cp.ones((4, ), dtype=cp.float32) * 2

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
   w = Worker.remote()
   workers.append(w)
   init_rets.append(w.setup.remote(num_workers, i))
a = ray.get(init_rets)
print(a)
results = ray.get([w.allreduce_call.remote() for w in workers])
print(results)
results = ray.get([w.p2p_call.remote() for w in workers])
print(results)