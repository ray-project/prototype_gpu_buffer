
import time

import cupy as cp
import numpy as np

import ray

# Pass GPU objects by referece
# Base case -- p2p translate to ray.collective p2p call
# Advanced case -- group allredue translate to all reduce call
@ray.remote(num_gpus=1)
class Worker:
    def __init__(self, size=10000):
        self.tensor = cp.random.rand(size, size, dtype=cp.float32)
        self.buffer = cp.zeros((size, size), dtype=cp.float32)

    def get_tensor(self):
        return self.tensor

    def get_buffer(self):
        return self.buffer

    def send_ray_object_store(self, target_rank=0):
        obj_ref = ray.put(self.tensor)
        return obj_ref

    def recv_ray_object_store(self, obj_ref, src_rank=0):
        self.buffer = ray.get(obj_ref)


TENSOR_SIZE = 20000
NUM_RUNS = 10
def run():
    # Create two actors
    A = Worker.remote(size=TENSOR_SIZE)
    B = Worker.remote(size=TENSOR_SIZE)

    start = time.time()
    obj_ref = A.send_ray_object_store.remote()
    ray.get(B.recv_ray_object_store.remote(obj_ref))
    time_diff_ms = (time.time() - start) * 1000
    print(f"Pair ray object store send-recv time: {time_diff_ms} ms.")

    return time_diff_ms

# Warm up
run()
stats = []
for _ in range(NUM_RUNS):
    stats.append(run())

mean = round(np.mean(stats), 2)
std = round(np.std(stats), 2)
print(f"2D Tensor dim: {TENSOR_SIZE}, mean_ms: {mean}, std_ms: {std}, num_runs: {NUM_RUNS}")
# 2D Tensor dim: 20000, mean_ms: 3867.72, std_ms: 64.1, num_runs: 10


# ============== Scratch pad =============

# def ring_all_reduce(send, recv):
#     print("A")
    # rank = dist.get_rank()
    # size = dist.get_world_size()
    # send_buff = send.clone()
    # recv_buff = send.clone()
    # accum = send.clone()

    # left = ((rank - 1) + size) % size
    # right = (rank + 1) % size

    # for i in range(size - 1):
    #     if i % 2 == 0:
    #         # Send send_buff
    #         send_req = dist.isend(send_buff, right)
    #         dist.recv(recv_buff, left)
    #         accum[:] += recv_buff[:]
    #     else:
    #         # Send recv_buff
    #         send_req = dist.isend(recv_buff, right)
    #         dist.recv(send_buff, left)
    #         accum[:] += send_buff[:]
    #     send_req.wait()
    # recv[:] = accum[:]

# ring_all_reduce()