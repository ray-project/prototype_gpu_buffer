
import time

import cupy as cp
import numpy as np

import ray
import ray.util.collective as collective


# Pass GPU objects by referece
# Base case -- p2p translate to ray.collective p2p call
# Advanced case -- group allredue translate to all reduce call
@ray.remote(num_gpus=1)
class Worker:
    def __init__(self, size=10000):
        self.size = size

        self.tensor = cp.random.rand(size, size, dtype=cp.float32)
        self.buffer = cp.random.rand(size, size, dtype=cp.float32)

    def get_tensor(self):
        return self.tensor

    def get_buffer(self):
        return self.buffer

    def re_init_tensors(self):
        self.tensor = cp.random.rand(self.size, self.size, dtype=cp.float32)
        self.buffer = cp.random.rand(self.size, self.size, dtype=cp.float32)

    def send_nccl(self, target_rank=0):
        # this call is blocking
        collective.send(self.tensor, target_rank, group_name="send_recv")

    def recv_nccl(self, src_rank=0):
        # this call is blocking
        collective.recv(self.buffer, src_rank, group_name="send_recv")

    def do_allreduce(self):
        # this call is blocking as well
        collective.allreduce(self.tensor, group_name="allreduce")
        # return self.buffer

    def destroy_collective_group(self, group_name="send_recv"):
        collective.destroy_collective_group(group_name=group_name)

TENSOR_SIZE = 20000
NUM_RUNS = 10

def run(stats, num_runs=NUM_RUNS):
    # Create two actors
    A = Worker.remote(size=TENSOR_SIZE)
    B = Worker.remote(size=TENSOR_SIZE)

    # Put A and B in a collective group
    collective.create_collective_group([A, B], 2, [0,1], group_name="send_recv")

    # let A to send a message to B; a send/recv has to be specified once at each worker
    # warm up
    ray.get([A.send_nccl.remote(target_rank=1), B.recv_nccl.remote(src_rank=0)])
    for _ in range(num_runs):
        start = time.time()
        ray.get([A.send_nccl.remote(target_rank=1), B.recv_nccl.remote(src_rank=0)])
        time_diff_ms = (time.time() - start) * 1000
        print(f"Pair ray.collective send-recv time: {time_diff_ms} ms.")
        stats.append(time_diff_ms)
        # Re-initialize value of tensors
        ray.get([A.re_init_tensors.remote(), B.re_init_tensors.remote()])

    ray.get([A.destroy_collective_group.remote(), B.destroy_collective_group.remote()])


stats = []
run(stats, num_runs=NUM_RUNS)
mean = round(np.mean(stats), 2)
std = round(np.std(stats), 2)
print(f"2D Tensor dim: {TENSOR_SIZE}, mean_ms: {mean}, std_ms: {std}, num_runs: {NUM_RUNS}")
# 2D Tensor dim: 20000, mean_ms: 1.73, std_ms: 0.89, num_runs: 10
