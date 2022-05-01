
import time

import cupy as cp
from cupy.cuda import Device

import ray
import ray.util.collective as collective


# Pass GPU objects by referece
# Base case -- p2p translate to ray.collective p2p call
# Advanced case -- group allredue translate to all reduce call
@ray.remote(num_gpus=1)
class Worker:
    def __init__(self, init_val=1, size=10000):
        self.tensor = cp.ones((size, size), dtype=cp.float32) * init_val
        self.buffer = cp.zeros((size, size), dtype=cp.float32)

    def get_tensor(self):
        return self.tensor

    def get_buffer(self):
        return self.buffer

    def send_nccl(self, target_rank=0):
        # this call is blocking
        collective.send(self.tensor, target_rank, group_name="send_recv")

    def send_ray_object_store(self, target_rank=0):
        obj_ref = ray.put(self.tensor)
        return obj_ref

    def recv_nccl(self, src_rank=0):
        # this call is blocking
        collective.recv(self.buffer, src_rank, group_name="send_recv")

    def recv_ray_object_store(self, obj_ref, src_rank=0):
        self.buffer = ray.get(obj_ref)

    def do_allreduce(self):
        # this call is blocking as well
        collective.allreduce(self.tensor, group_name="allreduce")
        # return self.buffer

TENSOR_SIZE = 10000
# Create two actors
A = Worker.remote(init_val=1, size=TENSOR_SIZE)
B = Worker.remote(init_val=2, size=TENSOR_SIZE)
C = Worker.remote(init_val=1, size=TENSOR_SIZE)
D = Worker.remote(init_val=2, size=TENSOR_SIZE)

# Put A and B in a collective group
collective.create_collective_group([A, B], 2, [0,1], group_name="send_recv")


# let A to send a message to B; a send/recv has to be specified once at each worker
start = time.time()
ray.get([A.send_nccl.remote(target_rank=1), B.recv_nccl.remote(src_rank=0)])
print(f"Pair ray.collective send-recv time: {(time.time() - start) * 1000} ms.")

print(ray.get(B.get_tensor.remote()))
print(ray.get(B.get_buffer.remote()))

start = time.time()
obj_ref = C.send_ray_object_store.remote()
ray.get(D.recv_ray_object_store.remote(obj_ref))
print(f"Pair ray object store send-recv time: {(time.time() - start) * 1000} ms.")

print(ray.get(D.get_tensor.remote()))
print(ray.get(D.get_buffer.remote()))


# ============ All reduce with 2 GPU on 2 workers ============

# C = Worker.remote()
# D = Worker.remote()
# collective.create_collective_group([C, D], 2, [0,1], group_name="allreduce")

# start = time.time()
# ray.get([C.do_allreduce.remote(), D.do_allreduce.remote()])
# print(f"Pair allreduce time: {(time.time() - start) * 1000} ms.")

# ============== Scratch pad =============

# SIZE = 10
# tensor_refs = [ray.put(i) for i in range(4)]
# workers = [Worker.remote(i, size=SIZE) for i in range(4)]

# for i in range(4):
#     workers[i].recv.remote([tensor_refs[i]])


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