
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
    def __init__(self, size=10):
        self.buffer = cp.ones((size,), dtype=cp.float32)

    def get_buffer(self):
        return self.buffer

    def do_send(self, target_rank=0):
        # this call is blocking
        collective.send(self.buffer, target_rank, group_name="send_recv")

    def do_recv(self, src_rank=0):
        # this call is blocking
        collective.recv(self.buffer, src_rank, group_name="send_recv")

    def do_allreduce(self):
        # this call is blocking as well
        collective.allreduce(self.buffer)
        return self.buffer

# Create two actors
A = Worker.remote()
B = Worker.remote()

# Put A and B in a collective group
collective.create_collective_group([A, B], 2, [0,1], group_name="send_recv")

# let A to send a message to B; a send/recv has to be specified once at each worker
ray.get([A.do_send.remote(target_rank=1), B.do_recv.remote(src_rank=0)])

# An anti-pattern: the following code will hang, because it does instantiate the recv side call
ray.get([A.do_send.remote(target_rank=1)])



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