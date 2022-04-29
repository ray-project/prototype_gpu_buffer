# %%
import os
import time

import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

import ray

if not ray.is_initialized():
    ray.init(address="auto")

# ========== Local matmul profile ===========
# key = random.PRNGKey(0)
# size = 5000
# x = random.normal(key, (size, size), dtype=jnp.float32)

# start = time.time()
# jnp.dot(x, x.T).block_until_ready()  # runs on the GPU
# print(f"Local: time spent, {(time.time() - start) * 1000} ms")

# ========= Launch matmul on 4 GPU tasks to saturate ========
# @ray.remote(num_gpus=1)
# def use_gpu():
#     print("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
#     print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
#     key = random.PRNGKey(0)
#     size = 5000
#     x = random.normal(key, (size, size), dtype=jnp.float32)

#     start = time.time()
#     jnp.dot(x, x.T).block_until_ready()  # runs on the GPU
#     print(f"Ray task, time spent: {(time.time() - start) * 1000} ms")


# ray.get([use_gpu.remote() for _ in range(4)])
# ray.get_gpu_ids()

# ============== Creating and sending tensors across GPUs =================
@ray.remote(num_gpus=1)
class GPUActor:
    def __init__(self):
        print("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
        print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))

        self.tensor = None
        self.key = random.PRNGKey(0)

    def create(self, size) -> ray.ObjectRef:
        print(f"Creating tensor with size: {size * size}")
        self.tensor = random.normal(self.key, (size, size), dtype=jnp.float32)
        return ray.put(self.tensor)

    def receive(self, tensor):
        self.tensor = tensor
        print(f"Received tensor of size {self.tensor.size}")

    def send(self, actor_name: str):
        actor = ray.get_actor(actor_name)
        ray.get(actor.receive.remote(self.tensor))
        print(f"Sent tensor of size {self.tensor.size} to {actor_name}")

a1 = GPUActor.options(name="GPU_0").remote()
a2 = GPUActor.options(name="GPU_1").remote()

a1.create.remote(5000)
start = time.time()
ray.get(a1.send.remote("GPU_1"))
print(f"Time taken: {(time.time() - start) * 1000} ms")