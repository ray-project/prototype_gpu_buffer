# %%
import os
import time

import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

import ray

if not ray.is_initialized():
    ray.init(address="auto")


# key = random.PRNGKey(0)
# size = 5000
# x = random.normal(key, (size, size), dtype=jnp.float32)

# start = time.time()
# jnp.dot(x, x.T).block_until_ready()  # runs on the GPU
# print(f"Local: time spent, {(time.time() - start) * 1000} ms")

@ray.remote(num_gpus=1)
def use_gpu():
    print("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
    print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    key = random.PRNGKey(0)
    size = 5000
    x = random.normal(key, (size, size), dtype=jnp.float32)

    start = time.time()
    jnp.dot(x, x.T).block_until_ready()  # runs on the GPU
    print(f"Ray task, time spent: {(time.time() - start) * 1000} ms")


ray.get([use_gpu.remote() for _ in range(4)])
ray.get_gpu_ids()