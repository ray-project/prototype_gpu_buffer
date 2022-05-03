import time
import ray
import os
import numpy as np

N_ITER = 100
tensor_size = 2 ** 30 // 4


@ray.remote(num_gpus=1)
class RemoteActor:
    def __init__(self, name, device_index, have_tensor=False):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7,8"
        import torch
        from torch.utils.cpp_extension import load
        self.device_index = device_index
        import torch_cuda_comm
        self.torch_cuda_comm = torch_cuda_comm
        # print(name, os.environ['CUDA_VISIBLE_DEVICES'], flush=True)
        torch.cuda.init()
        if have_tensor:
            self.t = torch.zeros(tensor_size).cuda(device_index)
        else:
            self.t = torch.zeros(1).cuda(device_index)
        self.name = name
        self.perf = {"get_token": [], "copy_tensor_from_token": [], "add": []}
        print(self.name, "init finshed", flush=True)

    def send(self):
        # print(self.name, "send", flush=True)
        start = time.time()
        handle = self.torch_cuda_comm.get_token(self.t)
        self.perf['get_token'].append(time.time() - start)
        return handle

    def recv(self, token):
        start = time.time()
        self.t = self.torch_cuda_comm.copy_tensor_from_token(token, self.device_index)
        self.perf['copy_tensor_from_token'].append(time.time() - start)
        start = time.time()
        self.t = self.t + 1
        import torch
        torch.cuda.synchronize(device=self.device_index)
        self.perf['add'].append(time.time() - start)
        # print(self.name, "recv", flush=True)

    def reset(self):
        self.perf['add'] = []
        self.perf['get_token'] = []
        self.perf['copy_tensor_from_token'] = []
        self.t.zero_()
        import torch
        torch.cuda.synchronize(device=self.device_index)

    def get_perf(self):
        return self.perf

    def get(self):
        if self.t is None:
            return None
        else:
            return self.t[0].item()


def pingpong(device_index, peer_index):
    a0 = RemoteActor.remote("a0", device_index, have_tensor=True)
    a1 = RemoteActor.remote("a1", peer_index, have_tensor=False)

    for i in range(10):
        if i % 2 == 0:
            handle = a0.send.remote()
            a1.recv.remote(handle)
        else:
            handle = a1.send.remote()
            a0.recv.remote(handle)

    ray.get(a1.get.remote())
    ray.get(a0.reset.remote())
    ray.get(a1.reset.remote())
    print("start", flush=True)
    start = time.time()
    for i in range(N_ITER):
        if i % 2 == 0:
            handle = a0.send.remote()
            handle = a1.recv.remote(handle)
        else:
            handle = a1.send.remote()
            handle = a0.recv.remote(handle)

    s = ray.get(a0.get.remote())
    print("average time", (time.time() - start) / N_ITER, flush=True)
    print("result", s, flush=True)

    perf0 = ray.get(a0.get_perf.remote())
    perf1 = ray.get(a1.get_perf.remote())
    op_time = np.mean(perf0['add'] + perf1['add'])
    get_token_time = np.mean(perf0['get_token'] + perf1['get_token'])
    copy_tensor_time = np.mean(perf0['copy_tensor_from_token'] + perf1['copy_tensor_from_token'])
    print(f'operation time: {op_time}\nget_token_time: {get_token_time}\ncopy_tensor_time: {copy_tensor_time}')


def baseline(device_index, peer_index):
    import torch
    a = torch.zeros(tensor_size).cuda(device_index)
    b = None
    for i in range(10):
        if i % 2 == 0:
            b = a.cuda(peer_index, non_blocking=False)
            b += 1
        else:
            a = b.cuda(device_index, non_blocking=False)
            a += 1
    start = time.time()
    for i in range(N_ITER):
        if i % 2 == 0:
            b = a.cuda(peer_index, non_blocking=False)
            b += 1
        else:
            a = b.cuda(device_index)
            a += 1

    torch.cuda.synchronize(device=device_index)
    torch.cuda.synchronize(device=peer_index)
    during = (time.time() - start) / N_ITER
    print(f'PyTorch tensor copy time: {during}')


if __name__ == '__main__':
    ray.init()
    for i in range(4, 8):
        for j in range(4, 8):
            print("## ", i, j)
            baseline(i, j)
            pingpong(i, j)
