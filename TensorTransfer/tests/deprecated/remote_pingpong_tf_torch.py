import time
import ray
import numpy as np

N_ITER = 10
tensor_size = 2 ** 30 // 4


@ray.remote(num_gpus=1)
class RemoteActor:
    def __init__(self, name, device_index, have_tensor=False):
        self.is_torch = have_tensor
        import torch
        import tensorflow as tf
        self.device_index = device_index
        import node2node
        self.p2p_service = node2node.TransferService()
        # print(name, os.environ['CUDA_VISIBLE_DEVICES'], flush=True)
        torch.cuda.init()
        if have_tensor:
            self.t = torch.zeros(tensor_size).cuda(device_index)
        else:
            with tf.device('/device:gpu:' + str(device_index)):
                self.t = tf.zeros(1)
        self.name = name
        self.perf = {"get_handle": [], "send_tensor": [],
                     "recv_tensor": [], "add": []}
        print(self.name, "init finished", flush=True)

    def get_addr_info(self):
        return self.p2p_service.address, self.p2p_service.port

    def get_handle(self):
        start = time.time()
        handle = self.p2p_service.get_handle(self.t)
        self.perf['get_handle'].append(time.time() - start)
        return handle

    def send(self):
        # print(self.name, "send", flush=True)
        start = time.time()
        handle = self.p2p_service.send_tensor(self.t)
        self.perf['send_tensor'].append(time.time() - start)
        return handle

    def recv(self, handle):
        start = time.time()
        if self.is_torch:
            backend = 'torch'
        else:
            backend = 'tensorflow'
        self.t = self.p2p_service.get_tensor_from_handle(
            handle, 'cuda', self.device_index, backend=backend)
        self.perf['recv_tensor'].append(time.time() - start)
        start = time.time()
        self.t = self.t + 1
        # print(self.name, self.t)
        if self.is_torch:
            import torch
            torch.cuda.synchronize(device=self.device_index)
        self.perf['add'].append(time.time() - start)
        # print(self.name, "recv", flush=True)

    def reset(self):
        self.perf['add'] = []
        self.perf['get_handle'] = []
        self.perf['send_tensor'] = []
        self.perf['recv_tensor'] = []
        if self.is_torch:
            self.t.zero_()
            import torch
            torch.cuda.synchronize(device=self.device_index)
        else:
            import tensorflow as tf
            self.t = tf.zeros_like(self.t)

    def get_perf(self):
        return self.perf

    def get(self):
        if self.t is None:
            return None
        else:
            if self.is_torch:
                return self.t.min().item(), self.t.max().item()
            else:
                import tensorflow as tf
                return float(tf.math.reduce_min(self.t)), float(tf.math.reduce_max(self.t))


def pingpong(device_index, peer_index):
    a0 = RemoteActor.remote("a0", device_index, have_tensor=True)
    a1 = RemoteActor.remote("a1", peer_index, have_tensor=False)

    for i in range(2):
        if i % 2 == 0:
            handle = a0.get_handle.remote()
            a0.send.remote()
            a1.recv.remote(handle)
        else:
            handle = a1.get_handle.remote()
            a1.send.remote()
            a0.recv.remote(handle)

    ray.get(a1.get.remote())
    ray.get(a0.reset.remote())
    ray.get(a1.reset.remote())
    print("start", flush=True)
    start = time.time()
    for i in range(N_ITER):
        if i % 2 == 0:
            handle = a0.get_handle.remote()
            a0.send.remote()
            a1.recv.remote(handle)
        else:
            handle = a1.get_handle.remote()
            a1.send.remote()
            a0.recv.remote(handle)

    s = ray.get(a0.get.remote())
    print("average time", (time.time() - start) / N_ITER, flush=True)
    print("result", s, flush=True)

    perf0 = ray.get(a0.get_perf.remote())
    perf1 = ray.get(a1.get_perf.remote())
    op_time = np.mean(perf0['add'] + perf1['add'])
    get_handle_time = np.mean(perf0['get_handle'] + perf1['get_handle'])
    send_tensor_time = np.mean(perf0['send_tensor'] + perf1['send_tensor'])
    recv_tensor_time = np.mean(perf0['recv_tensor'] + perf1['recv_tensor'])
    print(f"""operation time: {op_time}
get_handle_time: {get_handle_time}
send_tensor_time: {send_tensor_time}
recv_tensor_time: {recv_tensor_time}""")


if __name__ == '__main__':
    ray.init(address='auto')
    # print("running baseline...")
    # ray.get(baseline.remote(0, 0))
    print("running remote ping-pong test...")
    pingpong(0, 0)
