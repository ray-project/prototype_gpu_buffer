import pytest

import time
import ray
import numpy as np
import os


@ray.remote
class RemoteActor:
    def __init__(self, name, device_index, tensor_size, is_torch=True, route='auto'):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
        self.is_torch = is_torch
        self.route = route
        import torch
        import tensorflow as tf
        # tf.config.gpu.set_per_process_memory_fraction(0.4)
        # tf.config.gpu.set_per_process_memory_growth(True)
        gpus = tf.config.experimental.list_physical_devices('GPU')
        print("gpus", gpus)
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [
            tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
        self.device_index = device_index
        import tensortransfer
        self.p2p_service = tensortransfer.TransferService()
        # print(name, os.environ['CUDA_VISIBLE_DEVICES'], flush=True)
        torch.cuda.init()
        if self.is_torch:
            if device_index < 0:
                self.t = torch.zeros(tensor_size)
            else:
                self.t = torch.zeros(tensor_size).cuda(device_index)
        else:
            if device_index < 0:
                with tf.device('/device:cpu:0'):
                    self.t = tf.zeros(tensor_size)
            else:
                with tf.device('/device:gpu:' + str(device_index)):
                    self.t = tf.zeros(tensor_size)
        self.name = name
        self.perf = {"get_handle": [], "send_tensor": [],
                     "recv_tensor": [], "add": []}
        print(self.name, "init finished", flush=True)

    def get_addr_info(self):
        return self.p2p_service.address, self.p2p_service.port

    def get_handle(self, dummy1, dummy2):
        start = time.time()
        handle = self.p2p_service.get_handle(self.t)
        self.perf['get_handle'].append(time.time() - start)
        return handle

    def send(self, handle):
        print(time.time(), "===", self.name, "send", self.t, flush=True)
        start = time.time()
        self.p2p_service.send_tensor(handle, self.t)
        self.perf['send_tensor'].append(time.time() - start)

    def recv(self, handle):
        start = time.time()
        if self.is_torch:
            backend = 'torch'
        else:
            backend = 'tensorflow'
        if self.device_index < 0:
            self.t = self.p2p_service.get_tensor_from_handle(
                handle, 'cpu', backend=backend, route=self.route)
        else:
            self.t = self.p2p_service.get_tensor_from_handle(
                handle, 'cuda', self.device_index, backend=backend, route=self.route)
        print(time.time(), "===", self.name, "recv", backend, self.t, flush=True)
        self.perf['recv_tensor'].append(time.time() - start)
        start = time.time()
        self.t = self.t + 1
        if self.is_torch:
            import torch
            torch.cuda.synchronize(device=self.device_index)
        else:
            print(time.time(), "===", self.name, "self.t.numpy()", self.t.numpy())

        self.perf['add'].append(time.time() - start)

    def reset(self):
        print(time.time(), "===", self.name, "reset", flush=True)
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


def pingpong_local(tensor_size, device_index, peer_index, alice_use_torch, bob_use_torch, total_rounds, heatup_rounds=2):
    a0 = RemoteActor.remote("alice", device_index, tensor_size, is_torch=alice_use_torch)
    a1 = RemoteActor.remote("bob", peer_index, tensor_size, is_torch=bob_use_torch)

    dummy_send, dummy_recv = None, None

    for i in range(heatup_rounds):
        if i % 2 == 0:
            handle = a0.get_handle.remote(dummy_send, dummy_recv)
            dummy_send = a0.send.remote(handle)
            dummy_recv = a1.recv.remote(handle)
        else:
            handle = a1.get_handle.remote(dummy_send, dummy_recv)
            dummy_send = a1.send.remote(handle)
            dummy_recv = a0.recv.remote(handle)

    ray.get([dummy_send, dummy_recv])
    print("=" * 30, time.time(), flush=True)
    ray.get(a1.get.remote())
    ray.get(a0.reset.remote())
    ray.get(a1.reset.remote())
    print("start", time.time(),flush=True)
    start = time.time()
    for i in range(total_rounds):
        if i % 2 == 0:
            handle = a0.get_handle.remote(dummy_send, dummy_recv)
            dummy_send = a0.send.remote(handle)
            dummy_recv = a1.recv.remote(handle)
        else:
            handle = a1.get_handle.remote(dummy_send, dummy_recv)
            dummy_send = a1.send.remote(handle)
            dummy_recv = a0.recv.remote(handle)

    ray.get([dummy_send, dummy_recv])
    s = ray.get(a0.get.remote())
    print("average time", (time.time() - start) / total_rounds, flush=True)
    print("result", s, flush=True)
    assert int(s[0]) == int(s[1]) == int(total_rounds)

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


@pytest.mark.parametrize("tensor_size", [8, 1024, 2 ** 20, 2 ** 28])
@pytest.mark.parametrize("alice_use_torch", [True, False])
@pytest.mark.parametrize("bob_use_torch", [True, False])
@pytest.mark.parametrize("route", ["local", "remote"])
def test_local_pingpong(tensor_size, alice_use_torch, bob_use_torch, route):
    ray.init(redis_password="xizcu9fuafdshfdsjkhfdshkjfdsus")
    print("running remote ping-pong test...")
    pingpong_local(tensor_size, 0, 0, alice_use_torch, bob_use_torch, total_rounds=10, heatup_rounds=2)
    #tensor_size, device_index, peer_index, alice_use_torch, bob_use_torch, total_rounds, heatup_rounds=2
    ray.shutdown()

