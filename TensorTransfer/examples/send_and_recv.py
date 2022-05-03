import ray
import torch
import tensorflow as tf
from tensortransfer import TransferService


@ray.remote(num_gpus=0.1)
class Sender:
    def __init__(self, is_torch):
        self.is_torch = is_torch
        if self.is_torch:
            self.t = torch.randn(10).cuda()
        else:
            with tf.device('/device:gpu:0'):
                self.t = tf.random.normal(10)
        self.service = TransferService()
        print("Sender:", self.t)

    def get_handle(self):
        handle = self.service.get_handle(self.t)
        return handle

    def send(self, handle):
        self.service.send_tensor(handle, self.t)


@ray.remote(num_gpus=0.1)
class Receiver:
    def __init__(self, is_torch):
        if is_torch:
            self.backend = "torch"
        else:
            self.backend = "tensorflow"
        self.device_index = 0
        self.service = TransferService()

    def recv(self, handle):
        self.t = self.service.get_tensor_from_handle(
            handle, 'cuda', self.device_index, backend=self.backend)
        print("Receiver:", self.t)


if __name__ == "__main__":
    ray.init()
    sender = Sender.remote(True)
    receiver = Receiver.remote(False)
    handle = sender.get_handle.remote()
    ray.get([sender.send.remote(handle), receiver.recv.remote(handle)])

