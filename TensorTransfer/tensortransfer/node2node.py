import socket
from .utils import get_node_ip_address
from . import tensor_meta
from . import tensor_comm

kSocketBufferSize = 1024 * 1024 * 256


class TransferService:
    def __init__(self):
        self.address = get_node_ip_address()
        self.recv_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.recv_socket.setsockopt(
            socket.SOL_SOCKET, socket.SO_SNDBUF, kSocketBufferSize)
        self.port = None
        for port in range(18080, 30000):
            try:
                self.recv_socket.bind(('', port))
                self.port = port
                break
            except Exception:
                pass
        if self.port is None:
            raise IOError("Cannot open any port")
        self.recv_socket.listen(1)

    def get_handle(self, tensor):
        meta = tensor_meta.TensorMeta.from_tensor(tensor)
        meta.update_route(self.address, self.port)
        return meta.serialize()

    def get_tensor_from_handle(self, handle, to_device, device_index=None, backend=None, route="auto"):
        if to_device not in ('cpu', 'cuda'):
            raise ValueError("Unsupported device type: " + str(to_device))
        meta = tensor_meta.TensorMeta.deserialize(handle)
        if backend is not None:
            meta.backend = backend
        if route == "auto":
            if meta.ip_address == self.address:
                if to_device == 'cpu':
                    # TODO: Implement faster local tensor receiving method using shared memory.
                    return self._receive_tensor_remote(meta, to_device, device_index)
                else:
                    return self._receive_tensor_local(meta, to_device, device_index)
            else:
                return self._receive_tensor_remote(meta, to_device, device_index)
        elif route == "local":
            return self._receive_tensor_local(meta, to_device, device_index)
        elif route == "remote":
            return self._receive_tensor_remote(meta, to_device, device_index)
        else:
            raise ValueError("Unknown route", route)

    def send_tensor(self, handle, tensor):
        meta = tensor_meta.TensorMeta.deserialize(handle)
        self.recv_socket.setblocking(True)
        channel, _details = self.recv_socket.accept()
        return_code = channel.recv(4)
        self.recv_socket.setblocking(False)
        if return_code == b'DONE':
            tensor_comm.maybe_free_buffer(meta)
            return
        elif return_code == b'PUSH':
            tensor_comm.send_tensor_socket(channel, tensor)
        else:
            raise ValueError("Unknown return_code", return_code)
        channel.close()

    @classmethod
    def _create_tcp_connection(cls, meta):
        sender = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sender.setsockopt(socket.SOL_SOCKET,
                          socket.SO_SNDBUF, kSocketBufferSize)
        sender.connect((meta.ip_address, meta.port))
        return sender

    def _receive_tensor_local(self, meta, device, device_index=None):
        sender = self._create_tcp_connection(meta)
        tensor = meta.allocate_tensor(device, device_index)
        tensor_comm.recv_tensor_local(meta, tensor)
        sender.sendall(b'DONE')
        sender.close()
        return tensor

    def _receive_tensor_remote(self, meta, device, device_index=None):
        sender = self._create_tcp_connection(meta)
        tensor = meta.allocate_tensor(device, device_index)
        sender.sendall(b'PUSH')
        tensor_comm.recv_tensor_socket(sender, tensor)
        sender.close()
        return tensor
