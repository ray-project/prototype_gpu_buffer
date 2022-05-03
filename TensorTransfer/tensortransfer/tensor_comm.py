import torch
import tensortransfer_torch_ext as torch_tensor_comm

from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader
from tensorflow.python.framework.ops import EagerTensor as TFEagerTensor

tensorflow_comm = load_library.load_op_library(
    resource_loader.get_path_to_datafile('core/tensorflow_bind.so'))

kCUDAMemcpyBlockSize = 4096 * 256


def tensorflow_get_device_and_index(tensor):
    assert isinstance(
        tensor, TFEagerTensor), "unexpected type: " + str(type(tensor))
    device_info = tensor.device.split('/')[-1].split(':')
    assert len(device_info) == 3 and device_info[0] == 'device'
    if device_info[1] == 'CPU':
        device = 'cpu'
    elif device_info[1] == 'GPU':
        device = 'cuda'
    else:
        raise NotImplementedError
    return device, int(device_info[2])


def get_cuda_ipc_handle(tensor):
    if isinstance(tensor, torch.Tensor):
        assert tensor.is_cuda
        handle = torch_tensor_comm.get_cuda_ipc_handle(tensor)
        address = None
        return handle, address
    elif isinstance(tensor, TFEagerTensor):
        device, device_index = tensorflow_get_device_and_index(tensor)
        handle_tensor, address_tensor = tensorflow_comm.get_cuda_ipc_handle(tensor, device_index)
        handle = handle_tensor.numpy()  # TensorFlow magic name
        address = address_tensor.numpy()
        assert isinstance(handle, bytes)
        assert isinstance(address, bytes)
        return handle, address
    else:
        raise NotImplementedError


def send_tensor_socket(conn, tensor):
    fd = conn.fileno()
    if isinstance(tensor, torch.Tensor):
        if tensor.is_cuda:
            torch_tensor_comm.send_cuda_tensor(
                tensor, fd, kCUDAMemcpyBlockSize)
        else:
            torch_tensor_comm.send_cpu_tensor(tensor, fd)
    elif isinstance(tensor, TFEagerTensor):
        if tensorflow_get_device_and_index(tensor)[0] == 'cuda':
            tensorflow_comm.send_cuda_tensor(tensor, fd)
        else:
            tensorflow_comm.send_cpu_tensor(tensor, fd)
    else:
        raise NotImplementedError


def recv_tensor_socket(conn, tensor):
    fd = conn.fileno()
    if isinstance(tensor, torch.Tensor):
        if tensor.is_cuda:
            torch_tensor_comm.recv_cuda_tensor(
                tensor, fd, kCUDAMemcpyBlockSize)
        else:
            torch_tensor_comm.recv_cpu_tensor(tensor, fd)
    elif isinstance(tensor, TFEagerTensor):
        if tensorflow_get_device_and_index(tensor)[0] == 'cuda':
            tensorflow_comm.recv_cuda_tensor(tensor, fd)
        else:
            tensorflow_comm.recv_cpu_tensor(tensor, fd)
    else:
        raise NotImplementedError


def recv_tensor_local(meta, tensor):
    if isinstance(tensor, torch.Tensor):
        if meta.device == 'cuda':
            torch_tensor_comm.write_tensor_from_ipc_handle(
                tensor, meta.cuda_ipc_handle, meta.device_index)
        else:
            raise NotImplementedError
    elif isinstance(tensor, TFEagerTensor):
        if meta.device == 'cuda':
            is_cuda = tensorflow_get_device_and_index(tensor)[0] == 'cuda'
            tensorflow_comm.write_tensor_from_ipc_handle(
                tensor, meta.cuda_ipc_handle, meta.device_index, is_cuda)
        else:
            raise NotImplementedError

    else:
        raise NotImplementedError


def maybe_free_buffer(meta):
    if meta.cuda_tensor_address is not None:
        tensorflow_comm.free_cuda_memory(meta.cuda_tensor_address)
