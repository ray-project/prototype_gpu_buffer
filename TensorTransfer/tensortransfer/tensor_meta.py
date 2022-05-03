import pickle
import tensorflow
import torch

from .import tensor_comm


class TensorMeta:
    def __init__(self):
        self.backend = None
        self.ip_address = None
        self.port = None
        self.shape = None
        self.strides = None
        self.dtype = None
        self.numel = None
        self.device = None
        self.device_index = None
        self.itemsize = None
        self.cuda_ipc_handle = None
        self.cuda_tensor_address = None

    @classmethod
    def from_tensor(cls, tensor):
        tensor_meta = cls()
        tensor_meta.shape = tuple(tensor.shape)
        if isinstance(tensor, torch.Tensor):
            tensor_meta.backend = 'torch'
            tensor_meta.dtype = tensor.dtype.__reduce__()
            tensor_meta.strides = tuple(tensor.stride())
            tensor_meta.numel = tensor.numel()
            tensor_meta.device = tensor.device.type
            tensor_meta.device_index = tensor.device.index
            tensor_meta.itemsize = tensor.element_size()
        elif isinstance(tensor, tensor_comm.TFEagerTensor):
            tensor_meta.backend = 'tensorflow'
            tensor_meta.dtype = tensor.dtype.__reduce__()[1][0]
            tensor_meta.strides = None
            tensor_meta.numel = tensor._num_elements()
            device, device_index = tensor_comm.tensorflow_get_device_and_index(
                tensor)
            tensor_meta.device = device
            tensor_meta.device_index = device_index
            tensor_meta.itemsize = tensor.dtype.size
        if tensor_meta.device == 'cuda':
            tensor_meta.cuda_ipc_handle, tensor_meta.cuda_tensor_address = (
                tensor_comm.get_cuda_ipc_handle(tensor))
        return tensor_meta

    def allocate_tensor(self, device=None, device_index=None):
        if device is None:
            device = self.device
        if device_index is None:
            device_index = self.device_index
        if self.backend == 'tensorflow':
            if device == 'cpu':
                device_str = '/device:CPU:0'
            elif device == 'cuda':
                device_str = '/device:GPU:' + str(self.device_index)
            with tensorflow.device(device_str):
                return tensorflow.zeros(self.shape, dtype=self.dtype)
        elif self.backend == 'torch':
            if device == 'cuda':
                device += ':' + str(device_index)
            tensor = torch.empty(self.numel, device=device,
                                 dtype=getattr(torch, self.dtype))
            if self.strides is None:
                return tensor.view(self.shape)
            else:
                return tensor.as_strided(self.shape, self.strides)
        else:
            raise NotImplementedError

    def update_route(self, ip_address, port):
        self.ip_address = ip_address
        self.port = port

    @property
    def total_bytes(self):
        return self.numel * self.itemsize

    def serialize(self):
        return pickle.dumps({
            'backend': self.backend,
            'ip_address': self.ip_address,
            'port': self.port,
            'shape': self.shape,
            'strides': self.strides,
            'dtype': self.dtype,
            'itemsize': self.itemsize,
            'numel': self.numel,
            'device': self.device,
            'device_index': self.device_index,
            'cuda_ipc_handle': self.cuda_ipc_handle,
            'cuda_tensor_address': self.cuda_tensor_address,
        })

    @classmethod
    def deserialize(cls, tensor_meta_bytes):
        tensor_meta_dict = pickle.loads(tensor_meta_bytes)
        self = cls()
        self.backend = tensor_meta_dict['backend']
        self.ip_address = tensor_meta_dict['ip_address']
        self.port = tensor_meta_dict['port']
        self.shape = tensor_meta_dict['shape']
        self.strides = tensor_meta_dict['strides']
        self.dtype = tensor_meta_dict['dtype']
        self.itemsize = tensor_meta_dict['itemsize']
        self.numel = tensor_meta_dict['numel']
        self.device = tensor_meta_dict['device']
        self.device_index = tensor_meta_dict['device_index']
        self.cuda_ipc_handle = tensor_meta_dict['cuda_ipc_handle']
        self.cuda_tensor_address = tensor_meta_dict['cuda_tensor_address']
        return self
