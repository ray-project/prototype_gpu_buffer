# TensorTransfer

TensorTransfer is a library for point-to-point tensor transfer between two GPUs. TensorTransfer is designed for dynamic environments, which does not require an initialization phase for all of the nodes. TensorTransfer will detect whether two GPUs are on the same machine or not and transfer the tensor through the fastest path.

## Installation

```bash
git clone https://github.com/suquark/TensorTransfer.git
cd TensorTransfer
python setup.py install
```

## API

```python
from tensortransfer import TransferService
# Initialize the service
service = TransferService()
```

### Sender

```python
# Get the metadata of the tensor, e.g. shape, source IP & Port, source GPU device & CUDA IPC handle 
handle = service.get_handle(tensor)
# Start a service to transfer on the sender side
service.send_tensor(handle, tensor)
```

### Receiver

```python
# Receive a tensor according to the handle
# device={CPU, GPU}
# backend={tensorflow, pytorch}
tensor = service.get_tensor_from_handle(handle, device, device_index, backend)
```

## Examples

See [examples/send_and_recv.py](examples/send_and_recv.py) as an example use of the library.

## Performance

For cross machine transmission on p3.2xlarge, it can achieve throughput of > 90% bandwidth
