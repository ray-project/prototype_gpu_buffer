#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <sys/socket.h>
#include <unistd.h>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

void send_cpu_data(void *data_ptr, int socket_fd, size_t total_bytes);
void recv_cuda_data(void *data_ptr, int device_index, int socket_fd,
                    size_t total_bytes, size_t block_size);
void recv_cpu_data(void *data_ptr, int socket_fd, size_t total_bytes);
void send_cuda_data(void *data_ptr, int device_index, int socket_fd,
                    size_t length, size_t block_size);
cudaIpcMemHandle_t get_cuda_ipc_handle(void *data_ptr, int device_index);
void write_data_from_ipc_handle(void *dst, const char *handle_str,
                                int device_index, size_t size,
                                int peer_index, bool is_cuda);
size_t cuda_stream_create();
void cuda_stream_destroy(size_t stream);
void cuda_stream_sync(size_t stream);

void send_cpu_tensor(const torch::Tensor &tensor, int socket_fd) {
  void *data_ptr = tensor.data_ptr();
  size_t nbytes = tensor.nbytes();
  send_cpu_data(data_ptr, socket_fd, nbytes);
}

void recv_cuda_tensor(torch::Tensor &tensor, int socket_fd,
                      size_t block_size) {
  if (!tensor.device().is_cuda()) {
    std::cerr << "ERROR: not a cuda tensor" << std::endl;
    assert(-1);
  }
  int device_index = tensor.device().index();
  void *data_ptr = tensor.data_ptr();
  size_t nbytes = tensor.nbytes();
  recv_cuda_data(data_ptr, device_index, socket_fd, nbytes, block_size);
}

void recv_cpu_tensor(torch::Tensor &tensor, int socket_fd) {
  if (tensor.device().is_cuda()) {
    std::cerr << "ERROR: not a CPU tensor" << std::endl;
    assert(-1);
  }
  void *data_ptr = tensor.data_ptr();
  size_t nbytes = tensor.nbytes();
  recv_cpu_data(data_ptr, socket_fd, nbytes);
}

void send_cuda_tensor(const torch::Tensor &tensor, int socket_fd,
                      size_t block_size) {
  int device_index = tensor.device().index();
  void *data_ptr = tensor.data_ptr();
  size_t nbytes = tensor.nbytes();
  send_cuda_data(data_ptr, device_index, socket_fd, nbytes, block_size);
}

py::bytes _get_cuda_ipc_handle(const torch::Tensor &tensor) {
  int device_index = tensor.device().index();
  void *data_ptr = tensor.data_ptr();
  auto handle = get_cuda_ipc_handle(data_ptr, device_index);
  // To Python objects.
  std::string s((char *)&handle, sizeof(cudaIpcMemHandle_t));
  return py::bytes(s);
}

void write_tensor_from_ipc_handle(torch::Tensor tensor, const char *handle_str,
                                  int peer_index) {
  int device_index = tensor.device().index();
  void *data_ptr = tensor.data_ptr();
  size_t nbytes = tensor.nbytes();
  write_data_from_ipc_handle(data_ptr, handle_str, device_index, nbytes,
                             peer_index, tensor.is_cuda());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("get_cuda_ipc_handle", &_get_cuda_ipc_handle,
        "[doc] get_cuda_ipc_handle");
  m.def("write_tensor_from_ipc_handle", &write_tensor_from_ipc_handle,
        "[doc] write_tensor_from_ipc_handle");

  m.def("send_cpu_tensor", &send_cpu_tensor, "[doc] send_cpu_tensor");
  m.def("recv_cpu_tensor", &recv_cpu_tensor, "[doc] recv_cpu_tensor");
  m.def("send_cuda_tensor", &send_cuda_tensor, "[doc] send_cuda_tensor");
  m.def("recv_cuda_tensor", &recv_cuda_tensor, "[doc] recv_cuda_tensor");

  m.def("cuda_stream_create", &cuda_stream_create, "[doc] cuda_stream_create");
  m.def("cuda_stream_sync", &cuda_stream_sync, "[doc] cuda_stream_sync");
  m.def("cuda_stream_destroy", &cuda_stream_destroy,
        "[doc] cuda_stream_destroy");
}