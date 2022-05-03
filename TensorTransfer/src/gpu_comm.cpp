#include <cuda.h>
#include <cuda_runtime.h>

#include <sys/socket.h>
#include <unistd.h>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <assert.h>

// void print_current_time(const char *prompt) {
//   struct timeval tv;
//   gettimeofday(&tv, NULL);
//   unsigned long time_in_micros = 1000000 * tv.tv_sec + tv.tv_usec;
//   printf("%s%ld\n", prompt, time_in_micros);
// }

bool send_all(int socket, void *buffer, size_t length) {
  char *ptr = (char *)buffer;
  while (length > 0) {
    auto i = send(socket, ptr, length, 0);
    if (i < 1) return false;
    ptr += i;
    length -= i;
  }
  return true;
}

void send_cpu_data(void *data_ptr, int socket_fd, size_t total_bytes) {
  auto success = send_all(socket_fd, data_ptr, total_bytes);
  assert(success);
}

void recv_cpu_data(void *data_ptr, int socket_fd, size_t total_bytes) {
  size_t current_offset = 0;
  while (current_offset < total_bytes) {
    auto recv_bytes =
        recv(socket_fd, (void *)((uint8_t *)data_ptr + current_offset),
             total_bytes - current_offset, 0);
    assert(recv_bytes >= 0);
    current_offset += recv_bytes;
  }
}


void _socket_send(const void *buffer, int socket_fd, size_t buffer_len,
                  std::atomic<size_t> &progress) {
  size_t current_pos = 0;
  while (current_pos < buffer_len) {
    while (progress <= current_pos)
      ;
    void *ptr = (void *)((uint8_t *)buffer + current_pos);
    auto bytes_sent = send(socket_fd, ptr, progress - current_pos, 0);
    assert(bytes_sent >= 0);
    current_pos += bytes_sent;
  }
}

void send_cuda_data(void *data_ptr, int device_index, int socket_fd,
                    size_t length, size_t block_size) {
  cudaError_t status;
  if (device_index != -1) {
    status = cudaSetDevice(device_index);
    assert(!status);
  }

  void *buffer = malloc(length);
  assert(!status);
  std::atomic<size_t> streamed_size(0);
  // start a new thread for socket operation
  std::thread socket_send(_socket_send, buffer, socket_fd, length,
                          std::ref(streamed_size));

  while (streamed_size < length) {
    auto copy_size = std::min(block_size, length - streamed_size);
    status = cudaMemcpy((void *)((uint8_t *)buffer + streamed_size),
                        (void *)((uint8_t *)data_ptr + streamed_size),
                        copy_size, cudaMemcpyDeviceToHost);
    assert(!status);
    streamed_size += copy_size;
  }
  socket_send.join();
  free(buffer);
}


void _socket_recv(void *buffer, int socket_fd, size_t buffer_len,
                  std::atomic<size_t> &progress) {
  size_t current_offset = 0;
  while (current_offset < buffer_len) {
    auto recv_bytes =
        recv(socket_fd, (void *)((uint8_t *)buffer + current_offset),
             buffer_len - current_offset, 0);
    current_offset += recv_bytes;
    progress = current_offset;
  }
}

void recv_cuda_data(void *data_ptr, int device_index, int socket_fd,
                    size_t total_bytes, size_t block_size) {
  cudaError_t status;
  if (device_index != -1) {
    status = cudaSetDevice(device_index);
    assert(!status);
  }
  void *buffer = malloc(total_bytes);

  std::atomic<size_t> progress(0);

  // start a new thread for socket operation
  std::thread socket_recv(_socket_recv, buffer, socket_fd, total_bytes,
                          std::ref(progress));

  size_t copied_bytes = 0;
  size_t next_pos = 0;
  while (copied_bytes < total_bytes) {
    next_pos = progress;
    if (next_pos < total_bytes && next_pos - copied_bytes < block_size) {
      continue;
    }
    if (next_pos - copied_bytes >= block_size) {
      next_pos -= next_pos % block_size;  // alignment
    }
    assert(next_pos >= copied_bytes);
    status =
        cudaMemcpy((void *)((uint8_t *)data_ptr + copied_bytes),
                   (void *)((uint8_t *)buffer + copied_bytes),
                   (size_t)(next_pos - copied_bytes), cudaMemcpyHostToDevice);
    assert(!status);
    copied_bytes = next_pos;
  }
  socket_recv.join();
  free(buffer);
}

cudaIpcMemHandle_t get_cuda_ipc_handle(void *data_ptr, int device_index) {
  cudaIpcMemHandle_t handle;
  cudaError_t status;
  if (device_index >= 0)
  {
    status = cudaSetDevice(device_index);
    assert(!status);
  }
  status = cudaIpcGetMemHandle(&handle, data_ptr);
  if (status) std::cout << "CUDA status: " << status << std::endl;
  assert(!status);
  return handle;
}

void write_data_from_ipc_handle(void *dst, const char *handle_str,
                                int device_index, size_t size,
                                int peer_index, bool is_cuda) {
  auto handle = *(cudaIpcMemHandle_t *)handle_str;
  void *src;
  int original_device;
  cudaError_t status = cudaGetDevice(&original_device);
  assert(!status);
  status = cudaSetDevice(peer_index);
  assert(!status);
  status = cudaIpcOpenMemHandle(&src, handle, cudaIpcMemLazyEnablePeerAccess);
  if (status) std::cout << "CUDA status: " << status << std::endl;
  if (device_index >= 0) {
     status = cudaSetDevice(device_index);
     assert(!status);
  }
  else {
     status = cudaSetDevice(original_device);
     assert(!status);
  }

  assert(!status);
  if (is_cuda) {
    status = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
  } else {
    status = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
  }
  assert(!status);
  // ensure that we have the correct context
  status = cudaSetDevice(peer_index);
  assert(!status);
  status = cudaIpcCloseMemHandle(src);
  if (status)
    std::cout << "CUDA release handle status: " << status << std::endl;
  status = cudaSetDevice(original_device);
  assert(!status);
}

void write_cpu_data_from_ipc_handle(void *dst, const char *handle_str,
                                int device_index, size_t size,
                                int peer_index) {
  auto handle = *(cudaIpcMemHandle_t *)handle_str;
  void *src;
  cudaError_t status = cudaSetDevice(peer_index);
  assert(!status);
  status = cudaIpcOpenMemHandle(&src, handle, cudaIpcMemLazyEnablePeerAccess);
  if (status) std::cout << "CUDA status: " << status << std::endl;
  status = cudaSetDevice(device_index);
  assert(!status);
  status = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
  assert(!status);
  // ensure that we have the correct context
  status = cudaSetDevice(peer_index);
  assert(!status);
  status = cudaIpcCloseMemHandle(src);
  if (status)
    std::cout << "CUDA release handle status: " << status << std::endl;
}

size_t cuda_stream_create() {
  cudaStream_t stream;
  cudaError_t status;
  status = cudaStreamCreate(&stream);
  assert(!status);
  return (size_t)stream;
}

void cuda_stream_destroy(size_t stream = 0) {
  cudaError_t status;
  status = cudaStreamDestroy((cudaStream_t)stream);
  assert(!status);
}

void cuda_stream_sync(size_t stream = 0) {
  cudaError_t status;
  status = cudaStreamSynchronize((cudaStream_t)stream);
  assert(!status);
}
