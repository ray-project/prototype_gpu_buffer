#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <iostream>
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


using namespace tensorflow;


constexpr size_t kCUDAMemcpyBlockSize = 4096 * 256;


class SendCpuTensor : public OpKernel {
 public:
  explicit SendCpuTensor(OpKernelConstruction* context) : OpKernel(context) {
    // Get the index of the value to preserve
    OP_REQUIRES_OK(context,
                   context->GetAttr("socket_fd", &socket_fd_));
    // Check that preserve_index is positive
    OP_REQUIRES(context, socket_fd_ != -1,
                errors::InvalidArgument("Invalid socket fd ",
                                        socket_fd_));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    void *data_ptr = (void *)input_tensor.tensor_data().data();
    size_t nbytes = input_tensor.TotalBytes();
    send_cpu_data(data_ptr, socket_fd_, nbytes);
  }

 private:
  int socket_fd_;
};


REGISTER_KERNEL_BUILDER(Name("SendCpuTensor").Device(DEVICE_CPU), SendCpuTensor);
REGISTER_OP("SendCpuTensor")
    .Attr("socket_fd: int")
    .Attr("T: {float, double, int32, int64}")
    .Input("input: T");


class RecvCpuTensor : public OpKernel {
 public:
  explicit RecvCpuTensor(OpKernelConstruction* context) : OpKernel(context) {
    // Get the index of the value to preserve
    OP_REQUIRES_OK(context,
                   context->GetAttr("socket_fd", &socket_fd_));
    // Check that preserve_index is positive
    OP_REQUIRES(context, socket_fd_ != -1,
                errors::InvalidArgument("Invalid socket fd ",
                                        socket_fd_));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    void *data_ptr = (void *)input_tensor.tensor_data().data();
    size_t nbytes = input_tensor.TotalBytes();
    recv_cpu_data(data_ptr, socket_fd_, nbytes);
  }

 private:
  int socket_fd_;
};


REGISTER_KERNEL_BUILDER(Name("RecvCpuTensor").Device(DEVICE_CPU), RecvCpuTensor);
REGISTER_OP("RecvCpuTensor")
    .Attr("socket_fd: int")
    .Attr("T: {float, double, int32, int64}")
    .Input("input: T");


class SendCudaTensor : public OpKernel {
 public:
  explicit SendCudaTensor(OpKernelConstruction* context) : OpKernel(context) {
    // Get the index of the value to preserve
    OP_REQUIRES_OK(context,
                   context->GetAttr("socket_fd", &socket_fd_));
    // Check that preserve_index is positive
    OP_REQUIRES(context, socket_fd_ != -1,
                errors::InvalidArgument("Invalid socket fd ",
                                        socket_fd_));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);

    // Create an output tensor
    void *data_ptr = (void *)input_tensor.tensor_data().data();
    size_t nbytes = input_tensor.TotalBytes();
    send_cuda_data(data_ptr, -1, socket_fd_, nbytes, kCUDAMemcpyBlockSize);
  }

 private:
  int socket_fd_;
};


REGISTER_KERNEL_BUILDER(Name("SendCudaTensor").Device(DEVICE_GPU), SendCudaTensor);
REGISTER_OP("SendCudaTensor")
    .Attr("socket_fd: int")
    .Attr("T: {float, double, int32, int64}")
    .Input("input: T");


class RecvCudaTensor : public OpKernel {
 public:
  explicit RecvCudaTensor(OpKernelConstruction* context) : OpKernel(context) {
    // Get the index of the value to preserve
    OP_REQUIRES_OK(context,
                   context->GetAttr("socket_fd", &socket_fd_));
    // Check that preserve_index is positive
    OP_REQUIRES(context, socket_fd_ != -1,
                errors::InvalidArgument("Invalid socket fd ",
                                        socket_fd_));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    void *data_ptr = (void *)input_tensor.tensor_data().data();
    size_t nbytes = input_tensor.TotalBytes();
    recv_cuda_data(data_ptr, -1, socket_fd_, nbytes, kCUDAMemcpyBlockSize);
  }

 private:
  int socket_fd_;
};


REGISTER_KERNEL_BUILDER(Name("RecvCudaTensor").Device(DEVICE_GPU), RecvCudaTensor);
REGISTER_OP("RecvCudaTensor")
    .Attr("socket_fd: int")
    .Attr("T: {float, double, int32, int64}")
    .Input("input: T");

class GetCudaIpcHandle : public OpKernel {
 public:
  explicit GetCudaIpcHandle(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("device_index", &device_index_));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    void *data_ptr = (void *)input_tensor.tensor_data().data();
    void *new_data_ptr = NULL;
    cudaMalloc(&new_data_ptr, input_tensor.TotalBytes());
    cudaMemcpy(new_data_ptr, data_ptr, input_tensor.TotalBytes(), cudaMemcpyDefault);
    auto handle = get_cuda_ipc_handle(new_data_ptr, device_index_);
    // To Python objects.
    std::string s((char *)&handle, sizeof(cudaIpcMemHandle_t));
    // Create an output tensor
    Tensor* ipc_handle = NULL;
    OP_REQUIRES_OK(context, context->allocate_output("ipc_handle", {},
                                                     &ipc_handle));
    auto ipc_handle_flat = ipc_handle->flat<std::string>();
    ipc_handle_flat(0) = s;
    Tensor* pointer_address = NULL;
    OP_REQUIRES_OK(context, context->allocate_output("pointer_address", {},
                                                     &pointer_address));
    auto pointer_address_flat = pointer_address->flat<std::string>();
    std::string pointer_address_str((char *)&new_data_ptr, sizeof (new_data_ptr));
    pointer_address_flat(0) = pointer_address_str;
  }
  private:
    int device_index_;
};

REGISTER_KERNEL_BUILDER(Name("GetCudaIpcHandle").Device(DEVICE_GPU), GetCudaIpcHandle);

REGISTER_OP("GetCudaIpcHandle")
    .Attr("T: {float, double, int32, int64}")
    .Input("tensor: T")
    .Attr("device_index: int")
    .Output("ipc_handle: string")
    .Output("pointer_address: string");


class FreeCudaMemory : public OpKernel {
 public:
  explicit FreeCudaMemory(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input_tensor = context->input(0);
    std::string pointer_address_;
    pointer_address_ = input_tensor.flat<std::string>()(0);
    void *pointer = (void *)*(size_t*)pointer_address_.c_str();
    auto status = cudaFree(pointer);
    if (status != 0) {
        std::cerr << "cuda free failed, status = " << status << std::endl;
        assert(!status);
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("FreeCudaMemory").Device(DEVICE_GPU), FreeCudaMemory);

REGISTER_OP("FreeCudaMemory")
    .Input("pointer_address: string");

class WriteTensorFromIpcHandle : public OpKernel {
 public:
  explicit WriteTensorFromIpcHandle(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("ipc_handle", &ipc_handle_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("peer_index", &peer_index_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("is_cuda", &is_cuda_));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    void *data_ptr = (void *)input_tensor.tensor_data().data();
    size_t nbytes = input_tensor.TotalBytes();
    write_data_from_ipc_handle(data_ptr, ipc_handle_.c_str(), -1, nbytes,
                             peer_index_, is_cuda_);
  }
  private:
    string ipc_handle_;
    int peer_index_;
    bool is_cuda_;
};

REGISTER_KERNEL_BUILDER(Name("WriteTensorFromIpcHandle").Device(DEVICE_GPU), WriteTensorFromIpcHandle);

REGISTER_OP("WriteTensorFromIpcHandle")
    .Attr("T: {float, double, int32, int64}")
    .Input("tensor: T")
    .Attr("ipc_handle: string")
    .Attr("peer_index: int")
    .Attr("is_cuda: bool");
