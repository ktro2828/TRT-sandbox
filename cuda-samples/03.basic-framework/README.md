# Chapter 3: Basic framework of simple CUDA programs

## Basic framework of simple CUDA programs

基本的なフレームワークは以下のようになる。

```c++
header inclusion
const or macro definition
declarations of C++ functions and CUDA kernels

int main()
{
    allocate host and device memory
    initialize data in host memory
    transfer data from host to device
    launch (call) kernel to do calculations in the device
    transfer data from device to host
    free host and device memory
}

definitions of C++ functions and CUDA kernels
```

## Memory allocation in device

`cudaMalloc()`関数によってデバイス上にメモリを割り当てる。CUDAランタイムAPI関数のマニュアル: https://docs.nvidia.com/cuda/cuda-runtime-api

`cudaMalloc()`のプロトタイプは、

```c++
cudaError_t cudaMalloc(void ** address, size_t size);
```

`address`はアドレスのポインタ(=ダブルポインタ)で、`size`は割り当てるバイト数。割り当てに成功すると`cudaSuccess`が返される。

`cudaMalloc()`によって割り当てられたメモリは`cudaFree()`によって解放する必要がある。

```c++
cudaError_t cudaFree(void * address);
```

### Data transfer between host and device

ホストとデバイス間のデータ転送には`cudaMemcpy()`を使う。

```c++
cudaError_t cudaMemcpy(
    void                *dst,
    const void          *src,
    size_t              count,
    enum cudaMemcpyKind kind
);
```
