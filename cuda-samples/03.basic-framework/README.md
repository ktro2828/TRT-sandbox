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
    void                *dst,  //...コピー先のアドレス
    const void          *src,  //...ソースデータのアドレス
    size_t              count, //...バイトサイズ
    enum cudaMemcpyKind kind   //...コピー方法
);
```

### Correspondence between data and threads in CUDA kernel

`add1.cu`ではブロック数`128`でグリッド数`10^8/128`のカーネル呼び出しをしている。

カーネル定義時の要件としては、以下がある
- カーネルの戻り値は必ず`void`
- カーネルは必ず`__global__`修飾子を持つ
- カーネルをオーバロードすることは可能
- カーネルのパラメータ数は動的ではならない
- カーネルはメンバ関数として定義してはならない
- 動的並列処理を使用しない限り、カーネル内でカーネルを呼び出せない