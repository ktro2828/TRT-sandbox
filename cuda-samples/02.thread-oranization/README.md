# Chapter 2: Thread Organization in CUDA

## CUDA program containing a CUDA kernel

### A CUDA program containing host functions only

A `CUA kernel` is a function that is called by the host and executes in the device. There are many rules for defining a kernel.

```c++
__global__ void hello_from_gpu()
{
    printf("Hello World from the GPU!\n");
}
```

The order of the qualifiers, `__global__` and `void` are not important. That can also be written as follows:

```cu
void __global__ hello_from_gpu()
{
    printf("Hello World from the GPU!\n);
}
```

We then write a main function and call the kernel from the host:

```c++
#include <stdio.h>

__global__ void hello_from_gpu()
{
    printf("Hello World from the GPU!\n");
}

int main(void)
{
    hello_from_gpu<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

There must be an **execution configuration** like `<<<1, 1>>` between the kernel name `()`. The execution configuration specifies the number of threads and their organization for the kernel.

> 実行呼び出しは`<<<1, 1>>>`のようになる。

The threads for the kernel form a **grid**, which can contain multiple **blocks**. Each block in turn can contain multiple threads. The number of blocks within the grid is called the **grid size**. All the blocks in the grid have the same number of threads and this number is called **block size**. Then, for the simple execution configuration `<<<grid_size, block_size>>>` with two integer numbers.

> カーネルスレッドは複数の**ブロック**を内包した**グリッド**を形成する。各ブロックは複数のスレッドを内包できる。グリッドの持つブロック数は**グリッドサイズ**と呼ばれ、各ブロックは同じスレッド数を持ちそれを**ブロックサイズ**と呼ぶ。実行呼び出し時には`<<<grid_size, block_size>>>`で呼び出すことができる。


## Thread organization in CUDA

### A CUDA kernel using multiple threads

GPUには複数のコアがあり必要とあればカーネルに複数のスレッドを割り当てて実行できる。次のコード`hello3.cu`は1グリッドにつき2ブロックを割り当て、各ブロックに4スレッドを割り当ててカーネル実行する。

```c++
#include <stdio.h>

__global__ void hello_from_gpu()
{
    printf("Hello World from the GPU!\n");
}

int main(void)
{
    hello_from_gpu<<<2, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

実行結果は以下のようになる。

```shell
Hello World from the GPU!
Hello World from the GPU!
Hello World from the GPU!
Hello World from the GPU!
Hello World from the GPU!
Hello World from the GPU!
Hello World from the GPU!
Hello World from the GPU!
```

### Using thread indices in a CUDA kernel

全スレッドはユニークな名前もしくはインデックスを持ち、これらは組み込み変数として定義される。グリッド数には`gridDim.x`、ブロック数には`blockDim.x`のようにアクセスする。
インデックスにアクセスする場合には、`blockIdx.x`・`threadIdx.x`のようにアクセスする。

- `blockIdx.x`: グリッド内のブロックインデックスを示し、`gridDim.x - 1`までの値を取る。
- `threadIdx.`: ブロック内のスレッドのインデックスを示し、`blockDim.x - 1`までの値を取る。

`hello3.cu`内でこれらにアクセスするように修正したものが`hello4.cu`。

```c++
#include <stdio.h>

__global__ void hello_from_gpu()
{
    const int b_idx = blockIdx.x;
    const int t_idx = threadIdx.x;
    printf("Hello World from block %d and thread %d!\n", b_idx, t_idx);
}

int main(void)
{
    hello_from_gpu<<<2, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

実行結果は以下のようになる。blockインデックス番号の順番はその時々で変わり、つまり各ブロックは独立して実行されている。

```shell
Hello World from block 0 and thread 0!
Hello World from block 0 and thread 1!
Hello World from block 0 and thread 2!
Hello World from block 0 and thread 3!
Hello World from block 1 and thread 0!
Hello World from block 1 and thread 1!
Hello World from block 1 and thread 2!
Hello World from block 1 and thread 3!
```

### Generalization to multi-dimensional grids and blocks

`blockIdx`と`threadIdx`は`uint3`として`vector_types.h`に以下のように定義されている。

```c++
struct __device_builtin__ uint3
{
    unsigned int x, y, z;
};
typedef __device_builtin__ struct uint3 uint3;
```

`gridDim`と`blockDim`は`dim3`で定義されている。これらの組み込み変数はCUDAカーネル内からのみ参照可能である。

`dim3`を使うことで多次元のグリッドとブロックを定義できる。

```c++
dim3 grid_size(Gx, Gy, Gz);
dim3 block_size(Bx, By, Bz);
```

`z`の次元数が`1`の場合、以下のようにも書ける。

```c++
dim3 grid_size(Gx, Gy);
dim3 block_size(Bx, By);
```

`hello5.cu`は多次元のブロックを用いた実装の例である。

```c++
#include <stdio.h>

__global__ void hello_from_gpu()
{
    const int b = blockIdx.x;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    printf("Hello World from block-%d and thread-(%d, %d)!\n", b, tx, ty);
}

int main(void)
{
    const dim3 block_size(2, 4);
    hello_from_gpu<<<1, block_size>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

実行結果は以下のようになる。

```shell
Hello World from block-1 and thread-(0, 0)!
Hello World from block-1 and thread-(1, 0)!
Hello World from block-1 and thread-(0, 1)!
Hello World from block-1 and thread-(1, 1)!
Hello World from block-1 and thread-(0, 2)!
Hello World from block-1 and thread-(1, 2)!
Hello World from block-1 and thread-(0, 3)!
Hello World from block-1 and thread-(1, 3)!
Hello World from block-0 and thread-(0, 0)!
Hello World from block-0 and thread-(1, 0)!
Hello World from block-0 and thread-(0, 1)!
Hello World from block-0 and thread-(1, 1)!
Hello World from block-0 and thread-(0, 2)!
Hello World from block-0 and thread-(1, 2)!
Hello World from block-0 and thread-(0, 3)!
Hello World from block-0 and thread-(1, 3)!
Hello World from block-2 and thread-(0, 0)!
Hello World from block-2 and thread-(1, 0)!
Hello World from block-2 and thread-(0, 1)!
Hello World from block-2 and thread-(1, 1)!
Hello World from block-2 and thread-(0, 2)!
Hello World from block-2 and thread-(1, 2)!
Hello World from block-2 and thread-(0, 3)!
Hello World from block-2 and thread-(1, 3)!
```

スレッドインデックスは`threadIdx.y * blockDim.x + threadIdx.x`のように計算できる。

一般化すると、多次元でのスレッドインデックスは以下のように定式化される。

```c++
int t_idx = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
```

### Limits on the grid and block sizes

[Kepler(ケプラ)アーキテクチャ](https://en.wikipedia.org/wiki/Kepler_(microarchitecture))では、グリッド数・ブロック数の最大値は以下のようになる。

FYI: https://pc.watch.impress.co.jp/docs/column/kaigai/520640.html

```shell
# grid size limits
gridDim.x <= 2^{31} - 1
gridDim.y <= 2^{16} - 1 = 65535
gridDim.z <= 2^{16} - 1 = 65535

# block size limits
blockDim.x <= 1024
blockDim.y <= 1024
blockDim.z <= 64

blockDim.x * blockDim.y * blockDim.z <= 1024
```

## Using `nvcc` to compile CUDA programs

`nvcc`で`.cu`ファイルをコンパイルする際には`cuda.h`や`cuda_runtime.h`のようなヘッダファイルは自動的にインクルードされる。

デバイスコードをPTXコードにコンパイルすす際には`-arch=compute_XY`のようにCUDAフィーチャー(the compute capability of a virtual architecture)を指定できる。また、`-code=sm_ZW`でバイナリが使用できるGPU(the compute capability of a real architecture)を指定できる。
このとき、real architectureはvirtual architecture以上のものを指定する必要がある。

nvccの詳細については[NVIDIA CUDA Compiler Driver NVCC](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/)を参照。

```shell
$ nvcc -arch=compute_60 -code=sm_70 xxx.cu  # OK

$ nvcc -arch=compute_70 -code=sm_60 xxx.cu  # ERROR

$ nvcc -arch=compute_70 -code=sm_70 xxx.cu  # OK；基本的には同じものを指定する。
```