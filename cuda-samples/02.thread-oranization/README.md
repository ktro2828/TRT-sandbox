# Chapter 2: Thread Organization in CUDA

## CUDA program containing a CUDA kernel

### A CUDA program containing host functions only

A `CUA kernel` is a function that is called by the host and executes in the device. There are many rules for defining a kernel.

```cu
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

```cu
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

The threads for the kernel form a **grid**, which can contain multiple **blocks**. Each block in turn can contain multiple threads. The number of blocks within the grid is called the **grid size**. All the blocks in the grid have the same number of threads and this number is called **block size**. Then, for the simple execution configuration `<<<grid_size, block_size>>>` with two integer numbers.
