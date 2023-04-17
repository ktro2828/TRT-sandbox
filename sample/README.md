# CUDA programming samples

## Run
- Compile with `cmake`
```shell
$ mkdir build && cd build
$ cmake .. && make
```

- Compile directly with `nvcc`
```shell
$ nvcc vector_add.cu -o vector_add
$ ./vector_add
```

## Appendix

### [Execution configuration](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#execution-configuration)

All call to a `__global__` function must specify the execution configuration of that call. 
It is specified by `kernel_function<<<Dg, Db, Ns, S>>>`

- **Dg**(`dim3`): The dimension and size of the grid, such that `Dg.x * Dg.y * Dg.z` equals the number of blocks being launched.
- **Db**(`dim3`): The dimension and size of each block, such that `Db.x * Db.y * Db.z` equals the number of threads per block.
- **Ns**(`size_t`): The number of bytes in shared memory that is dynamically allocated per block. `Ns` is an optional argument which defaults to `0`.
- **S**(`cudaStream_t`): The associated stream. `S` is and optional argument which defaults to `0`.