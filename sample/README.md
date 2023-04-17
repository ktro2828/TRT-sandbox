# CUDA programming samples

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