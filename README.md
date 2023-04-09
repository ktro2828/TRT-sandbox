# TRT-sandbox

A sandbox of TensorRT.

## Build & Run

```shell
# Build
$ mkdir build && cd build
$ cmake ..
$ make -j$(nproc)

# Run
$ ./src/main IMG_PATH ENGINE_OR_ONNX_PATH
```