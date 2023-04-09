# TRT-sandbox

A sandbox of TensorRT.

## Build & Run

```shell
# Build
$ mkdir build && cd build
$ cmake ..
$ make -j$(nproc)

# Run
$ ./src/det2d IMG_PATH ENGINE_OR_ONNX_PATH
```