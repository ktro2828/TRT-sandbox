# TRT-sandbox

A sandbox of TensorRT.

## Build & Run

```shell
# Build
$ git clone https://github.com/ktro2828/trt-sandbox.git --recursive
$ cd trt-sandbox
$ cmake -B build
$ cmake --build build -j$(nproc)

# Run
$ ./src/det2d IMG_PATH ENGINE_OR_ONNX_PATH LABEL_PATH CONFIG_PATH
```
