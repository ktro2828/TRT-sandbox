# Chapter 4: Error checking in CUDA programs

## A macro function checking CUDA runtime API functions

CUDAのランタイムAPIの中のいくつかの関数は戻り値`cudaError_t`を返すので、それを基に各ランタイムAPIのエラー確認を行うマクロを定義する。(`error.cuh`)

```c++
#pragma once
#include <stdio.h>

#define CHECK(call)                                                 \
  do {                                                              \
    const cudaError_t error_code = call;                            \
    if (error_code != cudaSuccess) {                                \
      printf("CUDA Error:\n");                                      \
      printf("  File:   %s\n", __FILE__);                           \
      printf("  Line:   %d\n", __LINE__);                           \
      printf("  Error code: %d\n", error_code);                     \
      printf("  Error text: %s\n", cudaGetErrorString(error_code)); \
      exit(1);                                                      \
    }                                                               \
  } while (0)
```

### Checking CUDA runtime API functions using the macro function

チャプター3の`add2wrong.cu`を使って、`check1api.cu`を作成し実行してみると、以下のようなエラーが表示される。

```shell
$ nvcc check1api.cu && ./a.out
```

```shell
CUDA Error:
  File:   check1api.cu
  Line:   30
  Error code: 1
  Error text: invalid argument
```

### Checking CUDA kernels using the macro function

CUDAカーネル自体には戻り値を指定できないため、上のマクロを直接使うことはできないが、以下の2つを追加することで各カーネル呼び出しに対してエラーチェックができる。

```c++
CHECK(cudaGetLastError()); // ↓を実行する前に発生した最後のエラーをキャプチャ
CHECK(cudaDeviceSynchronize()); // ホストとデバイス間を同期
```

ホスト上では、カーネル実行後CUDAカーネルの実行とは非同期になるため`cudaDeviceSynchronize()`で強制的にカーネル実行が終了するまでホストを待機させる。
`check2kernel.cu`では、CUDAでは割り当て可能な最大ブロック数が`1024`なのに対して`1280`を割り当てているためエラーになる。

```shell
$ nvcc check2kernel.cu && ./a.out
```

```shell
CUDA Error:
  File:   check2kernel.cu
  Line:   36
  Error code: 9
  Error text: invalid configuration argument
```

`cudaDeviceSynchronize()`は強制的な同期処理によってCUDAプログラムのパフォーマンスを低下させる可能があるため、デバッグする際には`export CUDA_LAUNCH_BLOCKING=1`としてから実行してデバッグしたほうがいい。

## Using CUDA-MEMCHECK to check memory errors

`CUDA_LAUNCH_BLOCKING=1`以外のデバッグ方法として、`cuda-memcheck`または`compute-sanitizer`がある。
