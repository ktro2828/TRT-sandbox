#include <iostream>

__global__ void fooKernl(float * d_a)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  d_a[idx] = 1000 * blockIdx.x + threadIdx.x;
}

/**
 * Exercise 2
 * 1. `d_a`を定義し、カーネル結果に対するデバイスメモリを割当
 * 2. 1-Dスレッドブロックの1-Dグリッドを使用して、カーネルを構成し起動
 * 3. 各スレッドで以下のように`d_a`の要素を設定
 * ```cu
 * idx = blockIdx.x * blockDim.x + threadIdx.x;
 * d_a[idx] = 1000 * blockIdx.x + threadIdx.x;
 * ```
 * 4. `d_a`をホスト上の`h_a`にコピーバック
 */
int main()
{
  constexpr size_t N = 1024;

  float *h_a, *d_a;

  h_a = reinterpret_cast<float *>(malloc(sizeof(float) * N));

  for (size_t i = 0; i < N; ++i) {
    h_a[i] = 1.0f;
  }

  cudaMalloc(reinterpret_cast<void **>(&d_a), sizeof(float) * N);

  cudaMemcpy(d_a, h_a, sizeof(float) * N, cudaMemcpyHostToDevice);

  fooKernl<<<1, 32>>>(d_a);

  cudaMemcpy(h_a, d_a, sizeof(float) * N, cudaMemcpyDeviceToHost);

  cudaFree(d_a);

  std::cout << "(";
  for (size_t i = 0; i < N; ++i) {
    std::cout << h_a[i] << ", ";
  }
  std::cout << ")" << std::endl;

  free(h_a);
}