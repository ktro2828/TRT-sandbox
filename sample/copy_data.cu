#include <iostream>

constexpr size_t N = 1024;

/**
 * Exercize 1
 * 1. デバイス上にポインタ`d_a`と`d_b`のメモリを割当
 * 2. ホスト上の`h_a`をデバイス上の`d_a`にコピー
 * 3. `d_a`から`d_b`へデバイス間のコピー
 * 4. デバイス上の`d_b`をホスト上の`h_a`にコピーバック
 * 5. ホスト上の`d_a`と`d_b`のメモリを解放
 */
int main()
{
  float * h_a;
  float *d_a, *d_b;
  h_a = reinterpret_cast<float *>(malloc(sizeof(float) * N));

  // initialize array
  for (size_t i = 0; i < N; ++i) {
    h_a[i] = 1.0f;
  }

  cudaMalloc(reinterpret_cast<void **>(&d_a), sizeof(float) * N);
  cudaMalloc(reinterpret_cast<void **>(&d_b), sizeof(float) * N);

  cudaMemcpy(d_a, h_a, sizeof(float) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, d_a, sizeof(float) * N, cudaMemcpyDeviceToDevice);
  cudaMemcpy(h_a, d_b, sizeof(float) * N, cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);

  std::cout << "(";
  for (size_t i = 0; i < N; ++i) {
    std::cout << h_a[i] << ", ";
  }
  std::cout << ")" << std::endl;

  free(h_a);
}