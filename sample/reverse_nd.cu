#include <iostream>

__global__ void reverse(float * d_a, float * d_b, const size_t N)
{
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  d_b[N - 1 - idx] = d_a[idx];
}

int main()
{
  constexpr size_t N = 256;
  float *h_a, *h_b;
  float *d_a, *d_b;

  h_a = reinterpret_cast<float *>(malloc(sizeof(float) * N));
  h_b = reinterpret_cast<float *>(malloc(sizeof(float) * N));

  for (size_t i = 0; i < N; ++i) {
    h_a[i] = static_cast<float>(i);
  }

  cudaMalloc(reinterpret_cast<void **>(&d_a), sizeof(float) * N);
  cudaMalloc(reinterpret_cast<void **>(&d_b), sizeof(float) * N);

  cudaMemcpy(d_a, h_a, sizeof(float) * N, cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(N);
  dim3 numBlocks(N / threadsPerBlock.x);
  std::cout << "numBlocks: " << numBlocks.x << ", threadsPerBlock: " << threadsPerBlock.x
            << std::endl;
  reverse<<<numBlocks, threadsPerBlock>>>(d_a, d_b, N);

  cudaMemcpy(h_b, d_b, sizeof(float) * N, cudaMemcpyDeviceToHost);

  std::cout << "(";
  for (size_t i = 0; i < N; ++i) {
    std::cout << h_b[i] << ", ";
  }
  std::cout << ")" << std::endl;

  cudaFree(d_a);
  cudaFree(d_b);
}