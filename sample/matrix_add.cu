#include <chrono>
#include <iostream>

constexpr size_t N = 1024;
constexpr size_t M = 512;

__global__ void addMat(float * A, float * B, float * C)
{
  const size_t row = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t col = blockIdx.y;
  if (row < N && col < M) {
    C[row * col] = A[row * col] + B[row * col];
  }
}

__host__ void initializeMat(const float e, float * dst)
{
  for (size_t i = 0; i < N * M; ++i) {
    dst[i] = e;
  }
}

__host__ void printMat(float * src)
{
  std::cout << "(";
  for (size_t i = 0; i < N; ++i) {
    std::cout << "(";
    for (size_t j = 0; j < M; ++j) {
      std::cout << src[i * j];
      if (j != M - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "),\n";
  }
  std::cout << ")" << std::endl;
}

template <typename... Ptrs>
__host__ void freeHost(Ptrs... ptrs)
{
  for (auto p : std::initializer_list<float *>{ptrs...}) {
    free(p);
  }
}

template <typename... Ptrs>
__host__ void freeDevice(Ptrs... ptrs)
{
  for (auto p : std::initializer_list<float *>{ptrs...}) {
    cudaFree(p);
  }
}

int main()
{
  float *A, *B, *C;
  float *d_A, *d_B, *d_C;

  // Allocate memory
  A = reinterpret_cast<float *>(malloc(sizeof(float) * N * M));
  B = reinterpret_cast<float *>(malloc(sizeof(float) * N * M));
  C = reinterpret_cast<float *>(malloc(sizeof(float) * N * M));

  initializeMat(1.0f, A);
  initializeMat(2.0f, B);

  // Allocate device memory
  cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(float) * N * M);
  cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(float) * N * M);
  cudaMalloc(reinterpret_cast<void **>(&d_C), sizeof(float) * N * M);

  // Copy data from host to device
  cudaMemcpy(d_A, A, sizeof(float) * N * M, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, sizeof(float) * N * M, cudaMemcpyHostToDevice);

  std::chrono::system_clock::time_point start, end;
  start = std::chrono::system_clock::now();
  dim3 threadsPerBlock(N);
  dim3 numBlocks(N / threadsPerBlock.x, M / threadsPerBlock.y);
  addMat<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C);
  end = std::chrono::system_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  // Copy data back to host memory
  cudaMemcpy(C, d_C, sizeof(float) * N * M, cudaMemcpyDeviceToHost);

  // Display result
  printMat(C);
  std::cout << "[Elapsed]: " << elapsed << " [ns]" << std::endl;

  // Deallocate device memory
  freeDevice(d_A, d_B, d_C);

  // Deallocate host memory
  freeHost(A, B, C);
}