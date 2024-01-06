#include <chrono>
#include <iostream>

// NOTE: Maximum N is 1024
constexpr int N = 1024;

__global__ void addVec(float * out, float * a, float * b)
{
  int i = threadIdx.x;
  out[i] = a[i] + b[i];
}

__host__ void printVec(float * v)
{
  std::cout << "(";
  for (int i = 0; i < N; ++i) {
    std::cout << v[i];
    if (i != N - 1) {
      std::cout << ", ";
    }
  }
  std::cout << ")";
}

__host__ void initializeVec(const float e, float * v)
{
  for (size_t i = 0; i < N; ++i) {
    v[i] = e;
  }
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
  float *a, *b, *out;
  float *d_a, *d_b, *d_out;

  // Allocate memory
  a = reinterpret_cast<float *>(malloc(sizeof(float) * N));
  b = reinterpret_cast<float *>(malloc(sizeof(float) * N));
  out = reinterpret_cast<float *>(malloc(sizeof(float) * N));

  // Initialize array
  initializeVec(1.0f, a);
  initializeVec(2.0f, b);

  // Allocate device memory
  cudaMalloc(reinterpret_cast<void **>(&d_a), sizeof(float) * N);
  cudaMalloc(reinterpret_cast<void **>(&d_b), sizeof(float) * N);
  cudaMalloc(reinterpret_cast<void **>(&d_out), sizeof(float) * N);

  // Transfer data from host to device memory
  cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);

  // Executing kernel
  std::chrono::system_clock::time_point start, end;
  start = std::chrono::system_clock::now();
  dim3 threadsPerBlock(N);
  dim3 numBlocks(N / threadsPerBlock.x);
  addVec<<<numBlocks, threadsPerBlock>>>(d_out, d_a, d_b);
  end = std::chrono::system_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  // Transfer data back to host memory
  cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

  // Display result
  printVec(out);
  std::cout << "[Elapsed]: " << elapsed << " [ns]" << std::endl;

  // Deallocate device memory
  freeDevice(d_a, d_b, d_out);

  // Deallocate host memory
  freeHost(a, b, out);
}