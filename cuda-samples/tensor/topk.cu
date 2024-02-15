#include <algorithm>
#include <iostream>
#include <vector>

/**
 * NOTE: The behavior changes depending on the number of grids and blocks.
 */

/**
 * @brief Extract topK values and indices with descending order.
 *
 * @param K The number of topK.
 * @param N The number of source elements.
 * @param values Source values, in shape [N].
 * @param topk_values Output topK values.
 * @param topk_index Output topK indices.
 */
__global__ void topk_kernel(
  const int K, const int N, const float * values, float * topk_values, int * topk_index)
{
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;

  // Allocate shared memory
  extern __shared__ float shared_memory[];
  float * shared_values = shared_memory;
  int * shared_index = (int *)&shared_values[blockDim.x];

  // Load values and indices into shared memory
  if (tidx < N) {
    shared_values[threadIdx.x] = values[tidx];
    shared_index[threadIdx.x] = tidx;
  } else {
    shared_values[threadIdx.x] = INFINITY;
    shared_index[threadIdx.x] = -1;
  }

  __syncthreads();

  // Parallel reduction to find top-K elements
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      if (shared_values[threadIdx.x + stride] < shared_values[threadIdx.x]) {
        shared_values[threadIdx.x] = shared_values[threadIdx.x + stride];
        shared_index[threadIdx.x] = shared_index[threadIdx.x + stride];
      }
    }
    __syncthreads();
  }

  // Store the top-K elements
  if (threadIdx.x == 0) {
    topk_values[blockIdx.x] = shared_values[0];
    topk_index[blockIdx.x] = shared_index[0];
  }
}

int main()
{
  constexpr int N = 100000;
  constexpr int K = 5;

  std::vector<float> values(N);
  for (int i = 0; i < N; ++i) {
    values[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  float * d_values;
  cudaMalloc(reinterpret_cast<void **>(&d_values), sizeof(float) * N);
  cudaMemcpy(d_values, values.data(), sizeof(float) * N, cudaMemcpyHostToDevice);

  float * d_topk_values;
  int * d_topk_index;
  cudaMalloc(reinterpret_cast<void **>(&d_topk_values), sizeof(float) * K);
  cudaMalloc(reinterpret_cast<void **>(&d_topk_index), sizeof(int) * K);

  constexpr int threadsPerBlock = 1024;  // must be <=1024
  const int numBlocks = 128;             // must be K<=,<=128
  size_t sharedMemSize = 2 * threadsPerBlock * sizeof(float);

  topk_kernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(
    K, N, d_values, d_topk_values, d_topk_index);

  cudaDeviceSynchronize();

  float h_topk_values[K];
  int h_topk_index[K];
  cudaMemcpy(h_topk_values, d_topk_values, sizeof(float) * K, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_topk_index, d_topk_index, sizeof(int) * K, cudaMemcpyDeviceToHost);

  for (int k = 0; k < K; ++k) {
    std::cout << k << ": (idx, value)=(" << h_topk_index[k] << ", " << h_topk_values[k] << ")\n";
  }

  cudaFree(d_values);
  cudaFree(d_topk_values);
  cudaFree(d_topk_index);
}