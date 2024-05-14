#include <cub/cub.cuh>

#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

__device__ float distanceState(const float * state1, const float * state2)
{
  float x1 = state1[0];
  float y1 = state1[1];
  float x2 = state2[0];
  float y2 = state2[1];
  return hypotf(x2 - x1, y2 - y1);
}

template <unsigned int BLOCK_THREADS, unsigned int ITEMS_PER_THREAD>
__global__ void batchNMSKernel(
  const int B, const int M, const int T, const int D, const float * inScores, const float * inTrajs,
  const float threshold, const int K, float * outScores, float * outTrajs)
{
  int b = blockIdx.x;                             // Batch index
  int tid = threadIdx.x;                          // Thread local index in this CUDA block
  int t = blockIdx.y * blockDim.y + threadIdx.y;  // Timestamp index
  int d = blockIdx.z * blockDim.z + threadIdx.z;  // Dimension index

  using BlockRadixSortT = cub::BlockRadixSort<float, BLOCK_THREADS, ITEMS_PER_THREAD, unsigned int>;
  using TempStorageT = typename BlockRadixSortT::TempStorage;

  __shared__ TempStorageT temp_storage;

  float scores[ITEMS_PER_THREAD] = {0.0f};
  unsigned int score_indices[ITEMS_PER_THREAD] = {0};
  for (unsigned int i = 0; i < ITEMS_PER_THREAD; ++i) {
    int mode_idx = BLOCK_THREADS * i + tid;
    int score_idx = b * M + mode_idx;
    score_indices[i] = mode_idx;
    scores[i] = (mode_idx < M && score_idx < B * M) ? inScores[score_idx] : -FLT_MAX;
  }

  // Sort modes by its score in descending order
  BlockRadixSortT(temp_storage).SortDescending(scores, score_indices);
  // Block-wide barrier necessary to refer to the sort result
  __syncthreads();

  bool masks[ITEMS_PER_THREAD] = {true};
  for (unsigned int i = 0; i < ITEMS_PER_THREAD; ++i) {
    masks[i] = true;
    int cur_mode_idx = score_indices[i];
    int tgt_mode_idx = tid * ITEMS_PER_THREAD + i;
    if (cur_mode_idx == tgt_mode_idx || fabs(-FLT_MAX - scores[i]) < FLT_EPSILON) {
      continue;
    }
    int cur_traj_idx = ((b * M + cur_mode_idx) * T + T - 1) * D;
    int tgt_traj_idx = ((b * M + tgt_mode_idx) * T + T - 1) * D;
    if (distanceState(&inTrajs[cur_traj_idx], &inTrajs[tgt_traj_idx]) < threshold) {
      masks[i] = false;
    }
  }
  __syncthreads();

  unsigned int j = 0;
  for (unsigned int i = 0; i < ITEMS_PER_THREAD; ++i) {
    int cur_mode_idx = tid * ITEMS_PER_THREAD + j;
    if (cur_mode_idx >= K || fabs(-FLT_MAX - scores[i]) < FLT_EPSILON || !masks[i]) {
      continue;
    }
    int in_idx = b * M + score_indices[i];
    int out_idx = b * K + cur_mode_idx;
    outScores[out_idx] = inScores[in_idx];
    outTrajs[(out_idx * T + t) * D + d] = inTrajs[(in_idx * T + t) * D + d];
    ++j;
  }
}

int main()
{
  constexpr int B = 100;  // 2
  constexpr int M = 512;  // 3
  constexpr int T = 80;   // 4
  constexpr int D = 7;

  constexpr float threshold = 1.5f;  // 1.414 <= dist <= 2.828
  constexpr int K = 6;

  // float h_inScores[B][M] = {{0.1, 0.3, 0.3}, {0.2, 0.4, 0.1}};
  // float h_inTrajs[B][M][T][D] = {
  //   {
  //     {
  //       {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
  //       {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 1.0f},
  //       {3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 1.0f},
  //       {4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 1.0f},
  //     },
  //     {
  //       {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 0.0f},
  //       {3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 0.0f},
  //       {4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 1.0f},
  //       {5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 0.0f},
  //     },
  //     {
  //       {3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 1.0f},
  //       {4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 1.0f},
  //       {5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 1.0f},
  //       {6.0f, 6.0f, 6.0f, 6.0f, 6.0f, 6.0f, 1.0f},
  //     },
  //   },
  //   {
  //     {
  //       {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
  //       {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 1.0f},
  //       {3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 1.0f},
  //       {4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 1.0f},
  //     },
  //     {
  //       {3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 1.0f},
  //       {4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 1.0f},
  //       {5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 1.0f},
  //       {6.0f, 6.0f, 6.0f, 6.0f, 6.0f, 6.0f, 1.0f},
  //     },
  //     {
  //       {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 0.0f},
  //       {3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 0.0f},
  //       {4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 1.0f},
  //       {5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 0.0f},
  //     },
  //   }};

  std::random_device seed_gen;
  std::mt19937 engine(seed_gen());
  std::uniform_real_distribution<float> score_dist(0.0f, 1.0f);
  std::uniform_real_distribution<float> traj_dist(-2.0f, 2.0f);
  std::vector<float> inScores, inTrajs;
  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      inScores.push_back(score_dist(engine));
      for (int t = 0; t < T; ++t) {
        for (int d = 0; d < D; ++d) {
          inTrajs.push_back(traj_dist(engine));
        }
      }
    }
  }
  float *h_inScores = inScores.data(), *h_inTrajs = inTrajs.data();

  float *d_inScores, *d_inTrajs, *d_outScores, *d_outTrajs;
  cudaMalloc(&d_inScores, sizeof(float) * B * M);
  cudaMalloc(&d_inTrajs, sizeof(float) * B * M * T * D);
  cudaMemcpy(d_inScores, h_inScores, sizeof(float) * B * M, cudaMemcpyHostToDevice);
  cudaMemcpy(d_inTrajs, h_inTrajs, sizeof(float) * B * M * T * D, cudaMemcpyHostToDevice);

  cudaMalloc(&d_outScores, sizeof(float) * B * K);
  cudaMalloc(&d_outTrajs, sizeof(float) * B * K * T * D);

  constexpr unsigned int threadsPerBlock = 256;
  dim3 blocks(B, T, D);
  constexpr unsigned int itemsPerThread = 24;
  if (threadsPerBlock * itemsPerThread < M) {
    std::cerr << "Larger M (" << M << ") than acceptable range (<"
              << threadsPerBlock * itemsPerThread << ") detected." << std::endl;
    return -1;
  }

  float h_outScores[B][K], h_outTrajs[B][K][T][D];
  for (int i = 0; i < 10; ++i) {
    clock_t start = clock();

    batchNMSKernel<threadsPerBlock, itemsPerThread><<<blocks, threadsPerBlock>>>(
      B, M, T, D, d_inScores, d_inTrajs, threshold, K, d_outScores, d_outTrajs);

    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cerr << cudaGetErrorString(err) << std::endl;
      return -1;
    }

    cudaMemcpy(h_outScores, d_outScores, sizeof(float) * B * K, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_outTrajs, d_outTrajs, sizeof(float) * B * K * T * D, cudaMemcpyDeviceToHost);

    clock_t end = clock();

    const double time = static_cast<double>(end - start) / CLOCKS_PER_SEC * 1000.0;
    printf("time %lf[ms]\n", time);
  }

  // std::cout << "=== Out score ===\n";
  // for (int b = 0; b < B; ++b) {
  //   std::cout << "Batch " << b << ":\n";
  //   for (int k = 0; k < K; ++k) {
  //     std::cout << h_outScores[b][k] << " ";
  //   }
  //   std::cout << "\n";
  //   for (int m = 0; m < M; ++m) {
  //     std::cout << *(h_inScores + b * M + m) << " ";
  //   }
  //   std::cout << "\n";
  // }

  // std::cout << "=== Out trajectory ===\n";
  // for (int b = 0; b < B; ++b) {
  //   std::cout << "Batch " << b << ":\n";
  //   for (int k = 0; k < K; ++k) {
  //     std::cout << "  Mode " << k << ":\n";
  //     for (int t = 0; t < T; ++t) {
  //       std::cout << "  Time " << t << ": ";
  //       for (int d = 0; d < D; ++d) {
  //         std::cout << h_outTrajs[b][k][t][d] << " ";
  //       }
  //       std::cout << "\n";
  //     }
  //   }
  // }

  cudaFree(d_inScores);
  cudaFree(d_inTrajs);
  cudaFree(d_outScores);
  cudaFree(d_inTrajs);
}
