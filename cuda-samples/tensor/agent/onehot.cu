#include <iostream>

/**
 * @brief
 *
 * @param B The number of targets.
 * @param N The number of agents.
 * @param T The number of timestamps.
 * @param C The number of classes.
 * @param sdc_index The index of ego.
 * @param object_type_index The index of type. [0: VEHICLE, 1: PEDESTRIAN, 2: CYCLIST].
 * @param target_index The index of target.
 * @param output
 * @return __global__
 */
__global__ void onhot_mask_kernel(
  const int B, const int N, const int T, const int C, const int sdc_index,
  const int * object_type_index, const int * target_index, float * output)
{
  // output [B * N * T * (C + 2)]
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  int n = blockIdx.y * blockDim.y + threadIdx.y;
  int t = blockIdx.z * blockDim.z + threadIdx.z;

  if (b < B && n < N && t < T) {
    int idx = b * N * T + n * T + t;

    // set default value to 0.0
    output[idx] = 0.f;
    output[idx * (C + 2) + object_type_index[n]] = 1.f;

    if (target_index[b] == n) {
      output[idx * (C + 2) + C] = 1.f;
    }

    if (sdc_index == n) {
      output[idx * (C + 2) + C + 1] = 1.f;
    }
  }
}

int main()
{
  constexpr int B = 2;                        // The number of targets
  constexpr int N = 4;                        // The number of agents
  constexpr int T = 5;                        // The number of timestamps
  constexpr int C = 3;                        // The number of classes
  constexpr int sdc_index = 1;                // Index of Ego
  int h_object_type_index[N] = {0, 0, 2, 1};  // [VEHICLE: 0, PEDESTRIAN: 1, CYCLIST: 2]
  int h_target_index[B] = {0, 2};             // Indices of targets

  int *d_object_type_index, *d_target_index;
  float * d_onehot;
  const size_t OutSize = sizeof(float) * B * N * T * (C + 2);
  cudaMalloc(reinterpret_cast<void **>(&d_object_type_index), sizeof(int) * N);
  cudaMalloc(reinterpret_cast<void **>(&d_target_index), sizeof(int) * B);
  cudaMalloc(reinterpret_cast<void **>(&d_onehot), OutSize);
  cudaMemcpy(d_object_type_index, h_object_type_index, sizeof(int) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_target_index, h_target_index, sizeof(int) * B, cudaMemcpyHostToDevice);

  constexpr int threadsPerBlock = 256;
  dim3 blocks(B, N, T);
  onhot_mask_kernel<<<blocks, threadsPerBlock>>>(
    B, N, T, C, sdc_index, d_object_type_index, d_target_index, d_onehot);

  float h_onehot[B][N][T][C + 2];
  cudaMemcpy(h_onehot, d_onehot, OutSize, cudaMemcpyDeviceToHost);

  for (int b = 0; b < B; ++b) {
    std::cout << "Batch " << b << ":\n";
    for (int n = 0; n < N; ++n) {
      std::cout << " Agent " << n << ":\n";
      for (int t = 0; t < T; ++t) {
        std::cout << " Time " << t << ": ";
        for (int i = 0; i < C + 2; ++i) {
          std::cout << h_onehot[b][n][t][i] << " ";
        }
        std::cout << "\n";
      }
    }
  }

  cudaFree(d_object_type_index);
  cudaFree(d_target_index);
  cudaFree(d_onehot);
}

// tensor([[[[1., 0., 0., 1., 0.],
//           [1., 0., 0., 1., 0.],
//           [1., 0., 0., 1., 0.],
//           [1., 0., 0., 1., 0.],
//           [1., 0., 0., 1., 0.]],

//          [[1., 0., 0., 0., 1.],
//           [1., 0., 0., 0., 1.],
//           [1., 0., 0., 0., 1.],
//           [1., 0., 0., 0., 1.],
//           [1., 0., 0., 0., 1.]],

//          [[0., 0., 1., 0., 0.],
//           [0., 0., 1., 0., 0.],
//           [0., 0., 1., 0., 0.],
//           [0., 0., 1., 0., 0.],
//           [0., 0., 1., 0., 0.]],

//          [[0., 1., 0., 0., 0.],
//           [0., 1., 0., 0., 0.],
//           [0., 1., 0., 0., 0.],
//           [0., 1., 0., 0., 0.],
//           [0., 1., 0., 0., 0.]]],

//         [[[1., 0., 0., 0., 0.],
//           [1., 0., 0., 0., 0.],
//           [1., 0., 0., 0., 0.],
//           [1., 0., 0., 0., 0.],
//           [1., 0., 0., 0., 0.]],

//          [[1., 0., 0., 0., 1.],
//           [1., 0., 0., 0., 1.],
//           [1., 0., 0., 0., 1.],
//           [1., 0., 0., 0., 1.],
//           [1., 0., 0., 0., 1.]],

//          [[0., 0., 1., 1., 0.],
//           [0., 0., 1., 1., 0.],
//           [0., 0., 1., 1., 0.],
//           [0., 0., 1., 1., 0.],
//           [0., 0., 1., 1., 0.]],

//          [[0., 1., 0., 0., 0.],
//           [0., 1., 0., 0., 0.],
//           [0., 1., 0., 0., 0.],
//           [0., 1., 0., 0., 0.],
//           [0., 1., 0., 0., 0.]]]])