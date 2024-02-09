#include <cmath>
#include <iostream>

/**
 * @brief Genreate embeddings for timestamps and headings.
 *
 * @param B The number of targets.
 * @param N The number of agents.
 * @param T The number of timestamps.
 * @param D The number of trajectory dimensions.
 * @param timestamps The array of timestamps, in shape (T).
 * @param trajectory The array of trajectory, in shape (B*N*T*D).
 * @param out_time Output time embeddings, in shape (B*N*T*(T+1)).
 * @param out_yaw Output heading embeddings, in shape (B*N*T*2), ordering (sin, cos).
 * @return __global__
 */
__global__ void generate_embedding_kernel(
  const int B, const int N, const int T, const int D, const float * timestamps,
  const float * trajectory, float * out_time, float * out_yaw)
{
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  int n = blockIdx.y * blockDim.y + threadIdx.y;
  int t = blockIdx.z * blockDim.z + threadIdx.z;

  if (b < B && n < N && t < T) {
    // NOTE: time embedding is OK
    const int idx = b * N * T + n * T + t;
    out_time[idx] = 0.0f;
    out_time[idx * (T + 1) + t] = 1.0f;
    out_time[idx * (T + 1) + T] = timestamps[t];

    // TODO
    const float yaw = trajectory[idx * D + 6];
    out_yaw[idx] = 0.0f;
    out_yaw[idx * 2] = std::sin(yaw);
    out_yaw[idx * 2 + 1] = std::cos(yaw);
  }
}

int main()
{
  constexpr int B = 2;   // Batch size
  constexpr int N = 3;   // The number of agents
  constexpr int T = 5;   // The number of timestamps
  constexpr int D = 10;  // The number of state dimensions
  float h_timestamps[T] = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  float h_trajectory[B][N][T][D] = {
    {{{1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 1.0f, 0.5 * M_PI, 3.0f, 0.1f, 0.0f},
      {2.0f, 3.0f, 0.1f, 0.2f, 1.0f, 2.0f, 0.5 * M_PI, 0.1f, 0.2f, 1.0f},
      {1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 1.0f, 0.5 * M_PI, 3.0f, 0.1f, 0.0f},
      {1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 1.0f, 0.5 * M_PI, 3.0f, 0.1f, 1.0f},
      {1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 1.0f, 0.5 * M_PI, 3.0f, 0.1f, 1.0f}},
     {{2.0f, 2.0f, 3.0f, 0.1f, 0.2f, 1.0f, 0.5 * M_PI, 3.0f, 0.1f, 0.0f},
      {1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 1.0f, 0.5 * M_PI, 3.0f, 0.1f, 0.0f},
      {1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 1.0f, 0.5 * M_PI, 3.0f, 0.1f, 0.0f},
      {1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 1.0f, 0.5 * M_PI, 3.0f, 0.1f, 0.0f},
      {1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 1.0f, 0.5 * M_PI, 3.0f, 0.1f, 0.0f}},
     {{2.0f, 2.0f, 3.0f, 0.1f, 0.2f, 1.0f, 0.5 * M_PI, 3.0f, 0.1f, 0.0f},
      {1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 1.0f, 0.5 * M_PI, 3.0f, 0.1f, 0.0f},
      {1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 1.0f, 0.5 * M_PI, 3.0f, 0.1f, 0.0f},
      {1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 1.0f, 0.5 * M_PI, 3.0f, 0.1f, 0.0f},
      {1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 1.0f, 0.5 * M_PI, 3.0f, 0.1f, 0.0f}}},
    {{{1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 1.0f, 0.5 * M_PI, 3.0f, 0.1f, 0.0f},
      {2.0f, 3.0f, 0.1f, 0.2f, 1.0f, 2.0f, 0.5 * M_PI, 0.1f, 0.2f, 1.0f},
      {1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 1.0f, 0.5 * M_PI, 3.0f, 0.1f, 0.0f},
      {1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 1.0f, 0.5 * M_PI, 3.0f, 0.1f, 1.0f},
      {1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 1.0f, 0.5 * M_PI, 3.0f, 0.1f, 1.0f}},
     {{2.0f, 2.0f, 3.0f, 0.1f, 0.2f, 1.0f, 0.5 * M_PI, 3.0f, 0.1f, 0.0f},
      {1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 1.0f, 0.5 * M_PI, 3.0f, 0.1f, 0.0f},
      {1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 1.0f, 0.5 * M_PI, 3.0f, 0.1f, 0.0f},
      {1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 1.0f, 0.5 * M_PI, 3.0f, 0.1f, 0.0f},
      {1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 1.0f, 0.5 * M_PI, 3.0f, 0.1f, 0.0f}},
     {{2.0f, 2.0f, 3.0f, 0.1f, 0.2f, 1.0f, 0.5 * M_PI, 3.0f, 0.1f, 0.0f},
      {1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 1.0f, 0.5 * M_PI, 3.0f, 0.1f, 0.0f},
      {1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 1.0f, 0.5 * M_PI, 3.0f, 0.1f, 0.0f},
      {1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 1.0f, 0.5 * M_PI, 3.0f, 0.1f, 0.0f},
      {1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 1.0f, 0.5 * M_PI, 3.0f, 0.1f, 0.0f}}}};

  float *d_timestamps, *d_trajectory, *d_out_time, *d_out_yaw;
  const size_t InTrajSize = sizeof(float) * N * T * D;
  const size_t OutTimeSize = sizeof(float) * B * N * T * (T + 1);
  const size_t OutYawSize = sizeof(float) * B * N * T * 2;
  cudaMalloc(reinterpret_cast<void **>(&d_timestamps), sizeof(float) * T);
  cudaMalloc(reinterpret_cast<void **>(&d_trajectory), InTrajSize);
  cudaMalloc(reinterpret_cast<void **>(&d_out_time), OutTimeSize);
  cudaMalloc(reinterpret_cast<void **>(&d_out_yaw), OutYawSize);
  cudaMemcpy(d_timestamps, h_timestamps, sizeof(float) * T, cudaMemcpyHostToDevice);
  cudaMemcpy(d_trajectory, h_trajectory, InTrajSize, cudaMemcpyHostToDevice);

  constexpr int threadsPerBlock = 256;
  dim3 blocks(B, N, T);

  generate_embedding_kernel<<<blocks, threadsPerBlock>>>(
    B, N, T, D, d_timestamps, d_trajectory, d_out_time, d_out_yaw);

  float h_out_time[B][N][T][T + 1], h_out_yaw[B][N][T][2];
  cudaMemcpy(h_out_time, d_out_time, OutTimeSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_out_yaw, d_out_yaw, OutYawSize, cudaMemcpyDeviceToHost);

  std::cout << "Time embedding" << std::endl;
  for (int b = 0; b < B; ++b) {
    std::cout << "Batch " << b << ":\n";
    for (int n = 0; n < N; ++n) {
      std::cout << " Agent " << n << ":\n";
      for (int t = 0; t < T; ++t) {
        std::cout << " Time " << t << ": ";
        for (int i = 0; i < T + 1; ++i) {
          std::cout << h_out_time[b][n][t][i] << " ";
        }
        std::cout << "\n";
      }
    }
  }

  std::cout << "Yaw embedding" << std::endl;
  for (int b = 0; b < B; ++b) {
    std::cout << "Batch " << b << ":\n";
    for (int n = 0; n < N; ++n) {
      std::cout << " Agent " << n << ":\n";
      for (int t = 0; t < T; ++t) {
        std::cout << " Time " << t << ": ";
        for (int i = 0; i < 2; ++i) {
          std::cout << h_out_yaw[b][n][t][i] << " ";
        }
        std::cout << "\n";
      }
    }
  }

  cudaFree(d_timestamps);
  cudaFree(d_trajectory);
  cudaFree(d_out_time);
  cudaFree(d_out_yaw);
}

// ----- Expectaions -----
// TIME embedding
// tensor([[[[1., 0., 0., 0., 0., 2.],
//           [0., 1., 0., 0., 0., 3.],
//           [0., 0., 1., 0., 0., 4.],
//           [0., 0., 0., 1., 0., 5.],
//           [0., 0., 0., 0., 1., 6.]],

//          [[1., 0., 0., 0., 0., 2.],
//           [0., 1., 0., 0., 0., 3.],
//           [0., 0., 1., 0., 0., 4.],
//           [0., 0., 0., 1., 0., 5.],
//           [0., 0., 0., 0., 1., 6.]],

//          [[1., 0., 0., 0., 0., 2.],
//           [0., 1., 0., 0., 0., 3.],
//           [0., 0., 1., 0., 0., 4.],
//           [0., 0., 0., 1., 0., 5.],
//           [0., 0., 0., 0., 1., 6.]]],

//         [[[1., 0., 0., 0., 0., 2.],
//           [0., 1., 0., 0., 0., 3.],
//           [0., 0., 1., 0., 0., 4.],
//           [0., 0., 0., 1., 0., 5.],
//           [0., 0., 0., 0., 1., 6.]],

//          [[1., 0., 0., 0., 0., 2.],
//           [0., 1., 0., 0., 0., 3.],
//           [0., 0., 1., 0., 0., 4.],
//           [0., 0., 0., 1., 0., 5.],
//           [0., 0., 0., 0., 1., 6.]],

//          [[1., 0., 0., 0., 0., 2.],
//           [0., 1., 0., 0., 0., 3.],
//           [0., 0., 1., 0., 0., 4.],
//           [0., 0., 0., 1., 0., 5.],
//           [0., 0., 0., 0., 1., 6.]]]])

// YAW embedding
// tensor([[[[ 1.0000e+00, -4.3711e-08],
//           [ 1.0000e+00, -4.3711e-08],
//           [ 1.0000e+00, -4.3711e-08],
//           [ 1.0000e+00, -4.3711e-08],
//           [ 1.0000e+00, -4.3711e-08]],

//          [[ 1.0000e+00, -4.3711e-08],
//           [ 1.0000e+00, -4.3711e-08],
//           [ 1.0000e+00, -4.3711e-08],
//           [ 1.0000e+00, -4.3711e-08],
//           [ 1.0000e+00, -4.3711e-08]],

//          [[ 1.0000e+00, -4.3711e-08],
//           [ 1.0000e+00, -4.3711e-08],
//           [ 1.0000e+00, -4.3711e-08],
//           [ 1.0000e+00, -4.3711e-08],
//           [ 1.0000e+00, -4.3711e-08]]],

//         [[[ 1.0000e+00, -4.3711e-08],
//           [ 1.0000e+00, -4.3711e-08],
//           [ 1.0000e+00, -4.3711e-08],
//           [ 1.0000e+00, -4.3711e-08],
//           [ 1.0000e+00, -4.3711e-08]],

//          [[ 1.0000e+00, -4.3711e-08],
//           [ 1.0000e+00, -4.3711e-08],
//           [ 1.0000e+00, -4.3711e-08],
//           [ 1.0000e+00, -4.3711e-08],
//           [ 1.0000e+00, -4.3711e-08]],

//          [[ 1.0000e+00, -4.3711e-08],
//           [ 1.0000e+00, -4.3711e-08],
//           [ 1.0000e+00, -4.3711e-08],
//           [ 1.0000e+00, -4.3711e-08],
//           [ 1.0000e+00, -4.3711e-08]]]])
