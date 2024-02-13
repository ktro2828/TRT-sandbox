#include <iostream>

/**
 * @brief Transform the trajectory coords to the coords system centered around the target object.
 *
 * @param B
 * @param N
 * @param T
 * @param D
 * @param in_trajectory
 * @param center_xyz
 * @param center_yaw
 * @param output
 */
__global__ void transform_trajectory_kernel(
  const int B, const int N, const int T, const int D, const float * in_trajectory,
  const float * center_xyz, const float * center_yaw, float * output)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N * T) {
    const float x = in_trajectory[idx * D];
    const float y = in_trajectory[idx * D + 1];
    const float z = in_trajectory[idx * D + 2];
    const float dx = in_trajectory[idx * D + 3];
    const float dy = in_trajectory[idx * D + 4];
    const float dz = in_trajectory[idx * D + 5];
    const float yaw = in_trajectory[idx * D + 6];
    const float vx = in_trajectory[idx * D + 7];
    const float vy = in_trajectory[idx * D + 8];
    const float is_valid = in_trajectory[idx * D + 9];

    // transform for each target
    for (int b = 0; b < B; ++b) {
      const float center_x = center_xyz[b * 3];
      const float center_y = center_xyz[b * 3 + 1];
      const float center_z = center_xyz[b * 3 + 2];
      const float cos_val = std::cos(center_yaw[b]);
      const float sin_val = std::sin(center_yaw[b]);

      // transform
      const float trans_x = cos_val * (x - center_x) - sin_val * (y - center_y);
      const float trans_y = sin_val * (x - center_x) + cos_val * (y - center_y);
      const float trans_z = z - center_z;
      const float trans_yaw = yaw - center_yaw[b];
      const float trans_vx = cos_val * vx - sin_val * vy;
      const float trans_vy = sin_val * vx + cos_val * vy;

      const int trans_idx = (b * N * T + idx) * D;
      output[trans_idx] = trans_x;
      output[trans_idx + 1] = trans_y;
      output[trans_idx + 2] = trans_z;
      output[trans_idx + 3] = dx;
      output[trans_idx + 4] = dy;
      output[trans_idx + 5] = dz;
      output[trans_idx + 6] = trans_yaw;
      output[trans_idx + 7] = trans_vx;
      output[trans_idx + 8] = trans_vy;
      output[trans_idx + 9] = is_valid;
    }
  }
}

__global__ void extract_last_pos_kernel(
  const int B, const int N, const int T, const int D, const float * in_trajectory, float * output)
{
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  int n = blockIdx.y * blockDim.y + threadIdx.y;
  int t = blockIdx.z * blockDim.z + threadIdx.z;
  if (b < B && t == T - 1) {
    const int idx = b * N * T + n * T + t;
    const int out_idx = b * N + n;
    output[out_idx] = 0.0f;
    output[out_idx * 3] = in_trajectory[idx * D];
    output[out_idx * 3 + 1] = in_trajectory[idx * D + 1];
    output[out_idx * 3 + 2] = in_trajectory[idx * D + 2];
  }
}

int main()
{
  constexpr int B = 2;   // Batch size
  constexpr int N = 3;   // The number of agents
  constexpr int T = 5;   // The number of timestamps
  constexpr int D = 10;  // The number of state dimensions
  float h_src[N][T][D] = {
    {{1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 1.0f, 2.0f, 3.0f, 0.1f, 0.0f},
     {2.0f, 3.0f, 0.1f, 0.2f, 1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 1.0f},
     {1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 1.0f, 2.0f, 3.0f, 0.1f, 0.0f},
     {1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 1.0f, 2.0f, 3.0f, 0.1f, 1.0f},
     {1.0f, 1.0f, 1.0f, 0.1f, 0.2f, 1.0f, 2.0f, 3.0f, 0.1f, 1.0f}},
    {{2.0f, 2.0f, 3.0f, 0.1f, 0.2f, 1.0f, 2.0f, 3.0f, 0.1f, 0.0f},
     {1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 1.0f, 2.0f, 3.0f, 0.1f, 0.0f},
     {1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 1.0f, 2.0f, 3.0f, 0.1f, 0.0f},
     {1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 1.0f, 2.0f, 3.0f, 0.1f, 0.0f},
     {2.0f, 2.0f, 2.0f, 0.1f, 0.2f, 1.0f, 2.0f, 3.0f, 0.1f, 0.0f}},
    {{2.0f, 2.0f, 3.0f, 0.1f, 0.2f, 1.0f, 2.0f, 3.0f, 0.1f, 0.0f},
     {1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 1.0f, 2.0f, 3.0f, 0.1f, 0.0f},
     {1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 1.0f, 2.0f, 3.0f, 0.1f, 0.0f},
     {1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 1.0f, 2.0f, 3.0f, 0.1f, 0.0f},
     {3.0f, 3.0f, 3.0f, 0.1f, 0.2f, 1.0f, 2.0f, 3.0f, 0.1f, 0.0f}}};

  float h_xyz[B][3] = {{1.0f, 1.0f, 1.0f}, {2.0f, 2.0f, 2.0f}};  // (x, y, z)[m]
  float h_yaw[B] = {1.0f, 0.0f};                                 // [deg]

  float *d_src, *d_xyz, *d_yaw, *d_dst;
  const size_t in_size = sizeof(float) * N * T * D;
  const size_t xyz_size = sizeof(float) * B * 3;
  const size_t yaw_size = sizeof(float) * B;
  const size_t out_size = sizeof(float) * B * N * T * D;
  cudaMalloc(reinterpret_cast<void **>(&d_src), in_size);
  cudaMalloc(reinterpret_cast<void **>(&d_xyz), xyz_size);
  cudaMalloc(reinterpret_cast<void **>(&d_yaw), yaw_size);
  cudaMalloc(reinterpret_cast<void **>(&d_dst), out_size);
  cudaMemcpy(d_src, h_src, in_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_xyz, h_xyz, xyz_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_yaw, h_yaw, yaw_size, cudaMemcpyHostToDevice);

  dim3 blocks(N, T);
  transform_trajectory_kernel<<<blocks, 256>>>(B, N, T, D, d_src, d_xyz, d_yaw, d_dst);

  float h_dst[B][N][T][D];
  cudaMemcpy(h_dst, d_dst, out_size, cudaMemcpyDeviceToHost);

  std::cout << "Transform coords to the target centric coords..." << std::endl;
  for (int b = 0; b < B; ++b) {
    std::cout << "Batch " << b << ":\n";
    for (int n = 0; n < N; ++n) {
      std::cout << "  Agent " << n << ":\n";
      for (int t = 0; t < T; ++t) {
        std::cout << "  Time " << t << ": ";
        for (int i = 0; i < D; ++i) {
          std::cout << h_dst[b][n][t][i] << " ";
        }
        std::cout << "\n";
      }
    }
  }

  float * d_last_pos;
  cudaMalloc(reinterpret_cast<void **>(&d_last_pos), sizeof(float) * B * N * 3);
  dim3 nBlocks(B, N, T);
  extract_last_pos_kernel<<<nBlocks, 256>>>(B, N, T, D, d_dst, d_last_pos);

  float h_last_pos[B][N][3];
  cudaMemcpy(h_last_pos, d_last_pos, sizeof(float) * B * N * 3, cudaMemcpyDeviceToHost);

  std::cout << "Extract last positions..." << std::endl;
  for (int b = 0; b < B; ++b) {
    std::cout << "Batch " << b << ":\n";
    for (int n = 0; n < N; ++n) {
      std::cout << "  Agent " << n << ": ";
      for (int i = 0; i < 3; ++i) {
        std::cout << h_last_pos[b][n][i] << " ";
      }
      std::cout << "\n";
    }
  }

  cudaFree(d_src);
  cudaFree(d_xyz);
  cudaFree(d_yaw);
  cudaFree(d_dst);
  cudaFree(d_last_pos);
}
