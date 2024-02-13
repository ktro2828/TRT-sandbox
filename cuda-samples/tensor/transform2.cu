#include <iostream>

/**
 * @brief Transform the trajectory coords to the coords system centered around the target object.
 *
 * @param B
 * @param N
 * @param T
 * @param D
 * @param in_trajectory
 * @param target_index
 * @param output
 */
__global__ void transform_trajectory_kernel(
  const int B, const int N, const int T, const int D, const float * in_trajectory,
  const int * target_index, float * output)
{
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  int n = blockIdx.y * blockDim.y + threadIdx.y;
  int t = blockIdx.z * blockDim.z + threadIdx.z;
  if (b < B && n < N && t < T) {
    int src_idx = n * T + t;
    const float x = in_trajectory[src_idx * D];
    const float y = in_trajectory[src_idx * D + 1];
    const float z = in_trajectory[src_idx * D + 2];
    const float dx = in_trajectory[src_idx * D + 3];
    const float dy = in_trajectory[src_idx * D + 4];
    const float dz = in_trajectory[src_idx * D + 5];
    const float yaw = in_trajectory[src_idx * D + 6];
    const float vx = in_trajectory[src_idx * D + 7];
    const float vy = in_trajectory[src_idx * D + 8];
    const float ax = in_trajectory[src_idx * D + 9];
    const float ay = in_trajectory[src_idx * D + 10];
    const float is_valid = in_trajectory[src_idx * D + 11];

    // transform for each target
    const int tgt_idx = (target_index[b] * T + T - 1) * D;

    const float tgt_x = in_trajectory[tgt_idx];
    const float tgt_y = in_trajectory[tgt_idx + 1];
    const float tgt_z = in_trajectory[tgt_idx + 2];
    const float tgt_yaw = in_trajectory[tgt_idx + 6];
    const float cos_val = cos(tgt_yaw);
    const float sin_val = sin(tgt_yaw);

    // transform
    const float trans_x = cos_val * (x - tgt_x) - sin_val * (y - tgt_y);
    const float trans_y = sin_val * (x - tgt_x) + cos_val * (y - tgt_y);
    const float trans_z = z - tgt_z;
    const float trans_yaw = yaw - tgt_yaw;
    const float trans_vx = cos_val * vx - sin_val * vy;
    const float trans_vy = sin_val * vx + cos_val * vy;
    const float trans_ax = cos_val * ax - sin_val * ay;
    const float trans_ay = sin_val * ax + cos_val * ay;

    const int trans_idx = (b * N * T + n * T + t) * D;
    output[trans_idx] = trans_x;
    output[trans_idx + 1] = trans_y;
    output[trans_idx + 2] = trans_z;
    output[trans_idx + 3] = dx;
    output[trans_idx + 4] = dy;
    output[trans_idx + 5] = dz;
    output[trans_idx + 6] = trans_yaw;
    output[trans_idx + 7] = trans_vx;
    output[trans_idx + 8] = trans_vy;
    output[trans_idx + 9] = trans_ax;
    output[trans_idx + 10] = trans_ay;
    output[trans_idx + 11] = is_valid;
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
  constexpr int N = 4;   // The number of agents
  constexpr int T = 5;   // The number of timestamps
  constexpr int D = 12;  // The number of state dimensions
  float h_src[N][T][D] = {
    {
      {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, M_PI / 2, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
      {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, M_PI / 2, 2.0f, 2.0f, 2.0f, 2.0f, 1.0f},
      {3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, M_PI / 2, 3.0f, 3.0f, 3.0f, 3.0f, 1.0f},
      {4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, M_PI / 2, 4.0f, 4.0f, 4.0f, 4.0f, 1.0f},
      {5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, M_PI / 2, 5.0f, 5.0f, 5.0f, 5.0f, 1.0f},
    },
    {
      {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, M_PI / 2, 2.0f, 2.0f, 2.0f, 2.0f, 0.0f},
      {3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, M_PI / 2, 3.0f, 3.0f, 3.0f, 3.0f, 0.0f},
      {4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, M_PI / 2, 4.0f, 4.0f, 4.0f, 4.0f, 1.0f},
      {5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, M_PI / 2, 5.0f, 5.0f, 5.0f, 5.0f, 0.0f},
      {6.0f, 6.0f, 6.0f, 6.0f, 6.0f, 6.0f, M_PI / 2, 6.0f, 6.0f, 6.0f, 6.0f, 1.0f},
    },
    {
      {3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, M_PI / 2, 3.0f, 3.0f, 3.0f, 3.0f, 1.0f},
      {4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, M_PI / 2, 4.0f, 4.0f, 4.0f, 4.0f, 1.0f},
      {5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, M_PI / 2, 5.0f, 5.0f, 5.0f, 5.0f, 1.0f},
      {6.0f, 6.0f, 6.0f, 6.0f, 6.0f, 6.0f, M_PI / 2, 6.0f, 6.0f, 6.0f, 6.0f, 1.0f},
      {7.0f, 7.0f, 7.0f, 7.0f, 7.0f, 7.0f, M_PI / 2, 7.0f, 7.0f, 7.0f, 7.0f, 1.0f},
    },
    {
      {4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, M_PI / 2, 4.0f, 4.0f, 4.0f, 4.0f, 0.0f},
      {5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, M_PI / 2, 5.0f, 5.0f, 5.0f, 5.0f, 1.0f},
      {6.0f, 6.0f, 6.0f, 6.0f, 6.0f, 6.0f, M_PI / 2, 6.0f, 6.0f, 6.0f, 6.0f, 0.0f},
      {7.0f, 7.0f, 7.0f, 7.0f, 7.0f, 7.0f, M_PI / 2, 7.0f, 7.0f, 7.0f, 7.0f, 0.0f},
      {8.0f, 8.0f, 8.0f, 8.0f, 8.0f, 8.0f, M_PI / 2, 8.0f, 8.0f, 8.0f, 8.0f, 1.0f},
    },
  };
  int h_target_index[B] = {0, 2};

  float *d_src, *d_dst;
  int * d_target_index;
  const size_t in_size = sizeof(float) * N * T * D;
  const size_t out_size = sizeof(float) * B * N * T * D;
  cudaMalloc(reinterpret_cast<void **>(&d_src), in_size);
  cudaMalloc(reinterpret_cast<void **>(&d_target_index), sizeof(int) * B);
  cudaMalloc(reinterpret_cast<void **>(&d_dst), out_size);
  cudaMemcpy(d_src, h_src, in_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_target_index, h_target_index, sizeof(int) * B, cudaMemcpyHostToDevice);

  dim3 blocks(B, N, T);
  transform_trajectory_kernel<<<blocks, 1023>>>(B, N, T, D, d_src, d_target_index, d_dst);

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
  cudaFree(d_target_index);
  cudaFree(d_dst);
  cudaFree(d_last_pos);
}
