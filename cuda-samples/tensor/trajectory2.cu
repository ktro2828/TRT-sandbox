#include <iostream>

__global__ void agentPreprocessKernel(
  const int B, const int N, const int T, const int D, const int C, const int sdc_index,
  const int * target_index, const int * object_type_index, const float * timestamps,
  const float * in_trajectory, float * out_data, bool * out_mask, float * out_last_pos)
{
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  int n = blockIdx.y * blockDim.y + threadIdx.y;
  int t = blockIdx.z * blockDim.z + threadIdx.z;

  if (b >= B || n >= N || t >= T) {
    return;
  }

  const int out_idx = (b * N * T + n * T + t) * (D - 2 + C + 2 + T + 1 + 2);

  // === transform trajectory to target centric coords ===
  const int src_trajectory_idx = (n * T + t) * D;
  const float x = in_trajectory[src_trajectory_idx];
  const float y = in_trajectory[src_trajectory_idx + 1];
  const float z = in_trajectory[src_trajectory_idx + 2];
  const float dx = in_trajectory[src_trajectory_idx + 3];
  const float dy = in_trajectory[src_trajectory_idx + 4];
  const float dz = in_trajectory[src_trajectory_idx + 5];
  const float yaw = in_trajectory[src_trajectory_idx + 6];
  const float vx = in_trajectory[src_trajectory_idx + 7];
  const float vy = in_trajectory[src_trajectory_idx + 8];
  const float ax = in_trajectory[src_trajectory_idx + 9];
  const float ay = in_trajectory[src_trajectory_idx + 10];
  const float is_valid = in_trajectory[src_trajectory_idx + 11];

  // extract targets trajectories
  const int center_idx = (target_index[b] * T + T - 1) * D;
  const float center_x = in_trajectory[center_idx];
  const float center_y = in_trajectory[center_idx + 1];
  const float center_yaw = in_trajectory[center_idx + 6];
  printf(
    "target_index: %i, center_idx: %i, (x, y)=(%f, %f), yaw=%f\n", target_index[b], center_idx,
    center_x, center_y, center_yaw);
  const float center_cos = cos(center_yaw);
  const float center_sin = sin(center_yaw);

  // do transform
  const float trans_x = center_cos * (x - center_x) - center_sin * (y + center_y) + center_x;
  const float trans_y = center_sin * (x - center_x) + center_cos * (y + center_y) + center_y;
  const float trans_z = z;
  const float trans_yaw = yaw - center_yaw;
  const float trans_vx = center_cos * vx - center_sin * vy;
  const float trans_vy = center_sin * vx + center_cos * vy;
  const float trans_ax = center_cos * ax - center_sin * ay;
  const float trans_ay = center_sin * ax + center_cos * ay;

  out_data[out_idx] = trans_x;
  out_data[out_idx + 1] = trans_y;
  out_data[out_idx + 2] = trans_z;
  out_data[out_idx + 3] = dx;
  out_data[out_idx + 4] = dy;
  out_data[out_idx + 5] = dz;

  // === onehot ===
  const int onehot_idx = out_idx + 6;
  out_data[onehot_idx + object_type_index[n]] = 1.0f;

  if (target_index[b] == n) {
    out_data[onehot_idx + C] = 1.0f;
  }

  if (sdc_index == n) {
    out_data[onehot_idx + C + 1] = 1.0f;
  }

  // === embedding ===
  const int embed_idx = onehot_idx + C + 2;
  // time embedding
  out_data[embed_idx + t] = 1.0f;
  out_data[embed_idx + T] = timestamps[t];
  // heading embedding
  out_data[embed_idx + T + 1] = sin(trans_yaw);
  out_data[embed_idx + T + 2] = cos(trans_yaw);

  const int other_idx = embed_idx + T + 3;
  out_data[other_idx] = trans_vx;
  out_data[other_idx + 1] = trans_vy;
  out_data[other_idx + 2] = trans_ax;
  out_data[other_idx + 3] = trans_ay;

  // mask
  const int mask_idx = b * N * T + n * T + t;
  out_mask[mask_idx] = is_valid == 1.0f ? true : false;

  // last pos
  if (t == T - 1) {
    const int pos_idx = (b * N + n) * 3;
    out_last_pos[pos_idx] = trans_x;
    out_last_pos[pos_idx + 1] = trans_y;
    out_last_pos[pos_idx + 2] = trans_z;
  }
}

int main()
{
  constexpr int B = 2;
  constexpr int N = 4;
  constexpr int T = 5;
  constexpr int D = 12;
  constexpr int C = 3;
  constexpr int sdc_index = 1;

  int h_target_index[B] = {0, 2};
  int h_object_type_index[N] = {0, 0, 2, 1};
  float h_timestamps[T] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  // [x, y, z, dx, dy, dz, yaw, vx, vy, ax, ay, is_valid]
  // NOTE: `is_valid` at t = T (the latest timestamp) must be 1.0f.
  float h_trajectory[N][T][D] = {
    {
      {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, M_PI / 2.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
      {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, M_PI / 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 1.0f},
      {3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, M_PI / 2.0f, 3.0f, 3.0f, 3.0f, 3.0f, 1.0f},
      {4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, M_PI / 2.0f, 4.0f, 4.0f, 4.0f, 4.0f, 1.0f},
      {5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, M_PI / 2.0f, 5.0f, 5.0f, 5.0f, 5.0f, 1.0f},
    },
    {
      {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, M_PI / 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 0.0f},
      {3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, M_PI / 2.0f, 3.0f, 3.0f, 3.0f, 3.0f, 0.0f},
      {4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, M_PI / 2.0f, 4.0f, 4.0f, 4.0f, 4.0f, 1.0f},
      {5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, M_PI / 2.0f, 5.0f, 5.0f, 5.0f, 5.0f, 0.0f},
      {6.0f, 6.0f, 6.0f, 6.0f, 6.0f, 6.0f, M_PI / 2.0f, 6.0f, 6.0f, 6.0f, 6.0f, 1.0f},
    },
    {
      {3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, M_PI / 2.0f, 3.0f, 3.0f, 3.0f, 3.0f, 1.0f},
      {4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, M_PI / 2.0f, 4.0f, 4.0f, 4.0f, 4.0f, 1.0f},
      {5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, M_PI / 2.0f, 5.0f, 5.0f, 5.0f, 5.0f, 1.0f},
      {6.0f, 6.0f, 6.0f, 6.0f, 6.0f, 6.0f, M_PI / 2.0f, 6.0f, 6.0f, 6.0f, 6.0f, 1.0f},
      {7.0f, 7.0f, 7.0f, 7.0f, 7.0f, 7.0f, M_PI / 2.0f, 7.0f, 7.0f, 7.0f, 7.0f, 1.0f},
    },
    {
      {4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, M_PI / 2.0f, 4.0f, 4.0f, 4.0f, 4.0f, 0.0f},
      {5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, M_PI / 2.0f, 5.0f, 5.0f, 5.0f, 5.0f, 1.0f},
      {6.0f, 6.0f, 6.0f, 6.0f, 6.0f, 6.0f, M_PI / 2.0f, 6.0f, 6.0f, 6.0f, 6.0f, 0.0f},
      {7.0f, 7.0f, 7.0f, 7.0f, 7.0f, 7.0f, M_PI / 2.0f, 7.0f, 7.0f, 7.0f, 7.0f, 0.0f},
      {8.0f, 8.0f, 8.0f, 8.0f, 8.0f, 8.0f, M_PI / 2.0f, 8.0f, 8.0f, 8.0f, 8.0f, 1.0f},
    },
  };

  int *d_target_index, *d_object_type_index;
  float *d_timestamps, *d_trajectory;
  // allocate input memory
  cudaMalloc(reinterpret_cast<void **>(&d_target_index), sizeof(int) * B);
  cudaMalloc(reinterpret_cast<void **>(&d_object_type_index), sizeof(int) * N);
  cudaMalloc(reinterpret_cast<void **>(&d_timestamps), sizeof(float) * T);
  cudaMalloc(reinterpret_cast<void **>(&d_trajectory), sizeof(float) * N * T * D);
  // copy input data
  cudaMemcpy(d_target_index, h_target_index, sizeof(int) * B, cudaMemcpyHostToDevice);
  cudaMemcpy(d_object_type_index, h_object_type_index, sizeof(int) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_timestamps, h_timestamps, sizeof(float) * T, cudaMemcpyHostToDevice);
  cudaMemcpy(d_trajectory, h_trajectory, sizeof(float) * N * T * D, cudaMemcpyHostToDevice);

  float *d_out_data, *d_out_last_pos;
  bool * d_out_mask;
  size_t outDataSize = sizeof(float) * B * N * T * (D - 2 + C + 2 + T + 1 + 2);
  size_t outMaskSize = sizeof(bool) * B * N * T;
  size_t outLastPosSize = sizeof(float) * B * N * 3;
  // allocate output memory
  cudaMalloc(reinterpret_cast<void **>(&d_out_data), outDataSize);
  cudaMalloc(reinterpret_cast<void **>(&d_out_mask), outMaskSize);
  cudaMalloc(reinterpret_cast<void **>(&d_out_last_pos), outLastPosSize);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // DEBUG
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  cudaEventQuery(start);
  // Preprocess
  dim3 blocks(B, N, T);
  constexpr int THREADS_PER_BLOCK = 256;
  agentPreprocessKernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
    B, N, T, D, C, sdc_index, d_target_index, d_object_type_index, d_timestamps, d_trajectory,
    d_out_data, d_out_mask, d_out_last_pos);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float elapsed_time;
  cudaEventElapsedTime(&elapsed_time, start, stop);
  printf("Processing time = %g ms.\n", elapsed_time);

  float h_out_data[B][N][T][D - 2 + C + 2 + 2 + T + 1], h_out_last_pos[B][N][3];
  bool h_out_mask[B][N][T];

  cudaMemcpy(h_out_data, d_out_data, outDataSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_out_mask, d_out_mask, outMaskSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_out_last_pos, d_out_last_pos, outLastPosSize, cudaMemcpyDeviceToHost);

  std::cout << "===== Output data =====\n";
  for (int b = 0; b < B; ++b) {
    std::cout << "Batch " << b << ":\n";
    for (int n = 0; n < N; ++n) {
      std::cout << "  Agent " << n << ":\n";
      for (int t = 0; t < T; ++t) {
        std::cout << "  Time " << t << ": ";
        for (int i = 0; i < D - 2 + C + 2 + 2 + T + 1; ++i) {
          std::cout << h_out_data[b][n][t][i] << " ";
        }
        std::cout << "\n";
      }
    }
  }

  std::cout << "===== Output mask =====\n";
  for (int b = 0; b < B; ++b) {
    std::cout << "Batch " << b << ":\n";
    for (int n = 0; n < N; ++n) {
      std::cout << "  Agent " << n << ": ";
      for (int t = 0; t < T; ++t) {
        std::cout << h_out_mask[b][n][t] << " ";
      }
      std::cout << "\n";
    }
  }

  std::cout << "===== Output last pos =====\n";
  for (int b = 0; b < B; ++b) {
    std::cout << "Batch " << b << ":\n";
    for (int n = 0; n < N; ++n) {
      std::cout << "  Agent " << n << ": ";
      for (int i = 0; i < 3; ++i) {
        std::cout << h_out_last_pos[b][n][i] << " ";
      }
      std::cout << "\n";
    }
  }
}
