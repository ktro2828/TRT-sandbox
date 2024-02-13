#include <stdio.h>

#include <iostream>

__device__ void transform_trajectory(
  const int B, const int N, const int T, const int D, const int * target_index,
  const float * in_trajectory, float * output)
{
  // output [B * N * T * D]
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
    const int center_idx = (target_index[b] * T + T - 1) * D;
    const float center_x = in_trajectory[center_idx];
    const float center_y = in_trajectory[center_idx + 1];
    const float center_yaw = in_trajectory[center_idx + 6];
    const float cos_val = std::cos(center_yaw);
    const float sin_val = std::sin(center_yaw);

    // transform
    const float trans_x = cos_val * x - sin_val * y - center_x;
    const float trans_y = sin_val * x + cos_val * y - center_y;
    const float trans_vx = cos_val * vx - sin_val * vy;
    const float trans_vy = sin_val * vx + cos_val * vy;
    const float trans_ax = cos_val * ax - sin_val * ay;
    const float trans_ay = sin_val * ax + cos_val * ay;

    const int trans_idx = (b * N * T + n * T + t) * D;
    output[trans_idx] = trans_x;
    output[trans_idx + 1] = trans_y;
    output[trans_idx + 2] = z;
    output[trans_idx + 3] = dx;
    output[trans_idx + 4] = dy;
    output[trans_idx + 5] = dz;
    output[trans_idx + 6] = yaw - center_yaw;
    output[trans_idx + 7] = trans_vx;
    output[trans_idx + 8] = trans_vy;
    output[trans_idx + 9] = trans_ax;
    output[trans_idx + 10] = trans_ay;
    output[trans_idx + 11] = is_valid;
  }
}

__device__ void generate_onehot_mask(
  const int B, const int N, const int T, const int C, const int sdc_index, const int * target_index,
  const int * object_type_index, float * onehot)
{
  // output [B * N * T * (C + 2)]
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  int n = blockIdx.y * blockDim.y + threadIdx.y;
  int t = blockIdx.z * blockDim.z + threadIdx.z;

  if (b < B && n < N && t < T) {
    int idx = b * N * T + n * T + t;

    // set default value to 0.0
    onehot[idx] = 0.0f;
    onehot[idx * (C + 2) + object_type_index[n]] = 1.0f;

    if (target_index[b] == n) {
      onehot[idx * (C + 2) + C] = 1.0f;
    }

    if (sdc_index == n) {
      onehot[idx * (C + 2) + C + 1] = 1.0f;
    }
  }
}

__device__ void generate_embedding(
  const int B, const int N, const int T, const int D, const float * timestamps,
  const float * trajectory, float * time_embed, float * heading_embed)
{
  // time_embed [B * N * T * (T + 1)]
  // heading_embed [B * N * T * 2]
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  int n = blockIdx.y * blockDim.y + threadIdx.y;
  int t = blockIdx.z * blockDim.z + threadIdx.z;

  if (b < B && n < N && t < T) {
    const int idx = b * N * T + n * T + t;
    time_embed[idx] = 0.0f;
    time_embed[idx * (T + 1) + t] = 1.0f;
    time_embed[idx * (T + 1) + T] = timestamps[t];

    const float yaw = trajectory[idx * D + 6];
    heading_embed[idx * 2] = std::sin(yaw);
    heading_embed[idx * 2 + 1] = std::cos(yaw);
  }
}

__device__ void extract_last_pos(
  const int B, const int N, const int T, const int D, const float * trajectory, float * output)
{
  // output [B * N * 3]
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  int n = blockIdx.y * blockDim.y + threadIdx.y;
  int t = blockIdx.z * blockDim.z + threadIdx.z;

  if (b < B && t == T - 1) {
    const int idx = b * N * T + n * T + t;
    const int out_idx = b * N + n;
    output[out_idx] = 0.0f;
    output[out_idx * 3] = trajectory[idx * D];
    output[out_idx * 3 + 1] = trajectory[idx * D + 1];
    output[out_idx * 3 + 2] = trajectory[idx * D + 2];
  }
}

__device__ void concatenate_agent_data(
  const int B, const int N, const int T, const int D, const int C, const float * trajectory,
  const float * onehot, const float * time_embed, const float * heading_embed, float * out_data,
  bool * out_mask)
{
  const int dim = (D - 2 + C + 2 + 2 + T + 1);
  // out_data [B * N * T * (D - 2 + (C + 2) + 2 + (T + 1)]
  // out_mask [B * N * T]
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  int n = blockIdx.y * blockDim.y + threadIdx.y;
  int t = blockIdx.z * blockDim.z + threadIdx.z;

  if (b < B && n < N && t < T) {
    const int idx = b * N * T + n * T + t;
    const int out_idx = idx * dim;
    out_data[out_idx] = trajectory[idx * D];
    out_data[out_idx + 1] = trajectory[idx * D + 1];
    out_data[out_idx + 2] = trajectory[idx * D + 2];
    out_data[out_idx + 3] = trajectory[idx * D + 3];
    out_data[out_idx + 4] = trajectory[idx * D + 4];
    out_data[out_idx + 5] = trajectory[idx * D + 5];
    for (int c_idx = 0; c_idx < C + 2; ++c_idx) {
      out_data[out_idx + 6 + c_idx] = onehot[idx * (C + 2) + c_idx];
    }
    for (int t_idx = 0; t_idx < T + 1; ++t_idx) {
      out_data[out_idx + C + 2 + t_idx] = time_embed[idx * (T + 1) + t_idx];
    }
    out_data[out_idx + C + 2 + T + 1] = heading_embed[idx * 2];
    out_data[out_idx + C + 2 + T + 1 + 1] = heading_embed[idx * 2 + 1];
    out_data[out_idx + C + 2 + T + 1 + 2] = trajectory[idx * D + 7];
    out_data[out_idx + C + 2 + T + 1 + 3] = trajectory[idx * D + 8];
    out_data[out_idx + C + 2 + T + 1 + 4] = trajectory[idx * D + 9];
    out_data[out_idx + C + 2 + T + 1 + 5] = trajectory[idx * D + 10];
    out_mask[out_idx + dim - 1] = static_cast<bool>(trajectory[idx * D + D - 1]);
  }
}

__global__ void agentPreprocessKernel(
  const int B, const int N, const int T, const int D, const int C, const int sdc_index,
  const int * target_index, const int * object_type_index, const float * timestamps,
  const float * in_trajectory, float * out_data, bool * out_mask, float * out_last_pos)
{
  extern __shared__ float tmp[];

  const int dstTrajectorySize = B * N * T * D;
  const int onehotSize = B * N * T * (C + 2);
  const int timeEmbedSize = B * N * T * (T + 1);
  // const int headingEmbedSize = B * N * T * 2;

  float * dst_trajectory = tmp;
  float * onehot = (float *)&dst_trajectory[dstTrajectorySize];

  transform_trajectory(B, N, T, D, target_index, in_trajectory, dst_trajectory);
  generate_onehot_mask(B, N, T, C, sdc_index, target_index, object_type_index, onehot);

  __syncthreads();

  float * time_embed = (float *)&onehot[onehotSize];
  float * heading_embed = (float *)&time_embed[timeEmbedSize];

  generate_embedding(B, N, T, D, timestamps, dst_trajectory, time_embed, heading_embed);
  extract_last_pos(B, N, T, D, dst_trajectory, out_last_pos);

  __syncthreads();

  concatenate_agent_data(
    B, N, T, D, C, dst_trajectory, onehot, time_embed, heading_embed, out_data, out_mask);
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
  size_t outDataSize = sizeof(float) * B * N * T * (D - 2 + C + 2 + 2 + T + 1);
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
  size_t sharedMemSize = sizeof(float) * B * N * T * (D + C + 2 + T + 1 + 2);
  agentPreprocessKernel<<<blocks, THREADS_PER_BLOCK, sharedMemSize, stream>>>(
    B, N, T, D, C, sdc_index, d_target_index, d_object_type_index, d_timestamps, d_trajectory,
    d_out_data, d_out_mask, d_out_last_pos);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float elapsed_time;
  cudaEventElapsedTime(&elapsed_time, start, stop);
  printf("Processing time = %g ms.\n", elapsed_time);

  float h_out_data[B][N][T][D - 2 + C + 2 + 2 + T + 1], h_out_last_pos[B][N][3];
  float h_out_mask[B][N][T];

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

// obj_data=tensor([[[[-4.0000,  4.0000, -4.0000,  1.0000,  1.0000,  1.0000,  1.0000,
//             0.0000,  0.0000,  1.0000,  0.0000,  1.0000,  0.0000,  0.0000,
//             0.0000,  0.0000,  1.0000,  0.0000,  1.0000,  1.0000, -1.0000,
//             1.0000,  1.0000],
//           [-3.0000,  3.0000, -3.0000,  2.0000,  2.0000,  2.0000,  1.0000,
//             0.0000,  0.0000,  1.0000,  0.0000,  0.0000,  1.0000,  0.0000,
//             0.0000,  0.0000,  2.0000,  0.0000,  1.0000,  2.0000, -2.0000,
//             2.0000,  2.0000],
//           [-2.0000,  2.0000, -2.0000,  3.0000,  3.0000,  3.0000,  1.0000,
//             0.0000,  0.0000,  1.0000,  0.0000,  0.0000,  0.0000,  1.0000,
//             0.0000,  0.0000,  3.0000,  0.0000,  1.0000,  3.0000, -3.0000,
//             3.0000,  3.0000],
//           [-1.0000,  1.0000, -1.0000,  4.0000,  4.0000,  4.0000,  1.0000,
//             0.0000,  0.0000,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000,
//             1.0000,  0.0000,  4.0000,  0.0000,  1.0000,  4.0000, -4.0000,
//             4.0000,  4.0000],
//           [ 0.0000,  0.0000,  0.0000,  5.0000,  5.0000,  5.0000,  1.0000,
//             0.0000,  0.0000,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000,
//             0.0000,  1.0000,  5.0000,  0.0000,  1.0000,  5.0000, -5.0000,
//             5.0000,  5.0000]],

//          [[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
//             0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
//             0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
//             0.0000,  0.0000],
//           [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
//             0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
//             0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
//             0.0000,  0.0000],
//           [-1.0000,  1.0000, -1.0000,  4.0000,  4.0000,  4.0000,  1.0000,
//             0.0000,  0.0000,  0.0000,  1.0000,  0.0000,  0.0000,  1.0000,
//             0.0000,  0.0000,  3.0000,  0.0000,  1.0000,  4.0000, -4.0000,
//             4.0000,  4.0000],
//           [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
//             0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
//             0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
//             0.0000,  0.0000],
//           [ 1.0000, -1.0000,  1.0000,  6.0000,  6.0000,  6.0000,  1.0000,
//             0.0000,  0.0000,  0.0000,  1.0000,  0.0000,  0.0000,  0.0000,
//             0.0000,  1.0000,  5.0000,  0.0000,  1.0000,  6.0000, -6.0000,
//             6.0000,  6.0000]],

//          [[-2.0000,  2.0000, -2.0000,  3.0000,  3.0000,  3.0000,  0.0000,
//             0.0000,  1.0000,  0.0000,  0.0000,  1.0000,  0.0000,  0.0000,
//             0.0000,  0.0000,  1.0000,  0.0000,  1.0000,  3.0000, -3.0000,
//             3.0000,  3.0000],
//           [-1.0000,  1.0000, -1.0000,  4.0000,  4.0000,  4.0000,  0.0000,
//             0.0000,  1.0000,  0.0000,  0.0000,  0.0000,  1.0000,  0.0000,
//             0.0000,  0.0000,  2.0000,  0.0000,  1.0000,  4.0000, -4.0000,
//             4.0000,  4.0000],
//           [ 0.0000,  0.0000,  0.0000,  5.0000,  5.0000,  5.0000,  0.0000,
//             0.0000,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000,  1.0000,
//             0.0000,  0.0000,  3.0000,  0.0000,  1.0000,  5.0000, -5.0000,
//             5.0000,  5.0000],
//           [ 1.0000, -1.0000,  1.0000,  6.0000,  6.0000,  6.0000,  0.0000,
//             0.0000,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
//             1.0000,  0.0000,  4.0000,  0.0000,  1.0000,  6.0000, -6.0000,
//             6.0000,  6.0000],
//           [ 2.0000, -2.0000,  2.0000,  7.0000,  7.0000,  7.0000,  0.0000,
//             0.0000,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
//             0.0000,  1.0000,  5.0000,  0.0000,  1.0000,  7.0000, -7.0000,
//             7.0000,  7.0000]],

//          [[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
//             0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
//             0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
//             0.0000,  0.0000],
//           [ 0.0000,  0.0000,  0.0000,  5.0000,  5.0000,  5.0000,  0.0000,
//             0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  1.0000,  0.0000,
//             0.0000,  0.0000,  2.0000,  0.0000,  1.0000,  5.0000, -5.0000,
//             5.0000,  5.0000],
//           [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
//             0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
//             0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
//             0.0000,  0.0000],
//           [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
//             0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
//             0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
//             0.0000,  0.0000],
//           [ 3.0000, -3.0000,  3.0000,  8.0000,  8.0000,  8.0000,  0.0000,
//             0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
//             0.0000,  1.0000,  5.0000,  0.0000,  1.0000,  8.0000, -8.0000,
//             8.0000,  8.0000]]],

//         [[[-6.0000,  6.0000, -6.0000,  1.0000,  1.0000,  1.0000,  1.0000,
//             0.0000,  0.0000,  0.0000,  0.0000,  1.0000,  0.0000,  0.0000,
//             0.0000,  0.0000,  1.0000,  0.0000,  1.0000,  1.0000, -1.0000,
//             1.0000,  1.0000],
//           [-5.0000,  5.0000, -5.0000,  2.0000,  2.0000,  2.0000,  1.0000,
//             0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  1.0000,  0.0000,
//             0.0000,  0.0000,  2.0000,  0.0000,  1.0000,  2.0000, -2.0000,
//             2.0000,  2.0000],
//           [-4.0000,  4.0000, -4.0000,  3.0000,  3.0000,  3.0000,  1.0000,
//             0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  1.0000,
//             0.0000,  0.0000,  3.0000,  0.0000,  1.0000,  3.0000, -3.0000,
//             3.0000,  3.0000],
//           [-3.0000,  3.0000, -3.0000,  4.0000,  4.0000,  4.0000,  1.0000,
//             0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
//             1.0000,  0.0000,  4.0000,  0.0000,  1.0000,  4.0000, -4.0000,
//             4.0000,  4.0000],
//           [-2.0000,  2.0000, -2.0000,  5.0000,  5.0000,  5.0000,  1.0000,
//             0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
//             0.0000,  1.0000,  5.0000,  0.0000,  1.0000,  5.0000, -5.0000,
//             5.0000,  5.0000]],

//          [[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
//             0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
//             0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
//             0.0000,  0.0000],
//           [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
//             0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
//             0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
//             0.0000,  0.0000],
//           [-3.0000,  3.0000, -3.0000,  4.0000,  4.0000,  4.0000,  1.0000,
//             0.0000,  0.0000,  0.0000,  1.0000,  0.0000,  0.0000,  1.0000,
//             0.0000,  0.0000,  3.0000,  0.0000,  1.0000,  4.0000, -4.0000,
//             4.0000,  4.0000],
//           [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
//             0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
//             0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
//             0.0000,  0.0000],
//           [-1.0000,  1.0000, -1.0000,  6.0000,  6.0000,  6.0000,  1.0000,
//             0.0000,  0.0000,  0.0000,  1.0000,  0.0000,  0.0000,  0.0000,
//             0.0000,  1.0000,  5.0000,  0.0000,  1.0000,  6.0000, -6.0000,
//             6.0000,  6.0000]],

//          [[-4.0000,  4.0000, -4.0000,  3.0000,  3.0000,  3.0000,  0.0000,
//             0.0000,  1.0000,  1.0000,  0.0000,  1.0000,  0.0000,  0.0000,
//             0.0000,  0.0000,  1.0000,  0.0000,  1.0000,  3.0000, -3.0000,
//             3.0000,  3.0000],
//           [-3.0000,  3.0000, -3.0000,  4.0000,  4.0000,  4.0000,  0.0000,
//             0.0000,  1.0000,  1.0000,  0.0000,  0.0000,  1.0000,  0.0000,
//             0.0000,  0.0000,  2.0000,  0.0000,  1.0000,  4.0000, -4.0000,
//             4.0000,  4.0000],
//           [-2.0000,  2.0000, -2.0000,  5.0000,  5.0000,  5.0000,  0.0000,
//             0.0000,  1.0000,  1.0000,  0.0000,  0.0000,  0.0000,  1.0000,
//             0.0000,  0.0000,  3.0000,  0.0000,  1.0000,  5.0000, -5.0000,
//             5.0000,  5.0000],
//           [-1.0000,  1.0000, -1.0000,  6.0000,  6.0000,  6.0000,  0.0000,
//             0.0000,  1.0000,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000,
//             1.0000,  0.0000,  4.0000,  0.0000,  1.0000,  6.0000, -6.0000,
//             6.0000,  6.0000],
//           [ 0.0000,  0.0000,  0.0000,  7.0000,  7.0000,  7.0000,  0.0000,
//             0.0000,  1.0000,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000,
//             0.0000,  1.0000,  5.0000,  0.0000,  1.0000,  7.0000, -7.0000,
//             7.0000,  7.0000]],

//          [[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
//             0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
//             0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
//             0.0000,  0.0000],
//           [-2.0000,  2.0000, -2.0000,  5.0000,  5.0000,  5.0000,  0.0000,
//             0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  1.0000,  0.0000,
//             0.0000,  0.0000,  2.0000,  0.0000,  1.0000,  5.0000, -5.0000,
//             5.0000,  5.0000],
//           [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
//             0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
//             0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
//             0.0000,  0.0000],
//           [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
//             0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
//             0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
//             0.0000,  0.0000],
//           [ 1.0000, -1.0000,  1.0000,  8.0000,  8.0000,  8.0000,  0.0000,
//             0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
//             0.0000,  1.0000,  5.0000,  0.0000,  1.0000,  8.0000, -8.0000,
//             8.0000,  8.0000]]]]), obj_data.shape=torch.Size([2, 4, 5, 23])

// obj_mask=tensor([[[1., 1., 1., 1., 1.],
//          [0., 0., 1., 0., 1.],
//          [1., 1., 1., 1., 1.],
//          [0., 1., 0., 0., 1.]],

//         [[1., 1., 1., 1., 1.],
//          [0., 0., 1., 0., 1.],
//          [1., 1., 1., 1., 1.],
//          [0., 1., 0., 0., 1.]]]), obj_mask.shape=torch.Size([2, 4, 5])

// obj_last_pos=array([[[ 0.        ,  0.        ,  0.        ],
//         [ 0.99999994, -1.        ,  1.        ],
//         [ 1.9999999 , -2.        ,  2.        ],
//         [ 2.9999998 , -3.0000002 ,  3.        ]],

//        [[-1.9999999 ,  2.        , -2.        ],
//         [-0.99999994,  1.        , -1.        ],
//         [ 0.        ,  0.        ,  0.        ],
//         [ 0.99999994, -1.        ,  1.        ]]], dtype=float32), obj_last_pos.shape=(2, 4, 3)
