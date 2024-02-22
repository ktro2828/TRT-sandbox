#include <iostream>

__global__ void transformPolylineKernel(
  const int K, const int P, const int PointDim, const float * in_polyline, const int B,
  const int AgentDim, const float * target_state, float * out_polyline, bool * out_polyline_mask)
{
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;
  int p = blockIdx.z * blockDim.z + threadIdx.z;

  if (b >= B || k >= K || p >= P) {
    return;
  }

  const int src_polyline_idx = (k * P + p) * PointDim;

  const int out_mask_idx = b * K * P + k * P + p;
  bool is_valid = false;
  for (size_t i = 0; i < 6; ++i) {
    is_valid += in_polyline[src_polyline_idx + i] != 0.0f;
  }
  out_polyline_mask[out_mask_idx] = is_valid;

  if (is_valid) {
    const float x = in_polyline[src_polyline_idx];
    const float y = in_polyline[src_polyline_idx + 1];
    const float z = in_polyline[src_polyline_idx + 2];
    const float dx = in_polyline[src_polyline_idx + 3];
    const float dy = in_polyline[src_polyline_idx + 4];
    const float dz = in_polyline[src_polyline_idx + 5];
    const float type_id = in_polyline[src_polyline_idx + 6];

    const int center_idx = b * AgentDim;
    const float center_x = target_state[center_idx];
    const float center_y = target_state[center_idx + 1];
    const float center_z = target_state[center_idx + 2];
    const float center_yaw = target_state[center_idx + 6];
    const float center_cos = cos(center_yaw);
    const float center_sin = sin(center_yaw);

    // do transform
    const float trans_x = center_cos * (x - center_x) - center_sin * (y - center_y);
    const float trans_y = center_sin * (x - center_x) + center_cos * (y - center_y);
    const float trans_z = z - center_z;
    const float trans_dx = center_cos * dx - center_sin * dy;
    const float trans_dy = center_sin * dx + center_cos * dy;
    const float trans_dz = dz;

    const int out_idx = (b * K * P + k * P + p) * (PointDim + 2);
    out_polyline[out_idx] = trans_x;
    out_polyline[out_idx + 1] = trans_y;
    out_polyline[out_idx + 2] = trans_z;
    out_polyline[out_idx + 3] = trans_dx;
    out_polyline[out_idx + 4] = trans_dy;
    out_polyline[out_idx + 5] = trans_dz;
    out_polyline[out_idx + 6] = type_id;
  }
}

__global__ void setPreviousPositionKernel(
  const int B, const int K, const int P, const int D, const bool * mask, float * polyline)
{
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;
  int p = blockIdx.z * blockDim.z + threadIdx.z;

  if (b >= B || k >= K || p >= P) {
    return;
  }

  const int cur_idx = (b * K * P + k * P + p) * D;
  const int pre_idx = k == 0 ? cur_idx : (b * K * P + (k - 1) * P + p) * D;

  polyline[cur_idx + D - 2] = polyline[pre_idx];
  polyline[cur_idx + D - 1] = polyline[pre_idx + 1];

  const int mask_idx = b * K * P + k * P + p;
  if (!mask[mask_idx]) {
    for (int d = 0; d < D; ++d) {
      polyline[cur_idx + d] = 0.0f;
    }
  }
}

int main()
{
  constexpr int K = 5;
  constexpr int P = 5;
  constexpr int PointDim = 7;

  float h_in_polyline[K][P][PointDim] = {
    {
      {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
      {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f},
      {3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 1.0f},
      {4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 1.0f},
      {5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 1.0f},
    },
    {
      {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
      {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f},
      {3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 1.0f},
      {4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 1.0f},
      {5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 1.0f},
    },
    {
      {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
      {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f},
      {3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 1.0f},
      {4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 1.0f},
      {5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 1.0f},
    }};

  constexpr int B = 2;
  constexpr int AgentDim = 12;
  float h_target_state[B][AgentDim] = {
    {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
    {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 1.0f, 1.0f},
  };

  const size_t inPolylineSize = sizeof(float) * K * P * PointDim;
  const size_t targetStateSize = sizeof(float) * B * AgentDim;
  const size_t outPolylineSize = sizeof(float) * B * K * P * (PointDim + 2);
  const size_t outMaskSize = sizeof(bool) * K * P;

  float *d_in_polyline, *d_target_state, *d_out_polyline;
  bool * d_out_polyline_mask;

  cudaMalloc(reinterpret_cast<void **>(&d_in_polyline), inPolylineSize);
  cudaMalloc(reinterpret_cast<void **>(&d_target_state), targetStateSize);
  cudaMalloc(reinterpret_cast<void **>(&d_out_polyline), outPolylineSize);
  cudaMalloc(reinterpret_cast<void **>(&d_out_polyline_mask), outMaskSize);

  cudaMemcpy(d_in_polyline, h_in_polyline, inPolylineSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_target_state, h_target_state, targetStateSize, cudaMemcpyHostToDevice);

  dim3 blocks(B, K, P);
  constexpr int THREADS_PER_BLOCK = 256;
  transformPolylineKernel<<<blocks, THREADS_PER_BLOCK>>>(
    K, P, PointDim, d_in_polyline, B, AgentDim, d_target_state, d_out_polyline,
    d_out_polyline_mask);

  setPreviousPositionKernel<<<blocks, THREADS_PER_BLOCK>>>(
    B, K, P, PointDim, d_out_polyline_mask, d_out_polyline);

  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << cudaGetErrorString(err) << std::endl;
  }

  float h_out_polyline[B][K][P][PointDim + 2];
  bool h_out_polyline_mask[B][K][P];
  cudaMemcpy(h_out_polyline, d_out_polyline, outPolylineSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_out_polyline_mask, d_out_polyline_mask, outMaskSize, cudaMemcpyDeviceToHost);

  std::cout << "=== Out polyline data ===\n";
  for (int b = 0; b < B; ++b) {
    std::cout << "Batch " << b << ":\n";
    for (int k = 0; k < K; ++k) {
      std::cout << "  Polyline " << k << ":\n";
      for (int p = 0; p < P; ++p) {
        std::cout << "  Point " << p << ": ";
        for (int i = 0; i < PointDim + 2; ++i) {
          std::cout << h_out_polyline[b][k][p][i] << " ";
        }
        std::cout << "\n";
      }
    }
  }

  std::cout << "=== Out polyline mask ===\n";
  for (int b = 0; b < B; ++b) {
    std::cout << "Batch " << b << ":\n";
    for (int k = 0; k < K; ++k) {
      std::cout << "  Polyline " << k << ": ";
      for (int p = 0; p < P; ++p) {
        std::cout << h_out_polyline_mask[b][k][p] << " ";
      }
      std::cout << "\n";
    }
  }
}