#include <cmath>
#include <iostream>

/**
 * @brief
 *
 * @param N The number of all points.
 * @param D The number of dimensions of each point.
 * @param in_points Source points, in shape (A * D), ordering (x, y, z, ...).
 * @param output Output polylines, in shape (A * 5)
 */
__global__ void point2polyline_kernel(
  const int N, const int D, const float * in_points, float * output)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    int out_idx = idx * 5;
    const int prev_index = idx >= 1 ? (idx - 1) * D : 0;
    output[out_idx] = in_points[idx * D];
    output[out_idx + 1] = in_points[idx * D + 1];
    output[out_idx + 2] = in_points[prev_index];
    output[out_idx + 3] = in_points[prev_index + 1];
    output[out_idx + 4] = in_points[idx * D + D - 1];
  }
}

__global__ void find_break_index_kernel(
  const int N, const float threshold, const float * in_polyline, bool * break_index)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    const float dist = std::hypot(
      in_polyline[idx * 5] - in_polyline[idx * 5 + 2],
      in_polyline[idx * 5 + 1] - in_polyline[idx * 5 + 3]);
    break_index[idx] = dist > threshold;
  }
}

/**
 * @brief
 *
 * @param B The number of targets.
 * @param K The number of polylines for model input.
 * @param N The number of source polylines.
 * @param in_polylines (N, )
 * @param output (B * K * )
 */
__global__ void generate_polyline_kernel(
  const int B, const int K, const int N, const float * in_polylines, float * output)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // if N > K:
  //    polylines_centers = in_polylines[..., 0:2].sum(dim=1) / mask.sum(dim=1)
  //    center_offsets[B * 2] = (center_offset_x, center_offset_y).repeat(B)
  //    Rotate(center_offsets, center_yaw)
  //    map_centers = center_xyz[.., 0:2] + center_offsets
  //    disntaces = Distance(map_centers - polylines_centers)
  //    topk_idx = TopK(disntace, k=K);;
  // else:
}

/**
 * @brief
 *
 * @param B The number of targets.
 * @param K The number of polylines.
 * @param P The number of points for each polyline.
 * @param D The number of dimensions for each point.
 * @param polylines Source polylines, in shape (K * P * D).
 * @param center_xyz
 * @param center_yaw
 * @param output
 */
__global__ void transform_polyline_kernel(
  const int B, const int K, const int P, const int D, const float * polylines,
  const float * center_xyz, const float * center_yaw, float * output)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < K * P) {
    const float x = polylines[idx * D];
    const float y = polylines[idx * D + 1];
    const float z = polylines[idx * D + 2];
    const float dx = polylines[idx * D + 3];
    const float dy = polylines[idx * D + 4];
    const float dz = polylines[idx * D + 5];
    const float type_id = polylines[idx * D + 6];

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
      const float trans_dx = cos_val * dx - sin_val * dy;
      const float trans_dy = sin_val * dx + cos_val * dy;
      const float trans_dz = dz;

      const int trans_idx = (b * K * P + idx) * D;
      output[trans_idx] = trans_x;
      output[trans_idx + 1] = trans_y;
      output[trans_idx + 2] = trans_z;
      output[trans_idx + 3] = trans_dx;
      output[trans_idx + 4] = trans_dy;
      output[trans_idx + 5] = trans_dz;
      output[trans_idx + 6] = type_id;
    }
  }
}

int main()
{
  constexpr int N = 5;
  constexpr int D = 7;

  float h_points[N][D] = {
    {1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f, 1.0f},    {3.0f, 4.0f, 3.0f, 4.0f, 3.0f, 4.0f, 2.0f},
    {5.0f, 6.0f, 5.0f, 6.0f, 5.0f, 6.0f, 3.0f},    {7.0f, 8.0f, 7.0f, 8.0f, 7.0f, 8.0f, 4.0f},
    {9.0f, 10.0f, 9.0f, 10.0f, 9.0f, 10.0f, 5.0f},
  };

  float *d_points, *d_polyline;
  const size_t PtSize = sizeof(float) * N * D;
  const size_t PolySize = sizeof(float) * N * D;
  cudaMalloc(reinterpret_cast<void **>(&d_points), PtSize);
  cudaMalloc(reinterpret_cast<void **>(&d_polyline), PolySize);
  cudaMemcpy(d_points, h_points, PtSize, cudaMemcpyHostToDevice);

  dim3 blocks(8);
  point2polyline_kernel<<<blocks, 256>>>(N, D, d_points, d_polyline);

  cudaDeviceSynchronize();

  constexpr float distance_threshold = 1.0;
  bool * d_break_index;
  cudaMalloc(reinterpret_cast<void **>(&d_break_index), sizeof(bool) * N);
  find_break_index_kernel<<<blocks, 256>>>(N, distance_threshold, d_polyline, d_break_index);

  float h_polyline[N][5];
  cudaMemcpy(h_polyline, d_polyline, PolySize, cudaMemcpyDeviceToHost);

  bool h_break_index[N];
  cudaMemcpy(h_break_index, d_break_index, sizeof(bool) * N, cudaMemcpyDeviceToHost);

  std::cout << "Polylines" << std::endl;
  for (int a = 0; a < N; ++a) {
    std::cout << "Batch " << a << ":\n";
    for (int i = 0; i < 5; ++i) {
      std::cout << h_polyline[a][i] << " ";
    }
    std::cout << "\n";
  }

  std::cout << "Break index" << std::endl;
  for (int n = 0; n < N; ++n) {
    std::cout << h_break_index[n] << " ";
  }

  cudaFree(d_points);
  cudaFree(d_polyline);
  cudaFree(d_break_index);
}