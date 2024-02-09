#include <iostream>

/**
 * @brief
 *
 * @param N The number of polylines.
 * @param P The number of points of each polyline.
 * @param D The number of dimensions of each point.
 * @param in_polyline The polylines, in shape (N*P*D), ordering (x, y, z, ...).
 * @param out_polyline Output polylines, in shape (N * P * 7)
 * @param out_mask Output masks, in shape (N * P)
 */
__global__ void generate_bach_polyline_kernel(
  const int N, const int P, const int D, const float * in_polyline, float * out_polyline,
  float * out_mask)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N * P) {
  }
}

__global__ void generate_target_centric_polyline_kernel()
{
}