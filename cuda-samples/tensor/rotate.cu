#include <cmath>
#include <iostream>

/**
 * @brief Rotates points in shape (B * N * D).
 *
 * @param B The batch size.
 * @param N The number of other points.
 * @param D The number of dimensions of points.
 * @param src Source points, in shape (B * N * D) ordering (x, y, z, ...).
 * @param angles Source angles [deg], in shape (B).
 * @param dst Output points, in shape (B * N * D)
 * @return __global__
 */
__global__ void rotate_points_kernel(
  const int B, const int N, const int D, const float * src, const float * angles, float * dst)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (B * N <= idx) {
    return;
  }

  const int b = idx / N;
  const float angle = angles[b];
  const float cos_val = std::cos(angle);
  const float sin_val = std::sin(angle);

  const float x = src[idx * D];
  const float y = src[idx * D + 1];
  dst[idx * D] = cos_val * x - sin_val * y;
  dst[idx * D + 1] = sin_val * y + cos_val * y;
  for (int i = 2; i < D; ++i) {
    dst[idx * D + i] = src[idx * D + i];
  }
}

int main()
{
  const int B = 2;  // Batch size
  const int N = 5;  // The number of points
  const int D = 5;  // The number of point dimensions (x, y, z, ...)
  float h_src[B][N][D] = {
    {{1.0f, 2.0f, 3.0f, 0.1f, 0.2f},
     {4.0f, 5.0f, 6.0f, 0.3f, 0.4f},
     {7.0f, 8.0f, 9.0f, 0.5f, 0.6f},
     {10.0f, 11.0f, 12.0f, 0.7f, 0.8f},
     {13.0f, 14.0f, 15.0f, 0.9f, 1.0f}},
    {{2.0f, 3.0f, 4.0f, 0.2f, 0.3f},
     {5.0f, 6.0f, 7.0f, 0.4f, 0.5f},
     {8.0f, 9.0f, 10.0f, 0.6f, 0.7f},
     {11.0f, 12.0f, 13.0f, 0.8f, 0.9f},
     {14.0f, 15.0f, 16.0f, 1.0f, 1.1f}}};

  float h_angles[B] = {M_PI / 4.0f, M_PI / 4.0f};

  float *d_src, *d_angles, *d_dst;
  const size_t Psize = sizeof(float) * B * N * D;
  const size_t Asize = sizeof(float) * B;
  cudaMalloc(reinterpret_cast<void **>(&d_src), Psize);
  cudaMalloc(reinterpret_cast<void **>(&d_angles), Asize);
  cudaMalloc(reinterpret_cast<void **>(&d_dst), Psize);
  cudaMemcpy(d_src, h_src, Psize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_angles, h_angles, Asize, cudaMemcpyHostToDevice);

  dim3 blocks(B, N);
  rotate_points_kernel<<<blocks, 256>>>(B, N, D, d_src, d_angles, d_dst);

  float h_dst[B][N][D];
  cudaMemcpy(h_dst, d_dst, Psize, cudaMemcpyDeviceToHost);

  for (int b = 0; b < B; ++b) {
    std::cout << "Batch " << b << ":\n";
    for (int n = 0; n < N; ++n) {
      std::cout << "  Point " << n << ": ";
      for (int i = 0; i < D; ++i) {
        std::cout << h_dst[b][n][i] << " ";
      }
      std::cout << "\n";
    }
  }

  cudaFree(d_src);
  cudaFree(d_angles);
  cudaFree(d_dst);
}
