#include <iostream>

/**
 * @brief Calculate center distance from target agent to each polyline.
 *
 * @note Polyline must have been transformed to target agent coordinates system.
 *
 * @param B The number of target agents.
 * @param K The number of polylines.
 * @param P The number of points contained in each polyline.
 * @param D The number of point dimensions, expecting [x, y, ...].
 * @param polyline Source polyline, in shape [B*K*P*D].
 * @param polylineMask Source polyline mask, in shape [B*K*P].
 * @param distance Output calculated distances, in shape [B*K].
 */
__global__ void calculateCenterDistanceKernel(
  const int B, const int K, const int P, const int D, const float * polyline,
  const bool * polylineMask, float * distance)
{
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;
  if (b >= B || k >= K) {
    return;
  }

  // calculate polyline center
  float sumX = 0.0f, sumY = 0.0f;
  int numValid = 0;
  for (int p = 0; p < P; ++p) {
    int idx = b * K * P + k * P + p;
    if (polylineMask[idx]) {
      sumX += polyline[idx * D];
      sumY += polyline[idx * D + 1];
      ++numValid;
    }
  }
  float centerX = sumX / fmaxf(1.0f, numValid);
  float centerY = sumY / fmaxf(1.0f, numValid);

  distance[b * K + k] = hypot(centerX, centerY);
}

int main()
{
  constexpr int B = 2;
  constexpr int K = 3;
  constexpr int P = 4;
  constexpr int D = 9;  // (x, y, z, dx, dy, dz, typeID, preX, preY)

  float h_polyline[B][K][P][D] = {
    {{
       {1.0f, 1.5f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f},
       {2.0f, 2.5f, 2.0f, 2.0f, 2.0f, 2.0f, 1.0f, 0.0f, 0.0f},
       {3.0f, 3.5f, 3.0f, 3.0f, 3.0f, 3.0f, 1.0f, 0.0f, 0.0f},
       {4.0f, 4.5f, 4.0f, 4.0f, 4.0f, 4.0f, 1.0f, 0.0f, 0.0f},
     },
     {
       {2.0f, 2.5f, 2.0f, 2.0f, 2.0f, 2.0f, 1.0f, 0.0f, 0.0f},
       {3.0f, 3.5f, 3.0f, 3.0f, 3.0f, 3.0f, 1.0f, 0.0f, 0.0f},
       {4.0f, 4.5f, 4.0f, 4.0f, 4.0f, 4.0f, 1.0f, 0.0f, 0.0f},
       {5.0f, 5.5f, 5.0f, 5.0f, 5.0f, 5.0f, 1.0f, 0.0f, 0.0f},
     },
     {
       {3.0f, 3.5f, 3.0f, 3.0f, 3.0f, 3.0f, 1.0f, 0.0f, 0.0f},
       {4.0f, 4.5f, 4.0f, 4.0f, 4.0f, 4.0f, 1.0f, 0.0f, 0.0f},
       {5.0f, 5.5f, 5.0f, 5.0f, 5.0f, 5.0f, 1.0f, 0.0f, 0.0f},
       {5.0f, 5.5f, 5.0f, 5.0f, 5.0f, 5.0f, 1.0f, 0.0f, 0.0f},
     }},
    {{
       {1.0f, 1.5f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f},
       {2.0f, 2.5f, 2.0f, 2.0f, 2.0f, 2.0f, 1.0f, 0.0f, 0.0f},
       {3.0f, 3.5f, 3.0f, 3.0f, 3.0f, 3.0f, 1.0f, 0.0f, 0.0f},
       {5.0f, 5.5f, 5.0f, 5.0f, 5.0f, 5.0f, 1.0f, 0.0f, 0.0f},
     },
     {
       {1.0f, 1.5f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f},
       {2.0f, 2.5f, 2.0f, 2.0f, 2.0f, 2.0f, 1.0f, 0.0f, 0.0f},
       {3.0f, 3.5f, 3.0f, 3.0f, 3.0f, 3.0f, 1.0f, 0.0f, 0.0f},
       {5.0f, 5.5f, 5.0f, 5.0f, 5.0f, 5.0f, 1.0f, 0.0f, 0.0f},
     },
     {
       {1.0f, 1.5f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f},
       {2.0f, 2.5f, 2.0f, 2.0f, 2.0f, 2.0f, 1.0f, 0.0f, 0.0f},
       {3.0f, 3.5f, 3.0f, 3.0f, 3.0f, 3.0f, 1.0f, 0.0f, 0.0f},
       {4.0f, 4.5f, 4.0f, 4.0f, 4.0f, 4.0f, 1.0f, 0.0f, 0.0f},
     }}};

  bool h_polylineMask[B][K][P] = {
    {{true, true, true, true}, {true, true, true, true}, {true, true, true, true}},
    {{true, true, true, true}, {true, true, true, true}, {true, true, true, true}}};

  const size_t polylineNBytes = sizeof(float) * B * K * P * D;
  const size_t polylineMaskNBytes = sizeof(bool) * B * K * P;
  const size_t distanceNBytes = sizeof(float) * B * K;

  float *d_polyline, *d_distance;
  bool * d_polylineMask;
  cudaMalloc(&d_polyline, polylineNBytes);
  cudaMalloc(&d_polylineMask, polylineMaskNBytes);
  cudaMemcpy(d_polyline, h_polyline, polylineNBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_polylineMask, h_polylineMask, polylineMaskNBytes, cudaMemcpyHostToDevice);

  cudaMalloc(&d_distance, distanceNBytes);

  dim3 blocks(B, K);
  constexpr int THREADS_PER_BLOCK = 256;
  calculateCenterDistanceKernel<<<blocks, THREADS_PER_BLOCK>>>(
    B, K, P, D, d_polyline, d_polylineMask, d_distance);

  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << cudaGetErrorString(err) << std::endl;
  }

  float h_distance[B][K];
  cudaMemcpy(h_distance, d_distance, distanceNBytes, cudaMemcpyDeviceToHost);

  std::cout << "=== Out distance ===\n";
  for (int b = 0; b < B; ++b) {
    std::cout << "Batch " << b << ": ";
    for (int k = 0; k < K; ++k) {
      std::cout << h_distance[b][k] << " ";
    }
    std::cout << "\n";
  }

  cudaFree(d_polyline);
  cudaFree(d_polylineMask);
  cudaFree(d_distance);
}