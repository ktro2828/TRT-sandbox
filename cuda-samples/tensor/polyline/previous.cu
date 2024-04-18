#include <iostream>

/**
 * @brief Set the Previous Position Kernel object
 *
 * @param B The number of target agents.
 * @param K The number of polylines.
 * @param P The number of points contained in each polyline.
 * @param D The number of point dimensions, expecting (x, y, ..., preX, preY).
 * @param polyline Source polyline, in shape [B*K*P*D].
 */
__global__ void setPreviousPositionKernel(
  const int B, const int K, const int P, const int D, float * polyline)
{
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;
  int p = blockIdx.z * blockDim.z + threadIdx.z;

  if (b >= B || k >= K || p >= P) {
    return;
  }

  const int curIdx = (b * K * P + k * P + p) * D;
  const int preIdx = p == 0 ? curIdx : (b * K * P + k * P + p - 1) * D;

  polyline[curIdx + D - 2] = polyline[preIdx];      // x
  polyline[curIdx + D - 1] = polyline[preIdx + 1];  // y
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
       {1.0f, 1.5f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f},
       {2.0f, 2.5f, 2.0f, 2.0f, 2.0f, 2.0f, 1.0f, 0.0f, 0.0f},
       {3.0f, 3.5f, 3.0f, 3.0f, 3.0f, 3.0f, 1.0f, 0.0f, 0.0f},
       {4.0f, 4.5f, 4.0f, 4.0f, 4.0f, 4.0f, 1.0f, 0.0f, 0.0f},
     },
     {
       {1.0f, 1.5f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f},
       {2.0f, 2.5f, 2.0f, 2.0f, 2.0f, 2.0f, 1.0f, 0.0f, 0.0f},
       {3.0f, 3.5f, 3.0f, 3.0f, 3.0f, 3.0f, 1.0f, 0.0f, 0.0f},
       {4.0f, 4.5f, 4.0f, 4.0f, 4.0f, 4.0f, 1.0f, 0.0f, 0.0f},
     }},
    {{
       {1.0f, 1.5f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f},
       {2.0f, 2.5f, 2.0f, 2.0f, 2.0f, 2.0f, 1.0f, 0.0f, 0.0f},
       {3.0f, 3.5f, 3.0f, 3.0f, 3.0f, 3.0f, 1.0f, 0.0f, 0.0f},
       {4.0f, 4.5f, 4.0f, 4.0f, 4.0f, 4.0f, 1.0f, 0.0f, 0.0f},
     },
     {
       {1.0f, 1.5f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f},
       {2.0f, 2.5f, 2.0f, 2.0f, 2.0f, 2.0f, 1.0f, 0.0f, 0.0f},
       {3.0f, 3.5f, 3.0f, 3.0f, 3.0f, 3.0f, 1.0f, 0.0f, 0.0f},
       {4.0f, 4.5f, 4.0f, 4.0f, 4.0f, 4.0f, 1.0f, 0.0f, 0.0f},
     },
     {
       {1.0f, 1.5f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f},
       {2.0f, 2.5f, 2.0f, 2.0f, 2.0f, 2.0f, 1.0f, 0.0f, 0.0f},
       {3.0f, 3.5f, 3.0f, 3.0f, 3.0f, 3.0f, 1.0f, 0.0f, 0.0f},
       {4.0f, 4.5f, 4.0f, 4.0f, 4.0f, 4.0f, 1.0f, 0.0f, 0.0f},
     }}};

  const size_t nBytes = sizeof(float) * B * K * P * D;

  float * d_polyline;
  cudaMalloc(&d_polyline, nBytes);
  cudaMemcpy(d_polyline, h_polyline, nBytes, cudaMemcpyHostToDevice);

  dim3 blocks(B, K, P);
  constexpr int THREADS_PER_BLOCK = 256;
  setPreviousPositionKernel<<<blocks, THREADS_PER_BLOCK>>>(B, K, P, D, d_polyline);

  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << cudaGetErrorString(err) << std::endl;
  }

  float h_retPolyline[B][K][P][D];
  cudaMemcpy(h_retPolyline, d_polyline, nBytes, cudaMemcpyDeviceToHost);

  std::cout << "=== Out polyline ===\n";
  for (int b = 0; b < B; ++b) {
    std::cout << "Batch " << b << ":\n";
    for (int k = 0; k < K; ++k) {
      std::cout << "  Polyline " << k << ":\n";
      for (int p = 0; p < P; ++p) {
        std::cout << "  Point " << p << ": ";
        for (int d = 0; d < D; ++d) {
          std::cout << h_retPolyline[b][k][p][d] << " ";
        }
        std::cout << "\n";
      }
    }
  }
  cudaFree(d_polyline);
}