#include <float.h>

#include <iostream>

/**
 * @brief Extract K polylines with the smallest distances.
 *
 * @note Because this kernel supposes to allocate shared memory dynamically it is necessary to
 * specify `sizeof(float) * L` in the kernel execution configuration.
 *
 * @param K The number of polylines to be extracted.
 * @param B The number of target agents.
 * @param L The number of original polylines.
 * @param P The number of points contained in each polyline.
 * @param D The number of point dimensions.
 * @param inPolyline Source polyline, in shape [B*L*P*D].
 * @param inPolylineMask Source polyline mask, in shape [B*L*P].
 * @param inDistance Source distances from target agents to the centers of each polyline, in shape
 * [B*L].
 * @param outPolyline Output polyline, in shape [B*K*P*D].
 * @param outPolylineMask Output polyline mask, in shape [B*K*P].
 */
__global__ void extractTopKPolylineKernel(
  const int K, const int B, const int L, const int P, const int D, const float * inPolyline,
  const bool * inPolylineMask, const float * inDistance, float * outPolyline,
  bool * outPolylineMask)
{
  int b = blockIdx.x;                             // Batch index
  int p = blockIdx.y * blockDim.y + threadIdx.y;  // Point index
  int d = blockIdx.z * blockDim.z + threadIdx.z;  // Dim index
  extern __shared__ float distances[];

  // Load distances into shared memory
  int tid = threadIdx.x;  // Polyline index
  if (tid < L) {
    distances[tid] = inDistance[b * L + tid];
  }
  __syncthreads();

  // Simple selection of the smallest K distances
  // (this part should be replaced with a more efficient sorting/selecting algorithm)
  for (int k = 0; k < K; k++) {
    float minDistance = FLT_MAX;
    int minIndex = -1;

    for (int l = 0; l < L; l++) {
      if (distances[l] < minDistance) {
        minDistance = distances[l];
        minIndex = l;
      }
    }
    __syncthreads();

    if (tid == k) {  // this thread will handle copying the k-th smallest polyline
      int inIdx = b * L * P + minIndex * P + p;
      int outIdx = b * K * P + k * P + p;
      outPolyline[outIdx * D + d] = inPolyline[inIdx * D + d];
      outPolylineMask[outIdx] = inPolylineMask[inIdx];
    }
    distances[minIndex] = FLT_MAX;  // exclude this index from future consideration
  }
}

int main()
{
  constexpr int K = 2;
  constexpr int B = 2;
  constexpr int L = 3;
  constexpr int P = 4;
  constexpr int D = 9;  // (x, y, z, dx, dy, dz, typeID, preX, preY)

  float h_inPolyline[B][L][P][D] = {
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

  bool h_inPolylineMask[B][L][P] = {
    {{true, true, true, true}, {true, true, true, true}, {true, true, true, true}},
    {{true, true, true, true}, {true, true, true, true}, {true, true, true, true}}};

  float h_inDistance[B][L] = {{3.90512, 5.31507, 6.37377}, {4.25735, 4.25735, 3.90512}};

  const size_t inPolylineNBytes = sizeof(float) * B * L * P * D;
  const size_t inPolylineMaskNBytes = sizeof(bool) * B * L * P;
  const size_t inDistanceNBytes = sizeof(float) * B * L;

  const size_t outPolylineNBytes = sizeof(float) * B * K * P * D;
  const size_t outPolylineMaskNBytes = sizeof(bool) * B * K * P;

  float *d_inPolyline, *d_inDistance, *d_outPolyline;
  bool *d_inPolylineMask, *d_outPolylineMask;
  cudaMalloc(&d_inPolyline, inPolylineNBytes);
  cudaMalloc(&d_inPolylineMask, inPolylineMaskNBytes);
  cudaMalloc(&d_inDistance, inDistanceNBytes);
  cudaMemcpy(d_inPolyline, h_inPolyline, inPolylineNBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_inPolylineMask, h_inPolylineMask, inPolylineMaskNBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_inDistance, h_inDistance, inDistanceNBytes, cudaMemcpyHostToDevice);

  cudaMalloc(&d_outPolyline, outPolylineNBytes);
  cudaMalloc(&d_outPolylineMask, outPolylineMaskNBytes);

  dim3 blocks(B, P, D);
  constexpr int THREAD_PER_BLOCK = 256;
  constexpr size_t sharedMemSize = sizeof(float) * L;
  extractTopKPolylineKernel<<<blocks, THREAD_PER_BLOCK, sharedMemSize>>>(
    K, B, L, P, D, d_inPolyline, d_inPolylineMask, d_inDistance, d_outPolyline, d_outPolylineMask);

  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << cudaGetErrorString(err) << std::endl;
  }

  float h_outPolyline[B][K][P][D];
  bool h_outPolylineMask[B][K][P];
  cudaMemcpy(h_outPolyline, d_outPolyline, outPolylineNBytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_outPolylineMask, d_outPolylineMask, outPolylineMaskNBytes, cudaMemcpyDeviceToHost);

  std::cout << "=== Out polyline ===\n";
  for (int b = 0; b < B; ++b) {
    std::cout << "Batch " << b << ":\n";
    for (int k = 0; k < K; ++k) {
      std::cout << "  Polyline " << k << ":\n";
      for (int p = 0; p < P; ++p) {
        std::cout << "  Point " << p << ": ";
        for (int d = 0; d < D; ++d) {
          std::cout << h_outPolyline[b][k][p][d] << " ";
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
        std::cout << h_outPolylineMask[b][k][p] << " ";
      }
      std::cout << "\n";
    }
  }

  cudaFree(d_inPolyline);
  cudaFree(d_inPolylineMask);
  cudaFree(d_inDistance);
  cudaFree(d_outPolyline);
  cudaFree(d_outPolylineMask);
}