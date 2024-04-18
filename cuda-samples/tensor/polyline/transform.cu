#include <iostream>

/**
 * @brief Transform polylines to target agent coordinate system and extend its feature with previous
 * x, y`(D->D+2)`.
 * @brief Points which all elements are 0.0 except of typeID are filled by 0.0 and the corresponding
 * mask value is 0.0 too.
 *
 * @param K The number of polylines.
 * @param P The number of points contained in each polyline.
 * @param PointDim The number of point dimensions, expecting (x, y, z, dx, dy, dz, typeID).
 * @param inPolyline Source polyline, in shape [K*P*D].
 * @param B The number of target agents.
 * @param AgentDim The number of agent state dimensions, expecting (x, y, z, length, width, height,
 * yaw, vx, vy, ax, ay).
 * @param targetState Source target agent states, in shape [B*AgentDim].
 * @param outPolyline Output polyline, in shape [B*K*P*(PointDim+2)].
 * @param outPolylineMask Output polyline mask, in shape [B*K*P].
 */
__global__ void transformPolylineKernel(
  const int K, const int P, const int PointDim, const float * inPolyline, const int B,
  const int AgentDim, const float * targetState, float * outPolyline, bool * outPolylineMask)
{
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;
  int p = blockIdx.z * blockDim.z + threadIdx.z;

  if (b >= B || k >= K || p >= P) {
    return;
  }

  const int inIdx = (k * P + p) * PointDim;
  const int outIdx = b * K * P + k * P + p;
  bool isValid = false;
  for (int d = 0; d < PointDim - 1; ++d) {
    if (inPolyline[inIdx + d] != 0.0f) {
      isValid = true;
    }
  }
  outPolylineMask[outIdx] = isValid;

  // initialize output polyline with 0.0
  for (int d = 0; d < PointDim + 2; ++d) {
    outPolyline[outIdx * (PointDim + 2) + d] = 0.0f;
  }

  // set transformed values if valid, otherwise all 0.0.
  if (isValid) {
    const float x = inPolyline[inIdx];
    const float y = inPolyline[inIdx + 1];
    const float z = inPolyline[inIdx + 2];
    const float dx = inPolyline[inIdx + 3];
    const float dy = inPolyline[inIdx + 4];
    const float dz = inPolyline[inIdx + 5];
    const float typeID = inPolyline[inIdx + 6];

    const int centerIdx = b * AgentDim;
    const float centerX = targetState[centerIdx];
    const float centerY = targetState[centerIdx + 1];
    const float centerZ = targetState[centerIdx + 2];
    const float centerYaw = targetState[centerIdx + 6];
    const float centerCos = cosf(centerYaw);
    const float centerSin = sinf(centerYaw);

    // do transform
    const float transX = centerCos * (x - centerX) - centerSin * (y - centerY);
    const float transY = centerSin * (x - centerX) + centerCos * (y - centerY);
    const float transZ = z - centerZ;
    const float transDx = centerCos * dx - centerSin * dy;
    const float transDy = centerSin * dx + centerCos * dy;
    const float transDz = dz;

    outPolyline[outIdx * (PointDim + 2)] = transX;
    outPolyline[outIdx * (PointDim + 2) + 1] = transY;
    outPolyline[outIdx * (PointDim + 2) + 2] = transZ;
    outPolyline[outIdx * (PointDim + 2) + 3] = transDx;
    outPolyline[outIdx * (PointDim + 2) + 4] = transDy;
    outPolyline[outIdx * (PointDim + 2) + 5] = transDz;
    outPolyline[outIdx * (PointDim + 2) + 6] = typeID;
  }
}

int main()
{
  constexpr int K = 3;
  constexpr int P = 4;
  constexpr int PointDim = 7;  // (x, y, z, dx, dy, dz, typeID)

  float h_inPolyline[K][P][PointDim] = {
    {
      {1.0f, 1.5f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
      {2.0f, 2.5f, 2.0f, 2.0f, 2.0f, 2.0f, 1.0f},
      {3.0f, 3.5f, 3.0f, 3.0f, 3.0f, 3.0f, 1.0f},
      {4.0f, 4.5f, 4.0f, 4.0f, 4.0f, 4.0f, 1.0f},
    },
    {
      {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f},  // invalid point
      {2.0f, 2.5f, 2.0f, 2.0f, 2.0f, 2.0f, 1.0f},
      {3.0f, 3.5f, 3.0f, 3.0f, 3.0f, 3.0f, 1.0f},
      {4.0f, 4.5f, 4.0f, 4.0f, 4.0f, 4.0f, 1.0f},
    },
    {
      {1.0f, 1.5f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
      {2.0f, 2.5f, 2.0f, 2.0f, 2.0f, 2.0f, 1.0f},
      {3.0f, 3.5f, 3.0f, 3.0f, 3.0f, 3.0f, 1.0f},
      {4.0f, 4.5f, 4.0f, 4.0f, 4.0f, 4.0f, 1.0f},
    }};

  constexpr int B = 2;
  constexpr int AgentDim = 12;  // (x, y, z, length, width, height, yaw, vx, vy, ax, ay).
  float h_targetState[B][AgentDim] = {
    {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
    {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f}};

  const size_t inPolylineNBytes = sizeof(float) * K * P * PointDim;
  const size_t targetStateNBytes = sizeof(float) * B * AgentDim;
  const size_t outPolylineNBytes = sizeof(float) * B * K * P * (PointDim + 2);
  const size_t outPolylineMaskNBytes = sizeof(bool) * B * K * P;

  float *d_inPolyline, *d_targetState, *d_outPolyline;
  bool * d_outPolylineMask;

  cudaMalloc(&d_inPolyline, inPolylineNBytes);
  cudaMalloc(&d_targetState, targetStateNBytes);
  cudaMemcpy(d_inPolyline, h_inPolyline, inPolylineNBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_targetState, h_targetState, targetStateNBytes, cudaMemcpyHostToDevice);

  cudaMalloc(&d_outPolyline, outPolylineNBytes);
  cudaMalloc(&d_outPolylineMask, outPolylineMaskNBytes);

  dim3 blocks(B, K, P);
  constexpr int THREADS_PER_BLOCK = 256;
  transformPolylineKernel<<<blocks, THREADS_PER_BLOCK>>>(
    K, P, PointDim, d_inPolyline, B, AgentDim, d_targetState, d_outPolyline, d_outPolylineMask);

  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << cudaGetErrorString(err) << std::endl;
  }

  float h_outPolyline[B][K][P][PointDim + 2];
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
        for (int d = 0; d < (PointDim + 2); ++d) {
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
  cudaFree(d_targetState);
  cudaFree(d_outPolyline);
  cudaFree(d_outPolylineMask);
}