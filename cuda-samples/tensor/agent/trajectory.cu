#include <iostream>

/**
 * @brief Transform predicted trajectory from target agent coords to world coords.
 *
 * @param B The number of target agents.
 * @param M The number of modes.
 * @param T The number of future timestamps.
 * @param AgentDim The number of target agent state dimensions, expecting (x, y, z, length, width,
 * height, yaw, vx, vy, ax, ay).
 * @param targetState Source target agent state, in shape [B*AgentDim].
 * @param PredDim The number of predicted state dimension, expecting (x, y, ?, ?, ?, vx, vy).
 * @param predTrajectory Predicted trajectory, in shape [B*M*T*PredDim].
 */
__global__ void transformTrajectoryKernel(
  const int B, const int M, const int T, const int AgentDim, const float * targetState,
  const int PredDim, float * predTrajectory)
{
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  int m = blockIdx.y * blockDim.y + threadIdx.y;
  int t = blockIdx.z * blockDim.z + threadIdx.z;

  if (b >= B || m >= M || t >= T) {
    return;
  }

  const int predIdx = (b * M * T + m * T + t) * PredDim;
  const float predX = predTrajectory[predIdx];
  const float predY = predTrajectory[predIdx + 1];

  const int targetIdx = b * AgentDim;
  const float targetX = targetState[targetIdx];
  const float targetY = targetState[targetIdx + 1];
  const float targetYaw = targetState[targetIdx + 6];
  const float targetCos = cos(targetYaw);
  const float targetSin = sin(targetYaw);

  predTrajectory[predIdx] = targetCos * predX + targetSin * predY + targetX;
  predTrajectory[predIdx + 1] = -targetSin * predX + targetCos * predY + targetY;
}

int main()
{
  constexpr int B = 2;
  constexpr int M = 3;
  constexpr int T = 4;
  constexpr int PredDim = 7;  // (x, y, ?, ?, ?, vx, vy)

  float h_predTrajectory[B][M][T][PredDim] = {
    {
      {
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
        {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 1.0f},
        {3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 1.0f},
        {4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 1.0f},
      },
      {
        {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 0.0f},
        {3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 0.0f},
        {4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 1.0f},
        {5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 0.0f},
      },
      {
        {3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 1.0f},
        {4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 1.0f},
        {5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 1.0f},
        {6.0f, 6.0f, 6.0f, 6.0f, 6.0f, 6.0f, 1.0f},
      },
    },
    {
      {
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
        {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 1.0f},
        {3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 1.0f},
        {4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 1.0f},
      },
      {
        {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 0.0f},
        {3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 0.0f},
        {4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 1.0f},
        {5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 0.0f},
      },
      {
        {3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 1.0f},
        {4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 1.0f},
        {5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 1.0f},
        {6.0f, 6.0f, 6.0f, 6.0f, 6.0f, 6.0f, 1.0f},
      },
    }};

  constexpr int AgentDim = 12;  // (x, y, z, length, width,height, yaw, vx, vy, ax, ay).
  float h_targetState[B][AgentDim] = {
    {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
    {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f}};

  const size_t predTrajectoryNBytes = sizeof(float) * B * M * T * PredDim;
  const size_t targetStateNBytes = sizeof(float) * B * AgentDim;

  float *d_predTrajectory, *d_targetState;

  cudaMalloc(&d_predTrajectory, predTrajectoryNBytes);
  cudaMalloc(&d_targetState, targetStateNBytes);
  cudaMemcpy(d_predTrajectory, h_predTrajectory, predTrajectoryNBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_targetState, h_targetState, targetStateNBytes, cudaMemcpyHostToDevice);

  dim3 blocks(B, M, T);
  constexpr int THREADS_PER_BLOCK = 256;
  transformTrajectoryKernel<<<blocks, THREADS_PER_BLOCK>>>(
    B, M, T, AgentDim, d_targetState, PredDim, d_predTrajectory);

  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << cudaGetErrorString(err) << std::endl;
  }

  float h_retPredTrajectory[B][M][T][PredDim];
  cudaMemcpy(h_retPredTrajectory, d_predTrajectory, predTrajectoryNBytes, cudaMemcpyDeviceToHost);

  std::cout << "=== Out trajectory ===\n";
  for (int b = 0; b < B; ++b) {
    std::cout << "Batch " << b << ":\n";
    for (int m = 0; m < M; ++m) {
      std::cout << "  Mode " << m << ":\n";
      for (int t = 0; t < T; ++t) {
        std::cout << "  Time " << t << ": ";
        for (int d = 0; d < PredDim; ++d) {
          std::cout << h_retPredTrajectory[b][m][t][d] << " ";
        }
        std::cout << "\n";
      }
    }
  }

  cudaFree(d_predTrajectory);
  cudaFree(d_targetState);
}