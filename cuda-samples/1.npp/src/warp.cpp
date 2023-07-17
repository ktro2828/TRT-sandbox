#include <opencv2/imgcodecs.hpp>

#include <npp.h>

#include <iostream>

int main()
{
  cv::Mat src = cv::imread("imgs/sample.jpeg");
  cv::Mat dst;

  uint8_t *cudaSrc, *cudaDst;
  cudaMalloc(reinterpret_cast<void **>(&cudaSrc), src.rows * src.cols * 3);
  cudaMalloc(reinterpret_cast<void **>(&cudaDst), src.rows * src.cols * 3);

  cudaMemcpyAsync(cudaSrc, src.datastart, src.rows * src.cols * 3, cudaMemcpyHostToDevice);

  NppiSize imgSize;
  imgSize.width = src.cols;
  imgSize.height = src.rows;
  // RoI: Region of Interest
  NppiRect srcRoi = {0, 0, src.cols, src.rows};
  NppiRect dstRoi = {0, 0, src.cols, src.rows};

  double matrix[3][3] = {
    {9.45135927e-01, -4.92482404e-02, -9.16291224e+01},
    {1.86556287e-02, 9.08238651e-01, 1.29333648e+01},
    {1.78247084e-05, -4.62799593e-05, 9.97536602e-01}};

  NppStatus status = nppiWarpPerspective_8u_C3R(
    cudaSrc, imgSize, src.step, srcRoi, cudaDst, src.step, dstRoi, matrix, NPPI_INTER_LINEAR);

  if (status != NPP_SUCCESS) {
    std::cerr << "[NPP ERROR] status = " << status << std::endl;
    return 0;
  }

  uint8_t * cpuDst = reinterpret_cast<uint8_t *>(malloc(src.rows * src.step));
  cudaMemcpyAsync(cpuDst, cudaDst, src.rows * src.step, cudaMemcpyDeviceToHost);
  cv::Mat nppDst = cv::Mat(src.rows, src.cols, src.type(), cpuDst, src.step);

  cv::imwrite("npp.png", nppDst);

  return 0;
}