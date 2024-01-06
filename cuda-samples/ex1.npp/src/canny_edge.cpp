#include <opencv2/imgcodecs.hpp>

#include <npp.h>

#include <iostream>

int main()
{
  cv::Mat src = cv::imread("imgs/sample.jpeg", cv::IMREAD_GRAYSCALE);

  uint8_t *cudaSrc, *cudaDst;
  cudaMalloc(reinterpret_cast<void **>(&cudaSrc), src.rows * src.cols);
  cudaMalloc(reinterpret_cast<void **>(&cudaDst), src.rows * src.cols);

  cudaMemcpyAsync(cudaSrc, src.datastart, src.rows * src.cols, cudaMemcpyHostToDevice);

  NppiSize imgSize = {src.cols, src.rows};
  NppiPoint srcOffset = {0, 0};

  NppiSize dstRoi = {src.cols, src.rows};

  int bufferSize = 0;
  Npp8u * scratchBufferNPP = 0;
  NppStatus bufferStatus = nppiFilterCannyBorderGetBufferSize(dstRoi, &bufferSize);

  if (bufferStatus != NPP_SUCCESS) {
    std::cerr << "[NPP ERROR] status = " << bufferStatus << std::endl;
    return 0;
  }

  cudaMalloc(reinterpret_cast<void **>(&scratchBufferNPP), bufferSize);

  Npp16s lowThreshold = 72;
  Npp16s highThreshold = 256;
  if ((0 < bufferSize) && (scratchBufferNPP != 0)) {
    NppStatus status = nppiFilterCannyBorder_8u_C1R(
      cudaSrc, src.step, imgSize, srcOffset, cudaDst, src.step, dstRoi, NPP_FILTER_SOBEL,
      NPP_MASK_SIZE_3_X_3, lowThreshold, highThreshold, nppiNormL2, NPP_BORDER_REPLICATE,
      scratchBufferNPP);

    if (status != NPP_SUCCESS) {
      std::cerr << "[NPP ERROR] status = " << status << std::endl;
      return 0;
    }
  }

  uint8_t * cpuDst = reinterpret_cast<uint8_t *>(malloc(src.rows * src.step));
  cudaMemcpyAsync(cpuDst, cudaDst, src.rows * src.step, cudaMemcpyDeviceToHost);
  cv::Mat nppDst = cv::Mat(src.rows, src.cols, src.type(), cpuDst, src.step);

  cv::imwrite("canny_edge.png", nppDst);

  return 0;
}