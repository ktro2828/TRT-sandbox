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

  NppiSize imgSize = {src.cols, src.rows};
  NppiSize dstMask = {5, 5};
  NppiPoint srcOffset = {0, 0};
  NppiSize dstRoi = {src.cols, src.rows};
  NppiPoint dstAnchor = {dstMask.width / 2, dstMask.height / 2};

  NppStatus status = nppiFilterBoxBorder_8u_C3R(
    cudaSrc, src.step, imgSize, srcOffset, cudaDst, src.step, dstRoi, dstMask, dstAnchor,
    NPP_BORDER_REPLICATE);

  if (status != NPP_SUCCESS) {
    std::cerr << "[NPP ERROR] status = " << status << std::endl;
    return 0;
  }

  uint8_t * cpuDst = reinterpret_cast<uint8_t *>(malloc(src.rows * src.step));
  cudaMemcpyAsync(cpuDst, cudaDst, src.rows * src.step, cudaMemcpyDeviceToHost);
  cv::Mat nppDst = cv::Mat(src.rows, src.cols, src.type(), cpuDst, src.step);

  cv::imwrite("box_filter.png", nppDst);

  return 0;
}