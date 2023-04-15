#ifndef BATCHED_NMS_KERNEL_HPP_
#define BATCHED_NMS_KERNEL_HPP_

#include "plugins/nms_kernel.h"

#include <cuda_runtime_api.h>

namespace trt_plugin
{
pluginStatus_t nmsInterface(
  cudaStream_t stream, const int N, const int perBatchBoxesSize, const int perBatchScoresSize,
  const bool shareLocation, const int backgroundLabelId, const int numPredictionsPerClass,
  const int numClasses, const int topK, const int keepTopK, const float scoreThreshold,
  const float iouThreshold, const nvinfer1::DataType DT_BBOX, const void * locData,
  const nvinfer1::DataType DT_SCORE, const void * confData, void * nmsedDets, void * nmsedLabels,
  void * nmsedIndex, void * workspace, bool isNormalized, bool confSigmoid, bool clipBoxes,
  bool rotated = false);
}  // namespace trt_plugin
#endif  // BATCHED_NMS_KERNEL_HPP_