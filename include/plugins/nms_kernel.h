// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
// modify from
// https://github.com/NVIDIA/TensorRT/tree/master/plugin/batchedNMSPlugin

#ifndef NMS_KERNEL_H_
#define NMS_KERNEL_H_

#include "plugins/trt_plugin_helper.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cstdio>

template <typename T>
struct Bbox
{
  T x_min, y_min, x_max, y_max;
  Bbox(T x_min_, T y_min_, T x_max_, T y_max_)
  : x_min(x_min_), y_min(y_min_), x_max(x_max_), y_max(y_max_)
  {
  }
  Bbox() = default;
};  // struct Bbox

size_t get_cuda_arch(int devID);

int8_t * alignPtr(int8_t * ptr, uintptr_t to);

int8_t * nextWorkspacePtr(int8_t * ptr, uintptr_t previousWorkspaceSize);

size_t calculateTotalWorkspaceSize(size_t * workspace, int count);

void setUniformOffsets(cudaStream_t stream, int num_segments, int offset, int * d_offsets);

size_t detectionForwardBBoxDataSize(int N, int C1, nvinfer1::DataType DT_BBOX);

size_t detectionForwardBBoxPermuteSize(
  bool shareLocation, int N, int C1, nvinfer1::DataType DT_BBOX);

size_t detectionForwardPreNMSSize(int N, int C2);

size_t detectionForwardPostNMSSize(int N, int numClasses, int topK);

size_t detectionInterfaceWorkspaceSize(
  bool shareLocation, int N, int C1, int C2, int numClasses, int numPredictionsPerClass, int topK,
  nvinfer1::DataType DT_BBOX, nvinfer1::DataType DT_SCORE);
#endif  // NMS_KERNEL_H_