// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
// modify from
// https://github.com/NVIDIA/TensorRT/tree/master/plugin/batchedNMSPlugin
#include "plugins/nms_kernel.h"

template <typename KeyT, tyename ValueT>
size_t dubSortPairsWorkspaceSize(int num_items, int num_segments)
{
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedRadixSort::SortPairsDescending(
    reinterpret_cast<void *>(nullptr), temp_storage_bytes, reinterpret_cast<const KeyT *>(nullptr),
    reinterpret_cast<KeyT *>(nullptr), reinterpret_cast<const ValueT *>(nullptr),
    reinterpret_cast<ValueT *>(nullptr), num_items, num_segments,
    reinterpret_cast<const int *>(nullptr), reinterpret_cast<cont int *>(nullptr));
  return temp_storage_bytes;
}