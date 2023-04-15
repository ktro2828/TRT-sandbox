// Copyright (c) OpenMMLab. All rights reserved.

#include "plugins/grid_priors_kernel.hpp"
#include "plugins/trt_plugin_helper.hpp"
#include "utils/cuda_utils.hpp"

#include <cuda_fp16.h>

namespace trt_plugin
{
template <typename scalar_t>
__global__ void grid_priors_kernel(
  const scalar_t * base_anchor, scalar_t * output, int num_base_anchors, int feat_w, int feat_h,
  int stride_w, int stride_h)
{
  extern __shared__ scalar_t shared_base_anchor[];
  for (int i = threadIdx.x; i < num_base_anchors * 4; i += blockDim.x) {
    shared_base_anchor[i] = base_anchor[i];
  }
  __syncthreads();

  CUDA_1D_KERNEL_LOOP(index, num_base_anchors * feat_w * feat_h)
  {
    const int a_offset = (index % num_base_anchors) << 2;
    const scalar_t w = static_cast<scalar_t>(((index / num_base_anchors) % feat_w) * stride_w);
    const scalar_t h = static_cast<scalar_t>((index / (feat_w * num_base_anchors)) * stride_h);

    auto out_start = output + index * 4;
    out_start[0] = shared_base_anchor[a_offset] + w;
    out_start[1] = shared_base_anchor[a_offset + 1] + h;
    out_start[2] = shared_base_anchor[a_offset + 2] + w;
    out_start[3] = shared_base_anchor[a_offset + 3] + h;
  }
}

template <typename scalar_t>
void grid_priors_impl(
  const scalar_t * base_anchor, scalar_t * output, int num_base_anchors, int feat_w, int feat_h,
  int stride_w, int stride_h, cudaStream_t stream)
{
  grid_priors_kernel<<<
    GET_BLOCKS(num_base_anchors * feat_w * feat_h), THREADS_PER_BLOCK,
    DIVUP(num_base_anchors * 4, 32) * 32 * sizeof(scalar_t), stream>>>(
    base_anchor, output, num_base_anchors, feat_w, feat_h, stride_w, stride_h);
}

template void grid_priors_impl<float>(
  const float * base_anchor, float * output, int num_base_anchors, int feat_w, int feat_h,
  int stride_w, int stride_h, cudaStream_t stream);
}  // namespace trt_plugin