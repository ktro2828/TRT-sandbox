#include "plugins/cub_helper.h"
#include "plugins/nms_kernel.h"
#include "plugins/trt_plugin_helper.hpp"

#include <cub/cub.cuh>

template <typename T_SCORE, unsigned nthds_per_cta>
__launch_bounds__(nthds_per_cta) __global__ void prepareSortData(
  const int num, const int num_classes, const int num_preds_per_class, const int backgound_label_id,
  const float confidence_threshold, T_SCORE * conf_scores_gpu, T_SCORE * temp_scores,
  int * temp_idx, int * d_offsets)
{
  // Prepare scores data for sort
  const int cur_idx = blockIdx.x * nthds_per_cta + threadIdx.x;
}