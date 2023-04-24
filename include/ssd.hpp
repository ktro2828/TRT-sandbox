#ifndef TRT_SSD_HPP_
#define TRT_SSD_HPP_

#include "base_detection2d.hpp"
#include "plugins/gather_topk.hpp"
#include "plugins/grid_priors.hpp"

namespace trt
{
class SSD : public BaseDetection2D
{
public:
  /**
   * @brief Construct a new SSD from engine path
   *
   * @param engine_path
   * @param params
   * @param verbose
   */
  explicit SSD(
    const std::string & engine_path, const ModelParams & params, const bool verbose = false);

  /**
   * @brief Construct a new SSD from onnx path
   *
   * @param onnx_path
   * @param params
   * @param precision
   * @param max_batch_size
   * @param verbose
   * @param workspace_size
   */
  SSD(
    const std::string & onnx_path, const ModelParams & params, const std::string & precision,
    const int max_batch_size, const bool verbose = false, size_t workspace_size = (1ULL << 30));
};  // class SSD
}  // namespace trt

#endif  // TRT_SSD_HPP_
