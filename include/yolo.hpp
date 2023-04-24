#ifndef TRT_YOLO_HPP_
#define TRT_YOLO_HPP_

#include "base_detection2d.hpp"

namespace trt
{
class YOLO : public BaseDetection2D
{
public:
  /**
   * @brief Construct a new YOLO from engine path
   *
   * @param engine_path
   * @param params
   * @param verbose
   */
  explicit YOLO(
    const std::string & engine_path, const ModelParams & params, const bool verbose = false);

  /**
   * @brief Construct a new YOLO from onnx path
   *
   * @param onnx_path
   * @param params
   * @param precision
   * @param max_batch_size
   * @param verbose
   * @param workspace_size
   */
  YOLO(
    const std::string & onnx_path, const ModelParams & params, const std::string & precision,
    const int max_batch_size, const bool verbose = false, size_t workspace_size = (1ULL << 30));

};  // class YOLO
}  // namespace trt

#endif  // TRT_YOLO_HPP_