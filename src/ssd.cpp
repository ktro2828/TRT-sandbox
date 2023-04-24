#include "ssd.hpp"

namespace trt
{
SSD::SSD(const std::string & engine_path, const ModelParams & params, const bool verbose)
: BaseDetection2D(engine_path, params, verbose)
{
}

SSD::SSD(
  const std::string & onnx_path, const ModelParams & params, const std::string & precision,
  const int max_batch_size, const bool verbose, size_t workspace_size)
: BaseDetection2D(onnx_path, params, precision, max_batch_size, verbose, workspace_size)
{
}

}  // namespace trt