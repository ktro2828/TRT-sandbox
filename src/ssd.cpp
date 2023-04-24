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

std::vector<Detection2D> SSD::postprocess(const float * scores, const float * boxes) const
{
  std::vector<float> tlr_scores;
  constexpr int class_num = 2;
  constexpr int label_idx = 1;
  const int detection_per_class = getMaxDetections() / class_num;
  for (int i = 0; i < detection_per_class; ++i) {
    tlr_scores.push_back(scores[label_idx + i + class_num]);
  }
  std::vector<float>::iterator iter = std::max_element(tlr_scores.begin(), tlr_scores.end());
  size_t index = std::distance(tlr_scores.begin(), iter);
  size_t box_index = 4 * index;
  float x1, y1, x2, y2;
  if (params_.denormalize_box) {
    x1 = boxes[box_index] * params_.shape.width;
    y1 = boxes[box_index + 1] * params_.shape.height;
    x2 = boxes[box_index + 2] * params_.shape.width;
    y2 = boxes[box_index + 3] * params_.shape.height;
  } else {
    x1 = boxes[box_index];
    y1 = boxes[box_index + 1];
    x2 = boxes[box_index + 2];
    y2 = boxes[box_index + 3];
  }
  std::cout << "(x1, y1)=(" << x1 << ", " << y1 << ")" << std::endl;
  std::cout << "(x2, y2)=(" << x2 << ", " << y2 << ")" << std::endl;

  x1 = std::max(0.0f, x1);
  y1 = std::max(0.0f, y1);
  x2 = std::min(static_cast<float>(params_.shape.width), x2);
  y2 = std::min(static_cast<float>(params_.shape.height), y2);

  std::cout << "(x1, y1)=(" << x1 << ", " << y1 << ")" << std::endl;
  std::cout << "(x2, y2)=(" << x2 << ", " << y2 << ")" << std::endl;

  Detection2D det = {x1, y1, x2 - x1, y2 - y1, tlr_scores[index]};
  return {det};
}
}  // namespace trt