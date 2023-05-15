#ifndef TRT_BASE_HPP_
#define TRT_BASE_HPP_

#include "utils/cuda_utils.hpp"
#include "utils/npp_utils.hpp"
#include "utils/trt_utils.hpp"

#include <opencv2/opencv.hpp>

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <yaml-cpp/yaml.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace trt
{

struct Shape
{
  int channel, width, height;
  inline int size() const { return channel * width * height; }
  inline int area() const { return width * height; }
};  // struct Shape

class Dims2 : public nvinfer1::Dims2
{
public:
  Dims2(const int32_t d0, const int32_t d1) : nvinfer1::Dims2(d0, d1) {}
  inline int size() const { return d[0] * d[1]; }
};  // class Dims2

struct ModelParams
{
  Shape shape;
  bool is_box_first;
  bool is_box_normalized;
  bool highest_only{false};
  float threshold{0.5};
  int num_max_detections{1000};
  int max_batch_size{1};
  std::string precision{"FP32"};

  explicit ModelParams(const std::string & yaml_path) { loadFromFile(yaml_path); }

  ModelParams(
    const bool is_box_first_, const bool is_box_normalized_, const Shape & shape_,
    const bool highest_only_ = false, const float threshold_ = 0.5,
    const int num_max_detections_ = 100, const int max_batch_size_ = 1,
    const std::string & precision_ = "FP32")
  : is_box_first(is_box_first_),
    is_box_normalized(is_box_normalized_),
    shape(shape_),
    highest_only(highest_only_),
    threshold(threshold_),
    num_max_detections(num_max_detections_),
    max_batch_size(max_batch_size_),
    precision(precision_)
  {
  }

  void loadFromFile(const std::string & yaml_path);

  inline void debug() const
  {
    std::cout << "====== [DEBUG] ModelParams =====" << std::endl;
    std::cout << "is_box_first: " << is_box_first << std::endl;
    std::cout << "is_box_normalized: " << is_box_normalized << std::endl;
    std::cout << "shape: (" << shape.channel << ", " << shape.height << ", " << shape.width << ")"
              << std::endl;
    std::cout << "threshold: " << threshold << std::endl;
    std::cout << "num_max_detections: " << num_max_detections << std::endl;
    std::cout << "max_batch_size: " << max_batch_size << std::endl;
    std::cout << "precision: " << precision << std::endl;
    std::cout << "================================" << std::endl;
  }
};

struct Detection2D
{
  float x, y, w, h, score;
  std::string label;
  inline void debug() const
  {
    std::cout << "====== [DEBUG] Detection2D =====" << std::endl;
    std::cout << "(x, y, w, h)=(" << x << ", " << y << ", " << w << ", " << h << ")" << std::endl;
    std::cout << "score=" << score << std::endl;
    std::cout << "label=" << label << std::endl;
    std::cout << "================================" << std::endl;
  }
};  // struct Detection2D

class BaseDetection2D
{
  template <typename T>
  using unique_ptr = std::unique_ptr<T, Deleter>;

protected:
  unique_ptr<nvinfer1::IRuntime> runtime_{nullptr};
  unique_ptr<nvinfer1::IHostMemory> plan_{nullptr};
  unique_ptr<nvinfer1::ICudaEngine> engine_{nullptr};
  unique_ptr<nvinfer1::IExecutionContext> context_{nullptr};
  cudaStream_t stream_{nullptr};

  cuda::unique_ptr<float[]> input_d_{nullptr};
  cuda::unique_ptr<float[]> scores_d_{nullptr};
  cuda::unique_ptr<float[]> boxes_d_{nullptr};

  ModelParams params_;

  /**
   * @brief
   *
   * @param path
   */
  void load(const std::string & path);

  /**
   * @brief
   *
   * @return true
   * @return false
   */
  bool prepare();

  /**
   * @brief Preprocess of input image
   *
   * @param img
   * @return std::vector<float>
   */
  std::vector<float> preprocess(const cv::Mat & img) const;

  /**
   * @brief Do inference
   *
   * @param buffers
   * @param batch_size
   */
  void infer(std::vector<void *> & buffers, const int batch_size);

public:
  /**
   * @brief Construct a new BaseDetection2D object from engine path
   *
   * @param engine_path
   * @param verbose
   */
  explicit BaseDetection2D(
    const std::string & engine_path, const ModelParams & params, const bool verbose = false);

  /**
   * @brief Construct a new BaseDetection2D object from onnx path
   *
   * @param onnx_path
   * @param precision
   * @param max_batch_size
   * @param verbose
   * @param workspace_size
   */
  BaseDetection2D(
    const std::string & onnx_path, const ModelParams & params, const std::string & precision,
    const int max_batch_size, const bool verbose = false, size_t workspace_size = (1ULL << 30));

  /**
   * @brief Destroy the BaseDetection2D object
   *
   */
  ~BaseDetection2D();

  /**
   * @brief Save engine file
   *
   * @param path
   */
  void save(const std::string & path) const;

  /**
   * @brief
   *
   * @param img
   * @param scores
   * @param boxes
   * @param batch_size
   */
  void detect(const cv::Mat & img, float * scores, float * boxes, const int batch_size = 1);

  /**
   * @brief Get the Input Shape
   *
   * @return Shape
   */
  Shape getInputShape() const;

  inline int getInputSize() const { return getInputShape().size(); }

  /**
   * @brief Get the Output Dimensions from head name
   *
   * @param index
   * @return std::optional<Dims2>
   */
  std::optional<Dims2> getOutputDimensions(const size_t index) const;

  /**
   * @brief Get the Maximum Number of Batch Size
   *
   * @return int
   */
  inline int getMaxBatchSize() const
  {
    return engine_->getProfileDimensions(0, 0, nvinfer1::OptProfileSelector::kMAX).d[0];
  };

  /**
   * @brief Get the Maximum Number of Detections
   *
   * @return int
   */
  inline int getMaxDetections() const { return engine_->getBindingDimensions(1).d[1]; }

  virtual std::vector<Detection2D> postprocess(
    const float * scores, const float * boxes, const std::vector<std::string> & labels) const;

  /**
   * @brief Draw output boxes on image
   *
   * @param img
   * @param scores
   * @param boxes
   * @return cv::Mat
   */
  cv::Mat drawOutput(const cv::Mat & img, const std::vector<Detection2D> & detections) const;

};  // class ModelBase
}  // namespace trt
#endif  // TRT_BASE_HPP_