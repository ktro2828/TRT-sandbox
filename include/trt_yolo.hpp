#ifndef TRT_YOLO_HPP_
#define TRT_YOLO_HPP_

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "utils/cuda_utils.hpp"
#include "utils/trt_utils.hpp"

namespace yolo {
class Model {
  template <typename T> using unique_ptr = std::unique_ptr<T, trt::Deleter>;

private:
  unique_ptr<nvinfer1::IRuntime> runtime_{nullptr};
  unique_ptr<nvinfer1::IHostMemory> plan_{nullptr};
  unique_ptr<nvinfer1::ICudaEngine> engine_{nullptr};
  unique_ptr<nvinfer1::IExecutionContext> context_{nullptr};
  cudaStream_t stream_{nullptr};

  cuda::unique_ptr<float[]> input_d_{nullptr};
  cuda::unique_ptr<float[]> out_scores_d_{nullptr};
  cuda::unique_ptr<float[]> out_boxes_d_{nullptr};
  cuda::unique_ptr<float[]> out_classes_d_{nullptr};

  std::string box_head_name_{"boxes"};
  std::string score_head_name_{"scores"};
  int channel_{3};
  int width_{300};
  int height_{300};

  void load(const std::string &path);
  bool prepare();

  std::vector<float> preprocess(const cv::Mat &img) const;
  void infer(std::vector<void *> &buffers, const int batch_size);

public:
  explicit Model(const std::string &engine_path, bool verbose = false);
  Model(const std::string &onnx_path, const std::string &precision,
        const int max_batch_size, const bool verbose = false,
        const size_t workspace_size = (1ULL << 30));

  ~Model();

  void save(const std::string &path) const;
  bool detect(const cv::Mat &img, float *out_scores, float *out_boxes);

  std::vector<int> getInputDims() const;
  int getInputSize() const;
  int getMaxBatchSize() const;
  int getMaxDets() const;

  void setOuputHeadNames(std::string &box_head_name,
                         std::string &score_head_name);
  void setInputSize(const int channel, const int width, const int height);

}; // class Model
} // namespace yolo

#endif // TRT_YOLO_HPP_