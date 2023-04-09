#include "trt_ssd.hpp"

#include <fstream>
#include <functional>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace ssd
{
  void Model::load(const std::string &path)
  {
    std::ifstream file(path, std::ios::in | std::ios::binary);
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);

    char *buffer = new char[size];
    file.read(buffer, size);
    file.close();
    if (runtime_) {
      engine_ = unique_ptr<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(buffer, size));
    }
    delete[] buffer;
  }

  bool Model::prepare()
  {
    if (!engine_) {
      std::cerr << "[WARN] engine is unloaded!!" << std::endl;
      return false;
    }
    context_ = unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
    if (!context_) {
      std::cerr << "[WARN] context is unloaded!!" << std::endl;
      return false;
    }

    input_d_ = cuda::make_unique<float[]>(getMaxBatchSize() * getInputSize());
    out_scores_d_ = cuda::make_unique<float[]>(getMaxBatchSize() * getMaxDets());
    out_boxes_d_ = cuda::make_unique<float[]>(getMaxBatchSize() * getMaxDets() * 4);
    cudaStreamCreate(&stream_);

    return true;
  }

  Model::Model(const std::string & engine_path, bool verbose)
  {
    trt::Logger logger(verbose);
    runtime_ = unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    load(engine_path);
    if (!prepare()) {
      std::cerr << "[ERROR] Fail to prepare engine!!" << std::endl;
      std::exit(1);
    }
  }

  Model::~Model()
  {
    if (stream_) {
      cudaStreamDestroy(stream_);
    }
  }

  std::vector<float> Model::preprocess(const cv::Mat &img) const
  {
    cv::Mat rgb;
    cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);

    cv::resize(rgb, rgb, cv::Size(width_, height_));
    cv::Mat img_float;
    rgb.convertTo(img_float, CV_32FC3, 1 / 255.0);

    // HWC to CHW
    std::vector<cv::Mat> input_data(channel_);
    cv::split(img_float, input_data);

    std::vector<float> output(height_ * width_ * channel_);
    float* data = output.data();
    for (int i = 0; i < channel_; ++i)
    {
      memcpy(data, input_data[i].data, height_ * width_ * sizeof(float));
      data += height_ * width_;
    }

    return output;
  }

  void Model::infer(std::vector<void *> &buffers, const int batch_size)
  {
    if (!context_) {
      throw std::runtime_error("Fail to create context.");
    }

    auto input_dims = engine_->getBindingDimensions(0);
    context_->setBindingDimensions(0, nvinfer1::Dims4(batch_size, input_dims.d[1], input_dims.d[2], input_dims.d[3]));
    context_->enqueueV2(buffers.data(), stream_, nullptr);
    cudaStreamSynchronize(stream_);
  }

  bool Model::detect(const cv::Mat &img, float *out_scores, float *out_boxes)
  {
    const auto input_dims = getInputDims();
    const auto input = preprocess(img);
    CHECK_CUDA_ERROR(cudaMemcpy(input_d_.get(), input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
    std::vector<void *> buffers{input_d_.get(), out_scores_d_.get(), out_boxes_d_.get()};
    try {
      infer(buffers, 1);
    } catch (const std::runtime_error & e) {
      return false;
    }
    CHECK_CUDA_ERROR(cudaMemcpyAsync(out_scores, out_scores_d_.get(), sizeof(float) * getMaxDets(), cudaMemcpyDeviceToHost, stream_));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(out_boxes, out_boxes_d_.get(), sizeof(float) * 4 * getMaxDets(), cudaMemcpyDeviceToHost, stream_));
    cudaStreamSynchronize(stream_);
    return true;
  }

  std::vector<int> Model::getInputDims() const
  {
    auto dims = engine_->getBindingDimensions(0);
    return {dims.d[1], dims.d[2], dims.d[3]};
  }

  int Model::getInputSize() const
  {
    const auto input_dims = getInputDims();
    return std::accumulate(input_dims.begin(), input_dims.end(), 1, std::multiplies<int>());
  }

  int Model::getMaxBatchSize() const { return engine_->getProfileDimensions(0, 0, nvinfer1::OptProfileSelector::kMAX).d[0]; }
  int Model::getMaxDets() const { return engine_->getBindingDimensions(1).d[1]; }

  void Model::setOuputHeadNames(std::string &box_head_name, std::string &score_head_name)
  {
    box_head_name_ = box_head_name;
    score_head_name_ = score_head_name;
  }

  void Model::setInputSize(const int channel, const int width, const int height)
  {
    channel_ = channel;
    width_ = width;
    height_ = height;
  }
} // namespace ssd

