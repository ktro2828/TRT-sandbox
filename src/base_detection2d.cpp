#include "base_detection2d.hpp"

#include <NvOnnxConfig.h>
#include <NvOnnxParser.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>

namespace trt
{
void BaseDetection2D::load(const std::string & path)
{
  std::ifstream file(path, std::ios::in | std::ios::binary);
  file.seekg(0, file.end);
  size_t size = file.tellg();
  file.seekg(0, file.beg);

  auto buffer = std::make_unique<char[]>(size);
  file.read(buffer.get(), size);
  file.close();
  if (runtime_) {
    engine_ =
      unique_ptr<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(buffer.get(), size));
  }
  if (!engine_) {
    std::cerr << "[ERROR] Fail to deserialize engine!!" << std::endl;
    std::exit(1);
  }
}

bool BaseDetection2D::prepare()
{
  if (!engine_) {
    std::cerr << "[WARN] Engine is unloaded!!" << std::endl;
    return false;
  }
  context_ = unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
  if (!context_) {
    std::cerr << "[WARN] Fail to create context!!" << std::endl;
    return false;
  }

  input_d_ = cuda::make_unique<float[]>(getMaxBatchSize() * getInputSize());
  scores_d_ = cuda::make_unique<float[]>(getMaxBatchSize() * getMaxDetections());
  boxes_d_ = cuda::make_unique<float[]>(getMaxBatchSize() * getMaxDetections() * 4);
  cudaStreamCreate(&stream_);

  return true;
}

BaseDetection2D::BaseDetection2D(
  const std::string & engine_path, const ModelParams & params, const bool verbose)
: params_(params)
{
  Logger logger(verbose);
  runtime_ = unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
  if (!runtime_) {
    std::cerr << "[ERROR] Fail to create runtime!!" << std::endl;
    std::exit(1);
  }
  load(engine_path);
  if (!prepare()) {
    std::cerr << "[ERROR] Fail to prepare engine!!" << std::endl;
    std::exit(1);
  }
}

BaseDetection2D::BaseDetection2D(
  const std::string & onnx_path, const ModelParams & params, const std::string & precision,
  const int max_batch_size, const bool verbose, size_t workspace_size)
: params_(params)
{
  Logger logger(verbose);
  runtime_ = unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
  if (!runtime_) {
    std::cerr << "[ERROR] Fail to create runtime!!" << std::endl;
    std::exit(1);
  }

  bool fp16 = precision.compare("FP16") == 0;
  bool int8 = precision.compare("INT8") == 0;

  auto builder = unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
  if (!builder) {
    std::cerr << "[ERROR] Fail to create builder!!" << std::endl;
    std::exit(1);
  }
  auto config = unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
  if (!config) {
    std::cerr << "[ERROR] Fail to create builder config!!" << std::endl;
    std::exit(1);
  }

  if (fp16 || int8) {
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
  }

#if (NV_TENSORRT_MAJOR * 1000) + (NV_TENSORRT_MINOR * 100) + NV_TENSOR_PATCH >= 8400
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, workspace_size);
#else
  config->setMaxWorkspaceSize(workspace_size);
#endif

  std::cout << "[INFO] Building " << precision << " core model..." << std::endl;
  const auto flag =
    1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  auto network = unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(flag));
  if (!network) {
    std::cerr << "[ERROR] Fail to create network!!" << std::endl;
    std::exit(1);
  }
  auto parser = unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
  if (!parser) {
    std::cerr << "[ERROR] Fail to create parser!!" << std::endl;
    std::exit(1);
  }

  parser->parseFromFile(onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kERROR));

  auto profile = builder->createOptimizationProfile();
  if (!profile) {
    std::cerr << "[ERROR] Fail to create profile!!" << std::endl;
    std::exit(1);
  }
  profile->setDimensions(
    network->getInput(0)->getName(), nvinfer1::OptProfileSelector::kMIN,
    nvinfer1::Dims4(1, params_.shape.channel, params_.shape.height, params_.shape.width));
  profile->setDimensions(
    network->getInput(0)->getName(), nvinfer1::OptProfileSelector::kOPT,
    nvinfer1::Dims4(
      max_batch_size, params_.shape.channel, params_.shape.height, params_.shape.width));
  profile->setDimensions(
    network->getInput(0)->getName(), nvinfer1::OptProfileSelector::kMAX,
    nvinfer1::Dims4(
      max_batch_size, params_.shape.channel, params_.shape.height, params_.shape.width));

  std::cout << "[INFO] Applying optimizations and builder TRT CUDA engine..." << std::endl;
  plan_ = unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
  if (!plan_) {
    std::cerr << "[ERROR] Fail to create serialized network!!" << std::endl;
    std::exit(1);
  }
  engine_ = unique_ptr<nvinfer1::ICudaEngine>(
    runtime_->deserializeCudaEngine(plan_->data(), plan_->size()));
  if (!engine_) {
    std::cerr << "[ERROR] Fail to create engine!!" << std::endl;
    std::exit(1);
  }
  if (!prepare()) {
    std::cerr << "[ERROR] Fail to prepare stream!!" << std::endl;
    std::exit(1);
  }
}

BaseDetection2D::~BaseDetection2D()
{
  if (stream_) {
    cudaStreamDestroy(stream_);
  }
}

void BaseDetection2D::save(const std::string & path) const
{
  std::cout << "[INFO] Writing to " << path << "..." << std::endl;
  std::ofstream file(path, std::ios::out | std::ios::binary);
  file.write(reinterpret_cast<const char *>(plan_->data()), plan_->size());
}

std::vector<float> BaseDetection2D::preprocess(const cv::Mat & img) const
{
  cv::Mat rgb;
  cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);

  cv::resize(rgb, rgb, cv::Size(params_.shape.width, params_.shape.height));
  cv::Mat img_float;
  rgb.convertTo(img_float, CV_32FC3, 1 / 255.0);

  std::vector<cv::Mat> input_data(params_.shape.channel);
  cv::split(img_float, input_data);

  std::vector<float> output(params_.shape.size());
  float * data = output.data();
  for (int i = 0; i < params_.shape.channel; ++i) {
    memcpy(data, input_data[i].data, sizeof(float) * params_.shape.area());
    data += params_.shape.area();
  }

  return output;
}

void BaseDetection2D::infer(std::vector<void *> & buffers, const int batch_size)
{
  Shape input_shape = getInputShape();
  context_->setBindingDimensions(
    0, nvinfer1::Dims4(batch_size, input_shape.channel, input_shape.height, input_shape.width));
  context_->enqueueV2(buffers.data(), stream_, nullptr);
  cudaStreamSynchronize(stream_);
}

void BaseDetection2D::detect(
  const cv::Mat & img, float * scores, float * boxes, const int batch_size)
{
  const auto input_shape = getInputShape();
  const auto input = preprocess(img);
  CHECK_CUDA_ERROR(
    cudaMemcpy(input_d_.get(), input.data(), sizeof(float) * input.size(), cudaMemcpyHostToDevice));

  std::vector<void *> buffers;
  if (params_.boxes_first) {
    buffers = {input_d_.get(), boxes_d_.get(), scores_d_.get()};
  } else {
    buffers = {input_d_.get(), scores_d_.get(), boxes_d_.get()};
  }

  infer(buffers, batch_size);

  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    scores, scores_d_.get(), sizeof(float) * getMaxDetections(), cudaMemcpyDeviceToHost));
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    boxes, boxes_d_.get(), sizeof(float) * getMaxDetections() * 4, cudaMemcpyDeviceToHost));
  cudaStreamSynchronize(stream_);
}

Shape BaseDetection2D::getInputShape() const
{
  auto dims = engine_->getBindingDimensions(0);
  return {dims.d[1], dims.d[2], dims.d[3]};
}

std::optional<Dims2> BaseDetection2D::getOutputDimensions(const std::string & name) const
{
  auto index = engine_->getBindingIndex(name.c_str());
  if (index == -1) {
    return std::nullopt;
  } else {
    auto dims = engine_->getBindingDimensions(index);
    return Dims2(dims.d[1], dims.d[2]);
  }
}

void BaseDetection2D::calculateSoftMax(std::vector<float> & scores) const
{
  float exp_sum = std::accumulate(
    scores.begin(), scores.end(), 0, [&](float acc, float e) { return acc + exp(e); });

  for (auto & s : scores) {
    s = exp(s) / exp_sum;
  }
}

cv::Mat BaseDetection2D::drawOutput(
  const cv::Mat & img, const float * scores, const float * boxes) const
{
  const auto img_w = img.cols;
  const auto img_h = img.rows;
  cv::Mat viz = img.clone();
  for (int i = 0; i < params_.num_max_detections; ++i) {
    float x_min = boxes[4 * i];
    float y_min = boxes[4 * i + 1];
    float x_max = boxes[4 * i + 2];
    float y_max = boxes[4 * i + 3];
    if (params_.denormalize_box) {
      x_min *= (img_w / params_.shape.width);
      y_min *= (img_h / params_.shape.height);
      x_max *= (img_w / params_.shape.width);
      y_max *= (img_h / params_.shape.height);
    }
    const int x1 = std::max(0, static_cast<int>(x_min));
    const int y1 = std::max(0, static_cast<int>(y_min));
    const int x2 = std::min(img_w, static_cast<int>(x_max));
    const int y2 = std::min(img_h, static_cast<int>(y_max));
    cv::rectangle(viz, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 0, 255), 1, 8);
  }

  return viz;
}

}  // namespace trt