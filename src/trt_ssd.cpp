#include "trt_ssd.hpp"

#include <NvInferPlugin.h>
#include <NvOnnxParser.h>

#include <fstream>
#include <functional>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace ssd
{
void Model::load(const std::string & path)
{
  std::ifstream file(path, std::ios::in | std::ios::binary);
  file.seekg(0, file.end);
  size_t size = file.tellg();
  file.seekg(0, file.beg);

  char * buffer = new char[size];
  file.read(buffer, size);
  file.close();
  if (runtime_) {
    engine_ = unique_ptr<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(buffer, size));
  }
  if (!engine_) {
    std::cerr << "[ERROR]: Fail to deserialize engine!!" << std::endl;
    std::exit(1);
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

Model::Model(
  const ::std::string & onnx_path, const std::string & precision, const int max_batch_size,
  const bool verbose, const size_t workspace_size)
{
  trt::Logger logger(verbose);
  runtime_ = unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
  if (!runtime_) {
    std::cerr << "[ERROR] Fail to create engine!!" << std::endl;
    std::exit(1);
  }
  bool fp16 = precision.compare("FP16") == 0;
  bool int8 = precision.compare("INT8") == 0;

  // Create builder
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

  // Parse onnx FCN
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

  // Create profile
  auto profile = builder->createOptimizationProfile();
  if (!profile) {
    std::cerr << "[ERROR] Fail to create profile!!" << std::endl;
    std::exit(1);
  }
  profile->setDimensions(
    network->getInput(0)->getName(), nvinfer1::OptProfileSelector::kMIN,
    nvinfer1::Dims4{1, channel_, height_, width_});
  profile->setDimensions(
    network->getInput(0)->getName(), nvinfer1::OptProfileSelector::kOPT,
    nvinfer1::Dims4{max_batch_size, channel_, height_, width_});
  profile->setDimensions(
    network->getInput(0)->getName(), nvinfer1::OptProfileSelector::kMAX,
    nvinfer1::Dims4{max_batch_size, channel_, height_, width_});
  config->addOptimizationProfile(profile);

  // Build engine
  std::cout << "[INFO] Applying optimizations and building TRT CUDA engine..." << std::endl;
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

Model::~Model()
{
  if (stream_) {
    cudaStreamDestroy(stream_);
  }
}

void Model::save(const std::string & path) const
{
  std::cout << "[INFO] Writing to " << path << "..." << std::endl;
  std::ofstream file(path, std::ios::out | std::ios::binary);
  file.write(reinterpret_cast<const char *>(plan_->data()), plan_->size());
}

std::vector<float> Model::preprocess(const cv::Mat & img) const
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
  float * data = output.data();
  for (int i = 0; i < channel_; ++i) {
    memcpy(data, input_data[i].data, height_ * width_ * sizeof(float));
    data += height_ * width_;
  }

  return output;
}

void Model::infer(std::vector<void *> & buffers, const int batch_size)
{
  if (!context_) {
    throw std::runtime_error("Fail to create context.");
  }

  auto input_dims = engine_->getBindingDimensions(0);
  context_->setBindingDimensions(
    0, nvinfer1::Dims4(batch_size, input_dims.d[1], input_dims.d[2], input_dims.d[3]));
  context_->enqueueV2(buffers.data(), stream_, nullptr);
  cudaStreamSynchronize(stream_);
}

bool Model::detect(const cv::Mat & img, float * out_scores, float * out_boxes)
{
  const auto input_dims = getInputDims();
  const auto input = preprocess(img);
  // BUG: cudaErrorInvalidValue (1)@: ... invalid argument
  CHECK_CUDA_ERROR(
    cudaMemcpy(input_d_.get(), input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
  // Should be changed depending on input(s)/output(s) order
  // std::vector<void *> buffers{input_d_.get(), out_scores_d_.get(), out_boxes_d_.get()};
  std::vector<void *> buffers{input_d_.get(), out_boxes_d_.get(), out_scores_d_.get()};
  try {
    infer(buffers, 1);
  } catch (const std::runtime_error & e) {
    return false;
  }
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    out_scores, out_scores_d_.get(), sizeof(float) * getMaxDets(), cudaMemcpyDeviceToHost,
    stream_));
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    out_boxes, out_boxes_d_.get(), sizeof(float) * 4 * getMaxDets(), cudaMemcpyDeviceToHost,
    stream_));
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

int Model::getMaxBatchSize() const
{
  return engine_->getProfileDimensions(0, 0, nvinfer1::OptProfileSelector::kMAX).d[0];
}
int Model::getMaxDets() const
{
  return engine_->getBindingDimensions(1).d[1];
}

void Model::setOuputHeadNames(std::string & box_head_name, std::string & score_head_name)
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

void Model::debug() const
{
  const auto dims0 = engine_->getBindingDimensions(0);
  for (auto i = 0; i < dims0.nbDims; ++i) {
    std::cout << "dims0[" << i << "]" << dims0.d[i] << std::endl;
  }
  const auto dims1 = engine_->getBindingDimensions(1);
  for (auto i = 0; i < dims1.nbDims; ++i) {
    std::cout << "dims1[" << i << "]" << dims1.d[i] << std::endl;
  }
  const auto dims2 = engine_->getBindingDimensions(2);
  for (auto i = 0; i < dims2.nbDims; ++i) {
    std::cout << "dims1[" << i << "]" << dims2.d[i] << std::endl;
  }
  const int out_dims2 =
    std::accumulate(std::begin(dims2.d), std::end(dims2.d), 1, std::multiplies<int>());
  std::cout << "Output dims2: " << out_dims2 << std::endl;

  const std::string inputs_name = engine_->getBindingName(0);
  std::cout << "Inputs name: " << inputs_name << std::endl;
  const size_t input_idx = engine_->getBindingIndex("input");
  std::cout << "Inputs index: " << input_idx << std::endl;
  const size_t scores_idx = engine_->getBindingIndex("scores");
  std::cout << "Scores index: " << scores_idx << std::endl;
  const size_t boxes_idx = engine_->getBindingIndex("boxes");
  std::cout << "Boxes index: " << boxes_idx << std::endl;
  const auto ndims = engine_->getNbBindings();
  std::cout << ndims << std::endl;
  const auto invalid_dims = engine_->getBindingDimensions(5);
  std::cout << "Invalid dims: " << typeid(invalid_dims).name() << std::endl;
  for (auto i = 0; i < invalid_dims.nbDims; ++i) {
    std::cout << "Invalid dims[" << i << "]" << invalid_dims.d[i] << std::endl;
  }
}
}  // namespace ssd
