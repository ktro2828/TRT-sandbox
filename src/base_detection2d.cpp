#include "base_detection2d.hpp"

#include <NvOnnxConfig.h>
#include <NvOnnxParser.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>

namespace trt
{

void ModelParams::loadFromFile(const std::string & yaml_path)
{
  try {
    std::cout << "[INFO] Parsing: " << yaml_path << std::endl;
    YAML::Node data = YAML::LoadFile(yaml_path);
    is_box_first = data["is_box_first"].as<bool>();
    is_box_normalized = data["is_box_normalized"].as<bool>();
    shape.width = data["width"].as<int>();
    shape.height = data["height"].as<int>();
    shape.channel = data["channel"].as<int>();
    highest_only = data["highest_only"].as<bool>();
    threshold = data["threshold"].as<float>();
    num_max_detections = data["num_max_detections"].as<int>();
    max_batch_size = data["max_batch_size"].as<int>();
    precision = data["precision"].as<std::string>();
  } catch (YAML::ParserException & e) {
    std::cerr << e.what() << std::endl;
    std::exit(1);
  } catch (YAML::BadConversion & e) {
    std::cerr << e.what() << std::endl;
    std::exit(1);
  } catch (YAML::BadFile & e) {
    std::cerr << e.what() << std::endl;
    std::exit(1);
  }
}

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
  // npp::resize(rgb.data, rgb.cols, rgb.rows, params_.shape.width, params_.shape.height, stream_);
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
  if (params_.is_box_first) {
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

std::optional<Dims2> BaseDetection2D::getOutputDimensions(const size_t index) const
{
  if (index == 1 || index == 2) {
    auto dims = engine_->getBindingDimensions(index);
    return Dims2(dims.d[1], dims.d[2]);
  }
  return std::nullopt;
}

std::vector<Detection2D> BaseDetection2D::postprocess(
  const float * scores, const float * boxes, const std::vector<std::string> & labels) const
{
  std::vector<float> score_vector;
  const size_t num_classes = labels.size();
  for (int i = 0; i < params_.num_max_detections; ++i) {
    score_vector.push_back(scores[i]);
  }
  std::vector<Detection2D> detections;
  if (params_.highest_only) {
    std::vector<float>::iterator iter = std::max_element(score_vector.begin(), score_vector.end());
    size_t index = std::distance(score_vector.begin(), iter);
    size_t box_index = 4 * index;
    float x1, y1, x2, y2;
    if (params_.is_box_normalized) {
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
    x1 = std::max(0.0f, x1);
    y1 = std::max(0.0f, y1);
    x2 = std::min(static_cast<float>(params_.shape.width), x2);
    y2 = std::min(static_cast<float>(params_.shape.height), y2);

    const size_t label_index = index % num_classes;
    Detection2D det = {x1, y1, x2 - x1, y2 - y1, score_vector[index], labels[label_index]};
    detections.emplace_back(det);
  } else {
    for (int i = 0; i < params_.num_max_detections; ++i) {
      if (score_vector.at(i) < params_.threshold) {
        continue;
      }
      const size_t box_index = 4 * i;
      float x1, y1, x2, y2;
      if (params_.is_box_normalized) {
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

      x1 = std::max(0.0f, x1);
      y1 = std::max(0.0f, y1);
      x2 = std::min(static_cast<float>(params_.shape.width), x2);
      y2 = std::min(static_cast<float>(params_.shape.height), y2);

      const size_t label_index = i % num_classes;
      Detection2D det = {x1, y1, x2 - x1, y2 - y1, score_vector[i], labels[label_index]};
      detections.emplace_back(det);
    }
  }
  return detections;
}

cv::Mat BaseDetection2D::drawOutput(
  const cv::Mat & img, const std::vector<Detection2D> & detections) const
{
  cv::Mat viz = img.clone();
  for (const auto & det : detections) {
    const int x1 = static_cast<int>(det.x);
    const int y1 = static_cast<int>(det.y);
    const int x2 = static_cast<int>(det.x + det.w);
    const int y2 = static_cast<int>(det.y + det.h);
    cv::rectangle(viz, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 0, 255), 1, 8);
    cv::putText(viz, det.label, cv::Point(x1, y1), 1, 1, cv::Scalar(0, 0, 255));
  }
  return viz;
}

}  // namespace trt