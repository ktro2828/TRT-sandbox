#include "trt_ssd.hpp"

#include <fstream>
#include <memory>
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

  void Model::prepare()
  {
    if (engine_) {
      context_ = unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
    }
    cudaStreamCreate(&stream_);
  }

  Model::Model(const std::string & engine_path, bool verbose)
  {
    trt::Logger logger(verbose);
    runtime_ = unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    load(engine_path);
    prepare();
  }

  Model::~Model()
  {
    if (stream_) {
      cudaStreamDestroy(stream_);
    }
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

  std::vector<int> Model::getInputDims() const
  {
    auto dims = engine_->getBindingDimensions(0);
    return {dims.d[1], dims.d[2], dims.d[3]};
  }

  int Model::getMaxBatchSize() const { return engine_->getProfileDimensions(0, 0, nvinfer1::OptProfileSelector::kMAX).d[0]; }
  int Model::getMaxDets() const { return engine_->getBindingDimensions(1).d[1]; }

  void Model::setOuputHeadNames(std::string &box_head_name, std::string &score_head_name)
  {
    box_head_name_ = box_head_name;
    score_head_name_ = score_head_name;
  }
} // namespace ssd

