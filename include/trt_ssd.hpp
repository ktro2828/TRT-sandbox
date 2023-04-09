#include <cuda_runtime.h>
#include <NvInfer.h>

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "trt_utils.hpp"

namespace ssd
{
  class Model
  {

    template<typename T>
    using unique_ptr = std::unique_ptr<T, trt::Deleter>;

  private:
    unique_ptr<nvinfer1::IRuntime> runtime_{nullptr};
    unique_ptr<nvinfer1::IHostMemory> plan_{nullptr};
    unique_ptr<nvinfer1::ICudaEngine> engine_{nullptr};
    unique_ptr<nvinfer1::IExecutionContext> context_{nullptr};
    cudaStream_t stream_ = nullptr;

    std::string box_head_name_{"boxes"};
    std::string score_head_name_{"scores"};

    void load(const std::string &path);
    void prepare();

  public:
    explicit Model(const std::string &engine_path, bool verbose = false);

    ~Model();

    void infer(std::vector<void *> & buffers, const int batch_size);

    std::vector<int> getInputDims() const;
    int getMaxBatchSize() const;
    int getMaxDets() const;
    void setOuputHeadNames(std::string &box_head_name, std::string &score_head_name);

  }; // class Model
} // namespace ssd 
