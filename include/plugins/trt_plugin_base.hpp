#ifndef TRT_PLUGIN_BASE_HPP_
#define TRT_PLUGIN_BASE_HPP_

#include <NvInferRuntime.h>
#include <NvInferVersion.h>

#include <vector>

#include "plugins/trt_plugin_helper.hpp"

namespace trt_plugin {
#if NV_TENSORRT_MAJOR > 7
#define TRT_NOEXCEPT noexcept
#else
#define TRT_NOEXCEPT
#endif

#if NV_TENSORRT_MAJOR > 7
#define TRT_NOEXCEPT noexcept
#else
#define TRT_NOEXCEPT
#endif

class TRTPluginBase : public nvinfer1::IPluginV2DynamicExt {
 public:
  TRTPluginBase(const std::string &name) : mLayerName(name) {}
  // IPluginV2 Methods
  const char *getPluginVersion() const TRT_NOEXCEPT override { return "1"; }
  int initialize() TRT_NOEXCEPT override { return STATUS_SUCCESS; }
  void terminate() TRT_NOEXCEPT override {}
  void destroy() TRT_NOEXCEPT override { delete this; }
  void setPluginNamespace(const char *pluginNamespace) TRT_NOEXCEPT override {
    mNamespace = pluginNamespace;
  }
  const char *getPluginNamespace() const TRT_NOEXCEPT override {
    return mNamespace.c_str();
  }

  virtual void configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in,
                               int nbInputs,
                               const nvinfer1::DynamicPluginTensorDesc *out,
                               int nbOutputs) TRT_NOEXCEPT override {}

  virtual size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs,
                                  int nbInputs,
                                  const nvinfer1::PluginTensorDesc *outputs,
                                  int nbOutputs) const TRT_NOEXCEPT override {
    return 0;
  }

  virtual void attachToContext(
      cudnnContext *cudnnContext, cublasContext *cublasContext,
      nvinfer1::IGpuAllocator *gpuAllocator) TRT_NOEXCEPT override {}

  virtual void detachFromContext() TRT_NOEXCEPT override {}

 protected:
  const std::string mLayerName;
  std::string mNamespace;

#if NV_TENSORRT_MAJOR < 8
 protected:
  // To prevent compiler warnings.
  using nvinfer1::IPluginV2DynamicExt::canBroadcastInputAcrossBatch;
  using nvinfer1::IPluginV2DynamicExt::enqueue;
  using nvinfer1::IPluginV2DynamicExt::getOutputDimensions;
  using nvinfer1::IPluginV2DynamicExt::isOutputBroadcastAcrossBatch;
  using nvinfer1::IPluginV2DynamicExt::supportsFormat;
#endif
};  // class TRTPluginBase

class TRTPluginCreatorBase : public nvinfer1::IPluginCreator {
 public:
  const char *getPluginVersion() const TRT_NOEXCEPT override { return "1"; };

  const nvinfer1::PluginFieldCollection *getFieldNames() TRT_NOEXCEPT override {
    return &mFC;
  }

  void setPluginNamespace(const char *pluginNamespace) TRT_NOEXCEPT override {
    mNamespace = pluginNamespace;
  }

  const char *getPluginNamespace() const TRT_NOEXCEPT override {
    return mNamespace.c_str();
  }

 protected:
  nvinfer1::PluginFieldCollection mFC;
  std::vector<nvinfer1::PluginField> mPluginAttributes;
  std::string mNamespace;
};  // class TRTPluginCreatorBase
}  // namespace trt_plugin
#endif  // TRT_PLUGIN_BASE_HPP_