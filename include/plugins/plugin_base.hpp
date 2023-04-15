// Copyright (c) OpenMMLab. All rights reserved.

#ifndef PLUGIN_BASE_HPP_
#define PLUGIN_BASE_HPP_

#include "plugins/trt_plugin_helper.hpp"

#include <NvInferRuntime.h>
#include <NvInferVersion.h>

#include <string>
#include <vector>

namespace trt_plugin
{
class TRTPluginBase : public nvinfer1::IPluginV2DynamicExt
{
public:
  TRTPluginBase(const std::string & name) : mLayerName(name) {}
  const char * getPluginVersion() const noexcept override { return "1"; }
  int initialize() noexcept override { return STATUS_SUCCESS; }
  void terminate() noexcept override {}
  void destroy() noexcept override { delete this; }
  void setPluginNamespace(const char * pluginNamespace) noexcept override
  {
    mNamespace = pluginNamespace;
  }
  const char * getPluginNamespace() const noexcept override { return mNamespace.c_str(); }
  virtual void configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc * in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc * out, int nbOutputs) noexcept override
  {
  }

  virtual size_t getWorkspaceSize(
    const nvinfer1::PluginTensorDesc * inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc * outputs, int nbOutputs) const noexcept override
  {
    return 0;
  }

  virtual void attachToContext(
    cudnnContext * cudnnContext, cublasContext * cublasContext,
    nvinfer1::IGpuAllocator * gpuAllocator) noexcept override
  {
  }

  virtual void detachFromContext() noexcept override {}

protected:
  const std::string mLayerName;
  std::string mNamespace;
};  // class TRTPluginBase

class TRTPluginCreatorBase : public nvinfer1::IPluginCreator
{
public:
  const char * getPluginVersion() const noexcept override { return "1"; }

  const nvinfer1::PluginFieldCollection * getFieldNames() noexcept override { return &mFC; }

  void setPluginNamespace(const char * pluginNamespace) noexcept override
  {
    mNamespace = pluginNamespace;
  }

  const char * getPluginNamespace() const noexcept override { return mNamespace.c_str(); }

protected:
  nvinfer1::PluginFieldCollection mFC;
  std::vector<nvinfer1::PluginField> mPluginAttributes;
  std::string mNamespace;
};  // class TRTPluginCreatorBase

}  // namespace trt_plugin

#endif  // PLUGIN_BASE_HPP_