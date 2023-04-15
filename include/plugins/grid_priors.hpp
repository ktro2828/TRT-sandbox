// Copyright (c) OpenMMLab. All rights reserved.

#ifndef GRID_PRIORS_HPP_
#define GRID_PRIORS_HPP_

#include "plugins/plugin_base.hpp"
#include "plugins/trt_plugin_helper.hpp"

#include <NvInferRuntime.h>
#include <NvInferVersion.h>
#include <cublas_v2.h>

#include <memory>
#include <string>
#include <vector>

namespace trt_plugin
{
class GridPriors : public TRTPluginBase
{
private:
  nvinfer1::Dims mStride;
  cublasHandle_t m_cublas_handle;

public:
  explicit GridPriors(const std::string & name, const nvinfer1::Dims & stride);
  GridPriors(const std::string & name, const void *, size_t length);
  GridPriors() = delete;

  // IPluginV2 methods
  const char * getPluginVersion() const noexcept override;
  const char * getPluginType() const noexcept override;
  int getNbOutputs() const noexcept override;
  size_t getSerializationSize() const noexcept override;
  void serialize(void * buffer) const noexcept override;

  // IPluginV2Ext methods
  nvinfer1::DataType getOutputDataType(
    int index, const nvinfer1::DataType * inputTypes, int nbInputs) const noexcept override;

  // IPluginV2DynamicExt methods
  nvinfer1::IPluginV2DynamicExt * clone() const noexcept override;
  nvinfer1::DimsExprs getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs * inputs, int nbInputs,
    nvinfer1::IExprBuilder & exprBuilder) noexcept override;
  bool supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc * ioDesc, int nbInputs,
    int nbOutputs) noexcept override;
  void configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc * inputs, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc * outputs, int nbOutputs) noexcept override;
  size_t getWorkspaceSize(
    const nvinfer1::PluginTensorDesc *, int, const nvinfer1::PluginTensorDesc *,
    int) const noexcept override;
  int enqueue(
    const nvinfer1::PluginTensorDesc *, const nvinfer1::PluginTensorDesc *, const void * const *,
    void * const *, void *, cudaStream_t) noexcept override;
};  // class GridPriors

class GridPriorsCreator : public TRTPluginCreatorBase
{
public:
  GridPriorsCreator();
  const char * getPluginVersion() const noexcept override;
  const char * getPluginName() const noexcept override;
  nvinfer1::IPluginV2 * createPlugin(
    const char * name, const nvinfer1::PluginFieldCollection * fc) noexcept override;
  nvinfer1::IPluginV2 * deserializePlugin(
    const char * name, const void * serialData, size_t serialLength) noexcept override;
};  // class GridPriorsCreator
}  // namespace trt_plugin

#endif  // GRID_PRIORS_HPP_
