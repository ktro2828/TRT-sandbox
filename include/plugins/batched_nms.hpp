// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
// modify from
// https://github.com/NVIDIA/TensorRT/tree/master/plugin/batchedNMSPlugin

#ifndef BATCHED_NMS_HPP_
#define BATCHED_NMS_HPP_

#include "plugins/plugin_base.hpp"

#include <NvInferPluginUtils.h>

#include <string>
#include <vector>

enum NMSReturnType {
  RETURN_DETS = 1,
  RETURN_INDEX = 1 << 1,
};  // enum NMSReturnType

namespace trt_plugin
{
class BatchedNMS : public TRTPluginBase
{
public:
  BatchedNMS(const std::string & name, nvinfer1::plugin::NMSParameters params, bool returnIndex);

  BatchedNMS(const std::string & name, const void * data, size_t length);

  ~BatchedNMS() noexcept override = default;

  int getNbOutputs() const noexcept override;

  nvinfer1::DimsExprs getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs * inputs, int nbInputs,
    nvinfer1::IExprBuilder & exprBuilder) noexcept override;

  size_t getWorkspaceSize(
    const nvinfer1::PluginTensorDesc * inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc * outputs, int nbOutputs) const noexcept override;

  int enqueue(
    const nvinfer1::PluginTensorDesc * inputDesc, const nvinfer1::PluginTensorDesc * outputDesc,
    const void * const * inputs, void * const * outputs, void * workSpace,
    cudaStream_t stream) noexcept override;

  size_t getSerializationSize() const noexcept override;

  void serialize(void * buffer) const noexcept override;

  void configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc * inputs, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc * outputs, int nbOutputs) noexcept override;

  bool supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc * ioDesc, int nbInputs,
    int nbOutputs) noexcept override;

  const char * getPluginType() const noexcept override;

  const char * getPluginVersion() const noexcept override;

  nvinfer1::IPluginV2DynamicExt * clone() const noexcept override;

  nvinfer1::DataType getOutputDataType(
    int index, const nvinfer1::DataType * inputType, int nbInputs) const noexcept override;

  void setClipParam(bool clip);

private:
  nvinfer1::plugin::NMSParameters param{};
  bool mClipBoxes{};
  bool mReturnIndex{};
};  // class BatchedNMS

class BatchedNMSCreator : public TRTPluginCreatorBase
{
public:
  BatchedNMSCreator();

  ~BatchedNMSCreator() noexcept override = default;

  const char * getPluginName() const noexcept override;

  const char * getPluginVersion() const noexcept override;

  nvinfer1::IPluginV2Ext * createPlugin(
    const char * name, const nvinfer1::PluginFieldCollection * mFC) noexcept override;

  nvinfer1::IPluginV2Ext * deserializePlugin(
    const char * name, const void * serialData, size_t serialLength) noexcept override;
};  // BatchedNMSCreator
}  // namespace trt_plugin

#endif  // BATCHED_NMS_HPP_