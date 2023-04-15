// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
// modify from
// https://github.com/NVIDIA/TensorRT/tree/master/plugin/batchedNMSPlugin

#include "plugins/batched_nms.hpp"

#include "plugins/batched_nms_kernel.hpp"
#include "plugins/nms_kernel.h"
#include "plugins/trt_serialize.hpp"

#include <cstring>

namespace trt_plugin
{
namespace
{
static const char * PLUGIN_VERSION{"1"};
static const char * PLUGIN_NAME{"TRTBatchedNMS"};
}  // namespace

BatchedNMS::BatchedNMS(
  const std::string & name, nvinfer1::plugin::NMSParameters params, bool returnIndex)
: TRTPluginBase(name), param(params), mReturnIndex(returnIndex)
{
}

BatchedNMS::BatchedNMS(const std::string & name, const void * data, size_t length)
: TRTPluginBase(name)
{
  deserialize_value(&data, &length, &param);
  deserialize_value(&data, &length, &mClipBoxes);
  deserialize_value(&data, &length, &mReturnIndex);
}

int BatchedNMS::getNbOutputs() const noexcept
{
  return mReturnIndex ? 3 : 2;
}

nvinfer1::DimsExprs BatchedNMS::getOutputDimensions(
  int outputIndex, const nvinfer1::DimsExprs * inputs, int nbInputs,
  nvinfer1::IExprBuilder & exprBuilder) noexcept
{
  ASSERT(nbInputs == 2);
  ASSERT(outputIndex >= 0 && outputIndex < this->getNbOutputs());
  ASSERT(inputs[0].nbDims == 4);
  ASSERT(inputs[1].nbDims == 3);

  nvinfer1::DimsExprs ret;
  ret.d[0] = inputs[0].d[0];
  ret.d[1] = exprBuilder.constant(param.keepTopK);
  switch (outputIndex) {
    case 0:
      ret.nbDims = 3;
      ret.d[2] = exprBuilder.constant(5);
      break;
    case 1:
      ret.nbDims = 2;
      break;
    case 2:
      ret.nbDims = 2;
      break;
    default:
      break;
  }
  return ret;
}

size_t BatchedNMS::getWorkspaceSize(
  const nvinfer1::PluginTensorDesc * inputs, int nbInputs,
  const nvinfer1::PluginTensorDesc * outputs, int nbOutputs) const noexcept
{
  size_t batch_size = inputs[0].dims.d[0];
  size_t boxes_size = inputs[0].dims.d[1] * inputs[0].dims.d[2] * inputs[0].dims.d[3];
  size_t score_size = inputs[1].dims.d[1] * inputs[1].dims.d[2];
  size_t num_priors = inputs[0].dims.d[1];
  bool shareLocation = (inputs[0].dims.d[2] == 1);
  int topk = param.topK > 0 && param.topK <= inputs[1].dims.d[1] ? param.topK : inputs[1].dims.d[1];
  return detectionInterfaceWorkspaceSize(
    shareLocation, batch_size, boxes_size, score_size, param.numClasses, num_priors, topk,
    nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kFLOAT);
}

int BatchedNMS::enqueue(
  const nvinfer1::PluginTensorDesc * inputDesc, const nvinfer1::PluginTensorDesc * outputDesc,
  const void * const * inputs, void * const * outputs, void * workSpace,
  cudaStream_t stream) noexcept
{
  const void * const locData = inputs[0];
  const void * const confData = inputs[1];

  void * nmsedDets = outputs[0];
  void * nmsedLabels = outputs[1];
  void * nmsedIndex = mReturnIndex ? outputs[2] : nullptr;

  size_t batch_size = inputDesc[0].dims.d[0];
  size_t boxes_size = inputDesc[0].dims.d[1] * inputDesc[0].dims.d[2] * inputDesc[0].dims.d[3];
  size_t score_size = inputDesc[1].dims.d[1] * inputDesc[1].dims.d[2];
  size_t num_priors = inputDesc[0].dims.d[1];
  bool shareLocation = (inputDesc[0].dims.d[2] == 1);

  int topk =
    param.topK > 0 && param.topK <= inputDesc[1].dims.d[1] ? param.topK : inputDesc[1].dims.d[1];
  bool rotated = false;
  pluginStatus_t status = nmsInterface(
    stream, batch_size, boxes_size, score_size, shareLocation, param.backgroundLabelId, num_priors,
    param.numClasses, topk, param.keepTopK, param.scoreThreshold, param.iouThreshold,
    nvinfer1::DataType::kFLOAT, locData, nvinfer1::DataType::kFLOAT, confData, nmsedDets,
    nmsedLabels, nmsedIndex, workSpace, param.isNormalized, false, mClipBoxes, rotated);
  ASSERT(status == STATUS_SUCCESS);

  return 0;
}

size_t BatchedNMS::getSerializationSize() const noexcept
{
  return sizeof(nvinfer1::plugin::NMSParameters) + sizeof(mClipBoxes) + sizeof(mReturnIndex);
}

void BatchedNMS::serialize(void * buffer) const noexcept
{
  serialize_value(&buffer, param);
  serialize_value(&buffer, mClipBoxes);
  serialize_value(&buffer, mReturnIndex);
}

void BatchedNMS::configurePlugin(
  const nvinfer1::DynamicPluginTensorDesc *, int, const nvinfer1::DynamicPluginTensorDesc *,
  int) noexcept
{
}

bool BatchedNMS::supportsFormatCombination(
  int pos, const nvinfer1::PluginTensorDesc * ioDesc, int nbInputs, int nbOutputs) noexcept
{
  return (pos == 3 || pos == 4) ? ioDesc[pos].type == nvinfer1::DataType::kINT32 &&
                                    ioDesc[pos].format == nvinfer1::TensorFormat::kLINEAR
                                : ioDesc[pos].type == nvinfer1::DataType::kFLOAT &&
                                    ioDesc[pos].format == nvinfer1::TensorFormat::kLINEAR;
}

const char * BatchedNMS::getPluginType() const noexcept
{
  return PLUGIN_NAME;
}

const char * BatchedNMS::getPluginVersion() const noexcept
{
  return PLUGIN_VERSION;
}

nvinfer1::IPluginV2DynamicExt * BatchedNMS::clone() const noexcept
{
  auto * plugin = new BatchedNMS(mLayerName, param, mReturnIndex);
  plugin->setPluginNamespace(mNamespace.c_str());
  plugin->setClipParam(mClipBoxes);
  return plugin;
}

nvinfer1::DataType BatchedNMS::getOutputDataType(
  int index, const nvinfer1::DataType * inputTypes, int nbINputs) const noexcept
{
  ASSERT(index >= 0 && index < getNbOutputs());
  return (index == 1 || index == 2) ? nvinfer1::DataType::kINT32 : inputTypes[0];
}

void BatchedNMS::setClipParam(bool clip)
{
  mClipBoxes = clip;
}

// BatchedNMSCreator
BatchedNMSCreator::BatchedNMSCreator()
{
  mPluginAttributes.emplace_back(
    nvinfer1::PluginField("background_label_id", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
    nvinfer1::PluginField("num_classes", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
    nvinfer1::PluginField("topk", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
    nvinfer1::PluginField("keep_topk", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
    nvinfer1::PluginField("score_threshold", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1));
  mPluginAttributes.emplace_back(
    nvinfer1::PluginField("iou_threshold", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1));
  mPluginAttributes.emplace_back(
    nvinfer1::PluginField("is_normalized", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
    nvinfer1::PluginField("clip_boxes", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
    nvinfer1::PluginField("return_index", nullptr, nvinfer1::PluginFieldType::kINT32, 1));

  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char * BatchedNMSCreator::getPluginName() const noexcept
{
  return PLUGIN_NAME;
}

const char * BatchedNMSCreator::getPluginVersion() const noexcept
{
  return PLUGIN_VERSION;
}

nvinfer1::IPluginV2Ext * BatchedNMSCreator::createPlugin(
  const char * name, const nvinfer1::PluginFieldCollection * fc) noexcept
{
  const nvinfer1::PluginField * fields = fc->fields;
  bool clipBoxes = true;
  bool returnIndex = false;
  nvinfer1::plugin::NMSParameters params{};

  for (int i = 0; i < fc->nbFields; ++i) {
    const char * attrName = fields[i].name;
    if (!strcmp(attrName, "background_label_id")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
      params.backgroundLabelId = *(reinterpret_cast<const int *>(fields[i].data));
    } else if (!strcmp(attrName, "num_classes")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
      params.numClasses = *(reinterpret_cast<const int *>(fields[i].data));
    } else if (!strcmp(attrName, "topk")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
      params.topK = *(reinterpret_cast<const int *>(fields[i].data));
    } else if (!strcmp(attrName, "keep_topk")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
      params.keepTopK = *(reinterpret_cast<const int *>(fields[i].data));
    } else if (!strcmp(attrName, "score_threshold")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kFLOAT32);
      params.scoreThreshold = *(reinterpret_cast<const float *>(fields[i].data));
    } else if (!strcmp(attrName, "iou_threshold")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kFLOAT32);
      params.iouThreshold = *(reinterpret_cast<const float *>(fields[i].data));
    } else if (!strcmp(attrName, "is_normalized")) {
      params.isNormalized = *(reinterpret_cast<const bool *>(fields[i].data));
    } else if (!strcmp(attrName, "clip_boxes")) {
      clipBoxes = *(reinterpret_cast<const bool *>(fields[i].data));
    } else if (!strcmp(attrName, "return_index")) {
      returnIndex = *(reinterpret_cast<const bool *>(fields[i].data));
    }
  }

  BatchedNMS * plugin = new BatchedNMS(name, params, returnIndex);
  plugin->setClipParam(clipBoxes);
  plugin->setPluginNamespace(mNamespace.c_str());
  return plugin;
}

nvinfer1::IPluginV2Ext * BatchedNMSCreator::deserializePlugin(
  const char * name, const void * serialData, size_t serialLength) noexcept
{
  BatchedNMS * plugin = new BatchedNMS(name, serialData, serialLength);
  plugin->setPluginNamespace(mNamespace.c_str());
  return plugin;
}

REGISTER_TENSORRT_PLUGIN(BatchedNMSCreator);
}  // namespace trt_plugin