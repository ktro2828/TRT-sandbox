#include <opencv2/opencv.hpp>

// #if (defined(_MSC_VER) or (defined(__GNUC__) and (7 <= __GNUC_MAJOR__)))
#include <filesystem>
namespace fs = ::std::filesystem;
// #else
// #include <experimental/filesystem>
// namespace fs = ::std::experimental::filesystem;
// #endif

#include "ssd.hpp"
#include "utils/common_utils.hpp"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

int main(int argc, char * argv[])
{
  if (argc != 3) {
    std::cerr << "[ERROR] You must specify input path of image and engine!!" << std::endl;
    std::exit(1);
  }

  std::string img_path(argv[1]);
  cv::Mat img = cv::imread(img_path);
  cv::resize(img, img, cv::Size(300, 300));

  // TODO: support loading parameters from yaml?
  const trt::Shape input_shape = {3, 300, 300};
  const trt::ModelParams params("scores", "boxes", input_shape, false, false, true);
  const int max_batch_size{8};  // TODO: support 8
  const std::string precision{"FP32"};

  std::string model_path(argv[2]);
  std::string engine_path, onnx_path;
  if (model_path.substr(model_path.find_last_of(".") + 1) == "engine") {
    engine_path = model_path;
    onnx_path = fs::path{model_path}.replace_extension("onnx").string();
  } else if (model_path.substr(model_path.find_last_of(".") + 1) == "onnx") {
    onnx_path = model_path;
    engine_path = fs::path{model_path}.replace_extension("engine").string();
  } else {
    std::cerr << "[ERROR] Unexpected extension: " << model_path << std::endl;
    std::exit(1);
  }

  std::cout << "[INFO] engine: " << engine_path << ", onnx: " << onnx_path << std::endl;

  std::unique_ptr<trt::SSD> model_ptr;
  if (fs::exists(engine_path)) {
    std::cout << "[INFO] Found engine file: " << engine_path << std::endl;
    model_ptr.reset(new trt::SSD(engine_path, params));
    if (max_batch_size != model_ptr->getMaxBatchSize()) {
      std::cout << "[INFO] Required max batch size " << max_batch_size
                << "does not correspond to Profile max batch size " << model_ptr->getMaxBatchSize()
                << ". Rebuild engine from onnx." << std::endl;
      model_ptr.reset(new trt::SSD(onnx_path, params, precision, max_batch_size));
      model_ptr->save(engine_path);
    }
  } else {
    std::cout << "[INFO] Could not find " << engine_path
              << ", try making TensorRT engine from onnx." << std::endl;
    model_ptr.reset(new trt::SSD(onnx_path, params, precision, max_batch_size));
    model_ptr->save(engine_path);
  }

  auto scores =
    std::make_unique<float[]>(model_ptr->getMaxBatchSize() * model_ptr->getMaxDetections());
  auto boxes =
    std::make_unique<float[]>(model_ptr->getMaxBatchSize() * model_ptr->getMaxDetections() * 4);
  model_ptr->detect(img, scores.get(), boxes.get());

  std::vector<trt::Detection2D> detections = model_ptr->postprocess(scores.get(), boxes.get());

  cv::Mat viz = model_ptr->drawOutput(img, detections);
  cv::imshow("Output", viz);
  cv::waitKey(0);
}
