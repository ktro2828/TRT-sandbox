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

void USAGE()
{
  std::cout << "[USAGE] ./src/det2d <IMG_PATH> <ENGINE_OR_ONNX_PATH> <LABEL_PATH> <CONFIG_PATH>"
            << std::endl;
}

int main(int argc, char * argv[])
{
  if (argc != 5) {
    std::cerr << "[ERROR] You must specify input path of image, engine, labels and param!!"
              << std::endl;
    USAGE();
    std::exit(1);
  }

  std::string param_path(argv[4]);
  const trt::ModelParams params(param_path);
  params.debug();
  const int max_batch_size = params.max_batch_size;
  const std::string precision = params.precision;

  std::string img_path(argv[1]);
  cv::Mat img = cv::imread(img_path);
  cv::resize(img, img, cv::Size(params.shape.width, params.shape.height));

  std::string label_path(argv[3]);
  std::vector<std::string> labels = readLabelFile(label_path);

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

  std::vector<trt::Detection2D> detections =
    model_ptr->postprocess(scores.get(), boxes.get(), labels);

  std::for_each(detections.begin(), detections.end(), [&](const auto & d) { d.debug(); });

  cv::Mat viz = model_ptr->drawOutput(img, detections);
  cv::imshow("Output", viz);
  cv::waitKey(0);
}
