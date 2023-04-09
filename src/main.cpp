#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>
#include <cstdlib>
#include <memory>

#include "trt_ssd.hpp"

struct Config
{
    int num_max_detections{100};
    float threshold{0.5};
};  // struct config

cv::Mat drawOutput(const cv::Mat &img, const float *scores, const float *boxes, const Config &config)
{
    const auto img_w = img.cols;
    const auto img_h = img.rows;
    cv::Mat viz = img.clone();
    for (int i = 0; i < config.num_max_detections; ++i) {
        if (scores[i] < config.threshold) {
            std::cerr << "[WARN] No boxes are detected!!" << std::endl;
            break;
        }
        float x_offset = boxes[4 * i] * img_w;
        float y_offset = boxes[4 * i + 1] * img_h;
        float width = boxes[4 * i + 2] * img_w;
        float height = boxes[4 * i + 3]* img_h;
        const int left = std::max(0, static_cast<int>(x_offset));
        const int top = std::max(0, static_cast<int>(y_offset));
        const int right = std::min(img_w, static_cast<int>(x_offset +  width));
        const int bottom = std::min(img_h, static_cast<int>(y_offset + height));
        cv::rectangle(viz, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 255), 3, 8, 0);
    }

    return viz;
}

int main(int argc, char* argv[])
{
    if (argc != 3) {
        std::cerr << "[ERROR] You must specify input path of image and engine!!" << std::endl;
        std::exit(1);
    }

    std::string img_path(argv[1]);
    cv::Mat img = cv::imread(img_path);

    cv::imshow("Input img", img);
    cv::waitKey(0);

    Config config = {100, 0.5};

    std::string engine_path(argv[2]);
    ssd::Model model(engine_path, true);

    std::unique_ptr<float[]> out_scores = std::make_unique<float[]>(model.getMaxBatchSize() * model.getMaxDets());
    std::unique_ptr<float[]> out_boxes = std::make_unique<float[]>(model.getMaxBatchSize() * model.getMaxDets() * 4);
    if (!model.detect(img, out_scores.get(), out_boxes.get())) {
        std::cerr << "[ERROR] Fail to inference" << std::endl;
        std::exit(1);
    }

    cv::Mat viz = drawOutput(img, out_scores.get(), out_boxes.get(), config);
    cv::imshow("Output", viz);
    cv::waitKey(0);
}