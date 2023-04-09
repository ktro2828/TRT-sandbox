#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>
#include <cstdlib>

#include "trt_ssd.hpp"

int main(int argc, char* argv[])
{
    if (argc != 2) {
        std::cerr << "[ERROR] You must specify input image path!!" << std::endl;
        std::exit(1);
    }

    std::string filename(argv[1]);
    cv::Mat img = cv::imread(filename);

    cv::imshow("img", img);
    cv::waitKey(0);
}

// TODO: move to inference.hpp

// std::vector<cv::Mat> drawOutput(const std::vector<cv::Mat> &imgs, const float *scores, const float *boxes, const cv::Size &size)
// {
//     std::vector<cv::Mat> outputs;
//     for (const auto img : imgs)
//     {
//         cv::Mat resized;
//         cv::resize(img, resized, size)
//         outputs.emplace_back(resized);
//     }
// }