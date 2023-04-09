#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>
#include <cstdlib>

#include "trt_ssd.hpp"

int main(int argc, char* argv[])
{
    if (argc != 2) {
        std::cerr << "[ERROR] You must specify input images path!!" << std::endl;
        std::exit(1);
    }

    std::string filename(argv[1]);
    cv::Mat img = cv::imread(filename);

    cv::imshow("img", img);
    cv::waitKey(0);
}