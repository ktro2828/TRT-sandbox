#include <NvInfer.h>

#include <iostream>


namespace trt
{
  struct Deleter
  {
    template<typename T>
    void operator()(T *obj) const
    {
      if (obj) {
	delete obj;
      }
    }
  }; // struct Deleter

  class Logger : public nvinfer1::ILogger
  {
  private:
    bool verbose_{true};

  public:
    explicit Logger(bool verbose) : verbose_(verbose) {}

    void log(Serverity serverity, const char *msg) noexcept override
    {
      if (verbose_ || ((serverity != Serverity::kINFO) && (serverity != Serverity::kVERBOSE))) {
	std::cout << msg << std::endl;
      }
    }
  }; // class Logger

} // namespace trt_utils
