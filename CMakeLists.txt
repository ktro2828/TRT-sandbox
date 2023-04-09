cmake_minimum_required(VERSION 3.14)
project(trt_sandbox CXX)

# For OpenCV
find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
    message("OpenCV version: " ${OpenCV_VERSION})
    include_directories(${OpenCV_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "Can not find OpenCV")
endif()

# For CUDA
find_package(CUDA REQUIRED)
if (CUDA_FOUND)
    message("Cuda version: " ${CUDA_VERSION})
    include_directories(${CUDA_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "Can not find CUDA")
endif()

# For TensorRT
list(APPEND PLUGINS "nvinfer")
list(APPEND PLUGINS "nvonnxparser")
list(APPEND PLUGINS "nvparsers")

foreach(libName ${PLUGINS})
    find_library(${libName}_lib NAMES ${libName} "/usr" PATH_SUFFIES lib)
    list(APPEND PLUGIN_LIBS ${${libName}_lib})
    message("Add " ${${libName}_lib})
endforeach()

# For subdirectories
add_subdirectory(src)
