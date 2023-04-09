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
    find_library(CUBLAS_LIBRARIES cublas HINTS
        ${CUDA_TOOLKIT_ROOT_DIR}/lib64
        ${CUDA_TOOLKIT_ROOT_DIR}/lib
    )
    message("CUDA_LIBRARIES: " ${CUDA_LIBRARIES})
    message("CUBLAS_LIBRARIES: " ${CUBLAS_LIBRARIES})
else()
    message(FATAL_ERROR "Can not find CUDA")
endif()

# For TensorRT
list(APPEND TRT_PLUGINS "nvinfer")
list(APPEND TRT_PLUGINS "nvonnxparser")
list(APPEND TRT_PLUGINS "nvparsers")

foreach(libName ${TRT_PLUGINS})
    find_library(${libName}_lib NAMES ${libName} "/usr" PATH_SUFFIES lib)
    list(APPEND TRT_PLUGIN_LIBS ${${libName}_lib})
endforeach()
message("TRT_PLUGIN_LIBS: " ${TRT_PLUGIN_LIBS})

# For subdirectories
add_subdirectory(src)
