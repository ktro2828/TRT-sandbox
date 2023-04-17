cmake_minimum_required(VERSION 3.14)
project(cuda_sample)
find_package(CUDA REQUIRED)

# For debug
message("--- CUDA ---")
message("version: " ${CUDA_VERSION})
message("library: " ${CUDA_CUDA_LIBRARY})
message("runtime: " ${CUDA_CUDART_LIBRARY})
message("To refer to other CUDA options, run $ cmake --help-module FindCUDA")

cuda_add_executable(vector_add vector_add.cu)
cuda_add_executable(matrix_add matrix_add.cu)
