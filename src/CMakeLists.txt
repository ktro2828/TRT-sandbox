cuda_add_library(gather_topk_plugin SHARED plugins/gather_topk_kernel.cu
                 plugins/gather_topk.cpp)

cuda_add_library(grid_priors_plugin SHARED plugins/grid_priors_kernel.cu
                 plugins/grid_priors.cpp)

target_include_directories(gather_topk_plugin
                           PUBLIC ${PROJECT_SOURCE_DIR}/include)

target_include_directories(grid_priors_plugin
                           PUBLIC ${PROJECT_SOURCE_DIR}/include)

add_library(trt_base SHARED base_detection2d.cpp)

target_link_libraries(
  trt_base
  ${TRT_PLUGINS}
  ${TRT_PLUGIN_LIBS}
  ${CUDA_LIBRARIES}
  ${CUBLAS_LIBRARIES}
  ${CUDNN_LIBRARIES}
  ${CMAKE_BINARY_DIR}/3rdparty/yaml-cpp/libyaml-cpp.a)

target_include_directories(trt_base PUBLIC ${PROJECT_SOURCE_DIR}/include)

# === ssd ===
add_library(trt_ssd SHARED ssd.cpp)
target_link_libraries(trt_ssd gather_topk_plugin grid_priors_plugin)
target_include_directories(trt_ssd PUBLIC ${PROJECT_SOURCE_DIR}/include)

# === yolo ===
add_library(trt_yolo SHARED yolo.cpp)
target_include_directories(trt_yolo PUBLIC ${PROJECT_SOURCE_DIR}/include)

add_executable(det2d det2d.cpp)
target_include_directories(det2d PUBLIC ${OpenCV_INCLUDE_DIRS}
                                        ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(det2d PUBLIC ${OpenCV_LIBS} trt_base trt_ssd trt_yolo)
