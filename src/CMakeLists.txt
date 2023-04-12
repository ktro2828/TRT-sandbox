cuda_add_library(gather_topk_plugin SHARED plugins/gather_topk_kernel.cu
                 plugins/gather_topk.cpp)

cuda_add_library(grid_priors_plugin SHARED plugins/grid_priors_kernel.cu
                 plugins/grid_priors.cpp)

target_include_directories(gather_topk_plugin
                           PUBLIC ${PROJECT_SOURCE_DIR}/include)

target_include_directories(grid_priors_plugin
                           PUBLIC ${PROJECT_SOURCE_DIR}/include)

add_library(trt_ssd SHARED trt_ssd.cpp)

target_link_libraries(
  trt_ssd
  ${TRT_PLUGINS}
  ${TRT_PLUGIN_LIBS}
  ${CUDA_LIBRARIES}
  ${CUBLAS_LIBRARIES}
  ${CUDNN_LIBRARIES}
  gather_topk_plugin
  grid_priors_plugin)

target_include_directories(trt_ssd PUBLIC ${PROJECT_SOURCE_DIR}/include)
add_executable(det2d det2d.cpp)
target_include_directories(det2d PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(det2d PUBLIC ${OpenCV_LIBS} trt_ssd)

install(
  TARGETS gather_topk_plugin grid_priors_plugin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin)
