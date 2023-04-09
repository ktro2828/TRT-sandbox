add_library(trt_ssd SHARED trt_ssd.cpp)

target_include_directories(trt_ssd PUBLIC ${PROJECT_SOURCE_DIR}/include)
add_executable(main main.cpp)
target_include_directories(main PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(main PUBLIC ${OpenCV_LIBS} trt_ssd)