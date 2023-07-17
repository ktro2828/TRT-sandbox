cmake_minimum_required(VERSION 3.14)
project($PROJECT_NAME)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

add_executable(warp src/warp.cpp)

if(OpenCV_FOUND)
  if(CUDA_FOUND)
    message("OpenCV & CUDA found")
    message("CUDA_LIBRARIES: " ${CUDA_LIBRARIES})
    message("CUDA_npp_LIBRARY: " ${CUDA_npp_LIBRARY})
    message("CUDA_nppc_LIBRARY: " ${CUDA_nppc_LIBRARY})
    message("CUDA_nppi_LIBRARY: " ${CUDA_nppi_LIBRARY})
    message("CUDA_npps_LIBRARY: " ${CUDA_npps_LIBRARY})

    include_directories(${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

    target_link_libraries(
      warp
      ${CUDA_npp_LIBRARY}
      ${CUDA_nppc_LIBRARY}
      ${CUDA_nppi_LIBRARY}
      ${CUDA_npps_LIBRARY}
      ${CUDA_LIBRARIES}
      ${OpenCV_LIBS})
  endif(CUDA_FOUND)
endif(OpenCV_FOUND)
