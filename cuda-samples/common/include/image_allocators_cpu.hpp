#ifndef NPP_IMAGE_ALLOCATORS_CPU_HPP_
#define NPP_IMAGE_ALLOCATORS_CPU_HPP_

#include <memory.h>

namespace npp
{
template <typename D, size_t N>
class ImageAllocatorCPU
{
public:
  static D * malloc(unsigned int width, unsigned int height, unsigned int * pitch_ptr)
  {
    D * p_result = new D[width * N * height];
    *pitch_ptr = width * sizeof(D) * N;
    return p_result;
  }

  static void free(D * pix_ptr) { delete[] pix_ptr; }

  static void copy(
    D * dst_ptr, size_t dst_pitch, const D * src_ptr, size_t src_pitch, size_t width, size_t height)
  {
    const void * src_line_ptr = src_ptr;
    void * dst_line = dst_ptr;

    for (size_t i = 0; i < height; ++i) {
      memcpy(dst_ptr, src_ptr, width * N * sizeof(D));
      dst_ptr += dst_pitch;
      src_ptr += src_pitch;
    }
  }
};  // class ImageAllocatorCPU
}  //  namespace npp
#endif  // NPP_IMAGE_ALLOCATORS_CPU_HPP_