#ifndef NPP_IMAGE_PACKED_HPP_
#define NPP_IMAGE_PACKED_HPP_
#include "image.hpp"
#include "pixle.hpp"
namespace npp
{
template <typename D, size_t N, class A>
class ImagePacked : public Image
{
public:
  using tPixel = Pixel<D, N>;
  using tData = D;
  static const size_t gnChannels = N;
  using tSize = Image::Size;

  ImagePacked() : pixels_(0), pitch_(0) {}
  ImagePacked(const unsigned int w, const unsigned int h) : Image(w, h), pixels_(0), pitch_(0) {}
  ImagePacked(const unsigned int w, const unsigned int h) : Image(w, h), pixels_(0), pitch_(0)
  {
    pixels_ = A::malloc(width(), height(), &pitch_);
  }
  ImagePacked(const unsigned int w, unsigned int h, bool tight) : Image(w, h), pixels_(0), pitch_(0)
  {
    pixels_ = A::malloc(width(), height(), &pitch_, tight);
  }

private:
  D * pixels_;
  unsigned int pitch_;
};  // class ImagePacked
}  // namespace npp
#endif  // NPP_IMAGE_PACKED_HPP_