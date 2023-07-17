#ifndef NPP_IMAGE_HPP_
#define NPP_IMAGE_HPP_

#include "exceptions.hpp"

#include <algorithm>

namespace npp
{
class Image
{
public:
  struct Size
  {
    unsigned int width;
    unsigned int height;

    Size() : width(0), height(0) {}
    Size(unsigned int w, unsigned int h) : width(w), height(h) {}
    Size(const Size & size) : width(size.width), height(size.height) {}

    Size & operator=(const Size & size)
    {
      if (&size == this) {
        return *this;
      } else {
        width = size.width;
        height = size.height;
        return *this;
      }
    }

    void swap(Size & size)
    {
      std::swap(width, size.width);
      std::swap(height, size.height);
    }
  };  // struct Size

  Image() = default;
  Image(unsigned int w, unsigned int h) : size_(w, h) {}
  Image(const Image & img) : size_(img.size()) {}

  Image & operator=(const Image & img)
  {
    if (&img == this) {
      return *this;
    } else {
      size_ = img.size();
      return *this;
    }
  }

  const Size & size() const { return size_; }
  unsigned int width() const { return size_.width; }
  unsigned int height() const { return size_.height; }

private:
  Size size_;
};  // class Image

bool operator==(const Image::Size & lhs, const Image::Size & rhs)
{
  return lhs.width == rhs.width && lhs.height == rhs.height;
}

bool operator!=(const Image::Size & lhs, const Image::Size & rhs)
{
  return lhs.width != rhs.width || lhs.height != rhs.height;
}

bool operator==(const Image & lhs, const Image & rhs)
{
  return lhs.size() == rhs.size();
}

bool operator!=(const Image & lhs, const Image & rhs)
{
  return lhs.size() != rhs.size();
}

}  // namespace npp

#endif  // NPP_IMAGE_HPP_