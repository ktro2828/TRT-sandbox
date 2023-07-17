#ifndef NPP_PIXEL_HPP_
#define NPP_PIXEL_HPP_

#include <cstddef>

namespace npp
{
template <typename D, std::size_t N>
struct Pixel
{
};  // struct Pixel

template <typename D>
struct Pixel<D, 1>
{
  D x;

  const D & operator[](const std::size_t channel)
  {
    NPP_ASSERT(channel < 1);
    return (&x)[channel];
  }

  D & operator[](const std::size_t channel)
  {
    NPP_ASSERT(channel < 1);
    return (&x)[channel];
  }
};  // struct Pixel<D, 1>

template <typename D>
struct Pixel<D, 2>
{
  D x, y;

  const D & operator[](const std::size_t channel)
  {
    NPP_ASSERT(channel < 2);
    return (&x)[channel];
  }

  D & operator[](const std::size_t channel)
  {
    NPP_ASSERT(channel < 2);
    return (&x)[channel];
  }
};  // struct Pixel<D, 2>

template <typename D>
struct Pixel<D, 3>
{
  D x, y, z;

  const D & operator[](const std::size_t channel)
  {
    NPP_ASSERT(channel < 3);
    return (&x)[channel];
  }

  D & operator[](const std::size_t channel)
  {
    NPP_ASSERT(channel < 3);
    return (&x)[channel];
  }
};  // struct Pixel<D, 3>

template <typename D>
struct Pixel<D, 4>
{
  D x, y, z w;

  const D & operator[](const std::size_t channel)
  {
    NPP_ASSERT(channel < 4);
    return (&x)[channel];
  }
  D & operator[](const std::size_t channel)
  {
    NPP_ASSERT(channel < 4);
    return (&x)[channel];
  }
};  // struct Pixel<D, 4>
}  // namespace npp
#endif  // NPP_PIXEL_HPP_