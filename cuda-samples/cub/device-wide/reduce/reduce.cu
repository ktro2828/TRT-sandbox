#include <cub/cub.cuh>

#include <iostream>
#include <vector>

/**
 * Computes a device-wide reduction using the specified binary `reduction_op` functor and initial
 * value `init`.
 */

template <typename T>
std::ostream & operator<<(std::ostream & os, const std::vector<T> & v)
{
  os << "[";
  for (const auto & e : v) {
    os << e << ", ";
  }
  os << "]";
  return os;
}

// CustomMin functor
template <typename T>
struct CustomMin
{
  __host__ __forceinline__ T operator()(const T & a, const T & b) const { return (b < a) ? b : a; }
};

int main()
{
  std::vector<float> h_in{8.0f, 6.0f, 7.0f, 5.0f, 3.0f, 0.0f, 9.0f};
  std::cout << "[Before]:\n";
  std::cout << "in: ";
  std::cout << h_in << std::endl;

  size_t num_items = h_in.size();

  float *d_in, *d_out;
  cudaMalloc(&d_in, sizeof(float) * num_items);
  cudaMalloc(&d_out, sizeof(float) * num_items);
  cudaMemcpy(d_in, h_in.data(), sizeof(float) * num_items, cudaMemcpyHostToDevice);

  CustomMin<float> min_op;
  float init = FLT_MAX;

  // Determine temporary device storage requirements
  void * d_temp_storage{nullptr};
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::Reduce(
    d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, min_op, init);

  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);

  // Run reduction
  cub::DeviceReduce::Reduce(
    d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, min_op, init);

  std::vector<float> h_out(num_items);
  cudaMemcpy(h_out.data(), d_out, sizeof(float) * num_items, cudaMemcpyDeviceToHost);

  std::cout << "[Before]:\n";
  std::cout << "out: ";
  std::cout << h_out << std::endl;
}