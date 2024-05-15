#include <cub/cub.cuh>

#include <iostream>
#include <vector>

/**
 * cub::DeviceRadixSort::SortPairs(...)
 *
 * Sorts key-value pairs into ascensing order. (~2N auxiliary storage required)
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

int main()
{
  std::vector<float> h_keys{8.0f, 6.0f, 7.0f, 5.0f, 3.0f, 0.0f, 9.0f};
  std::vector<float> h_values{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0, 6.0f};

  std::cout << "[Before]:\n";
  std::cout << "keys: ";
  std::cout << h_keys << std::endl;
  std::cout << "values: ";
  std::cout << h_values << std::endl;

  size_t num_items = h_keys.size();
  float *d_keys_in, *d_keys_out;
  float *d_values_in, *d_values_out;

  cudaMalloc(&d_keys_in, sizeof(float) * num_items);
  cudaMalloc(&d_keys_out, sizeof(float) * num_items);
  cudaMalloc(&d_values_in, sizeof(float) * num_items);
  cudaMalloc(&d_values_out, sizeof(float) * num_items);

  cudaMemcpy(d_keys_in, h_keys.data(), sizeof(float) * num_items, cudaMemcpyHostToDevice);
  cudaMemcpy(d_values_in, h_values.data(), sizeof(float) * num_items, cudaMemcpyHostToDevice);

  // Determinate temporary device storage requirements
  void * d_temp_storage{nullptr};
  size_t temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairs(
    d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out,
    num_items);

  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);

  // Run sorting operation
  cub::DeviceRadixSort::SortPairs(
    d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out,
    num_items);

  std::vector<float> h_keys_out(num_items);
  std::vector<float> h_values_out(num_items);
  cudaMemcpy(h_keys_out.data(), d_keys_out, sizeof(float) * num_items, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_values_out.data(), d_values_out, sizeof(float) * num_items, cudaMemcpyDeviceToHost);

  // keys: [0, 3, 5, 6, 7, 8, 9]
  // values: [5, 4, 3, 1, 2, 0, 6]
  std::cout << "[After]:\n";
  std::cout << "keys: ";
  std::cout << h_keys_out << std::endl;
  std::cout << "values: ";
  std::cout << h_values_out << std::endl;
}