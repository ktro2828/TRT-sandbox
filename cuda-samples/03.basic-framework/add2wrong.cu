#include <math.h>
#include <stdio.h>

const double EPSILON = 1.0e-15;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;

__global__ void add(const double * x, const double * y, double * z);

void check(const double * z, const int N);

int main(void)
{
  const int N = 100000000;
  const int M = sizeof(double) * N;
  double * h_x = reinterpret_cast<double *>(std::malloc(M));
  double * h_y = reinterpret_cast<double *>(std::malloc(M));
  double * h_z = reinterpret_cast<double *>(std::malloc(M));

  for (int n = 0; n < N; ++n) {
    h_x[n] = a;
    h_y[n] = b;
  }

  double *d_x, *d_y, *d_z;
  cudaMalloc(reinterpret_cast<void **>(&d_x), M);
  cudaMalloc(reinterpret_cast<void **>(&d_y), M);
  cudaMalloc(reinterpret_cast<void **>(&d_z), M);
  cudaMemcpy(d_x, h_x, M, cudaMemcpyDeviceToHost);  // WRONG
  cudaMemcpy(d_y, h_y, M, cudaMemcpyDeviceToHost);  // WRONG

  constexpr int block_size = 128;
  const int grid_size = N / block_size;
  add<<<grid_size, block_size>>>(d_x, d_y, d_z);

  cudaMemcpy(h_z, d_z, M, cudaMemcpyDeviceToHost);
  check(h_z, M);

  free(h_x);
  free(h_y);
  free(h_z);
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);
  return 0;
}

__global__ void add(const double * x, const double * y, double * z)
{
  const int n = blockDim.x * blockIdx.x + threadIdx.x;
  z[n] = x[n] + y[n];
}

void check(const double * z, const int N)
{
  bool has_err = false;
  for (int n = 0; n < N; ++n) {
    if (fabs(z[n] - c) > EPSILON) {
      has_err = true;
    }
  }
  printf("%s\n", has_err ? "Has erros" : "No errors");
}