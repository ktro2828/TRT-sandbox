#include <iostream>

constexpr int BLOCK_SIZE = 16;

struct Matrix
{
  int width;
  int height;
  int stride;
  float * elements;
  __host__ __device__
  Matrix(const int width_, const int height_, const int stride_, bool alloc = false)
  : width(width_), height(height_), stride(stride_)
  {
    if (alloc) {
      elements = reinterpret_cast<float *>(malloc(sizeof(float) * width * height));
    }
  }
  __host__ __device__ Matrix(const int width_, const int height_, bool alloc = false)
  : width(width_), height(height_), stride(width_)
  {
    if (alloc) {
      elements = reinterpret_cast<float *>(malloc(sizeof(float) * width * height));
    }
  }
};  // struct Matrix

// --- __device__ ---
__device__ Matrix getSubMatrix(Matrix M, const int row, const int col)
{
  Matrix Msub(BLOCK_SIZE, BLOCK_SIZE, M.stride);
  Msub.elements = &M.elements[M.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
}

__device__ float getElement(const Matrix & M, const int row, const int col)
{
  return M.elements[row * M.stride + col];
}

__device__ void setElement(Matrix & M, const int row, const int col, const float value)
{
  M.elements[row * M.stride + col] = value;
}

// --- __global__ ---
__global__ void mulMatShared(const Matrix A, const Matrix B, Matrix C)
{
  // Block row and column
  const int blockRow = blockIdx.y;
  const int blockCol = blockIdx.x;
  // Each thread block computes one sub-matrix Csub of C
  Matrix Csub = getSubMatrix(C, blockRow, blockCol);

  // Each thread computes one element of Csub by accumulating results into Cvalue
  float Cvalue = 0.0f;
  // Thread row and column within Csub
  const int row = threadIdx.y;
  const int col = threadIdx.x;
  for (int i = 0; i < (A.width / BLOCK_SIZE); ++i) {
    // Get sub-matrices
    Matrix Asub = getSubMatrix(A, blockRow, i);
    Matrix Bsub = getSubMatrix(B, i, blockCol);

    // Shared memory used to store Asub and Bsub respectively
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    As[row][col] = getElement(Asub, row, col);
    Bs[row][col] = getElement(Bsub, row, col);

    // Synchronize to make sure the sub-matrices are loaded before starting the computation
    __syncthreads();

    // Multiply Asub and Bsub together
    for (int j = 0; j < (B.width / BLOCK_SIZE); ++j) {
      Cvalue += As[row][j] * Bs[j][col];
    }
    // Synchronize to make sure that the preceding computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }
  // Write Csub to device memory
  // Each thread writes one element
  setElement(Csub, row, col, Cvalue);
}

__host__ void setElement(Matrix & M, float value)
{
  for (int i = 0; i < M.height; ++i) {
    for (int j = 0; j < M.width; ++j) {
      M.elements[i * j] = value;
    }
  }
}

template <typename... Matrices>
__host__ void freeDevice(Matrices... mats)
{
  for (auto m : std::initializer_list<Matrix>{mats...}) {
    cudaFree(m.elements);
  }
}

template <typename... Matrices>
__host__ void freeHost(Matrices... mats)
{
  for (auto m : std::initializer_list<Matrix>{mats...}) {
    free(m.elements);
  }
}

__host__ void printElement(const Matrix & M)
{
  std::cout << "(";
  for (int i = 0; i < M.height; ++i) {
    std::cout << "(";
    for (int j = 0; j < M.width; ++j) {
      std::cout << M.elements[i * j];
      if (j != M.width - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "),\n";
  }
  std::cout << ")" << std::endl;
}

int main()
{
  const int Aw = 10, Bh = 10;
  const int Ah = 10, Bw = 10;

  Matrix A(Aw, Ah, true), B(Bw, Bh, true), C(Aw, Bh, true);
  setElement(A, 1.0f);
  setElement(B, 2.0f);

  // Load A and B to device memory
  Matrix d_A(Aw, Ah);
  Matrix d_B(Bw, Bh);
  Matrix d_C(Aw, Bh);

  size_t nbytes_a = sizeof(float) * Aw * Ah;
  size_t nbytes_b = sizeof(float) * Bw * Bh;
  size_t nbytes_c = sizeof(float) * Aw * Bh;
  cudaMalloc(reinterpret_cast<void **>(&d_A.elements), nbytes_a);
  cudaMalloc(reinterpret_cast<void **>(&d_B.elements), nbytes_b);
  cudaMalloc(reinterpret_cast<void **>(&d_C.elements), nbytes_c);

  cudaMemcpy(d_A.elements, A.elements, nbytes_a, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B.elements, B.elements, nbytes_b, cudaMemcpyHostToDevice);
  cudaMemcpy(d_C.elements, C.elements, nbytes_c, cudaMemcpyHostToDevice);

  // Invoke kernel
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
  mulMatShared<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

  // Read C from device memory
  cudaMemcpy(C.elements, d_C.elements, nbytes_c, cudaMemcpyDeviceToHost);
  printElement(C);

  // Free device memory
  freeDevice(d_A, d_B, d_C);
  freeHost(A, B, C);
}