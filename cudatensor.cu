#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
using namespace std;

__global__ void
cuda_saxpy(float alpha, float* x, float*y, size_t size) {
  size_t i = threadIdx.x;
  if (i < size) y[i] += alpha * x[i];
}

__global__ void
cuda_sscal(float alpha, float* x, size_t size) {
  size_t i = threadIdx.x;
  if (i < size) x[i] *= alpha;
}

template <typename Dtype>
class CudaTensor {
public:
  Dtype* ddata = nullptr; // data on device
  size_t shape = 0;

  CudaTensor(size_t length) {
    shape = length;
    cudaMalloc((void**)&ddata, sizeof(Dtype)*length);
  }

  void take(Dtype* src, size_t sz) {
    cudaMemcpy(ddata, src, sizeof(Dtype)*sz,
      cudaMemcpyHostToDevice);
  }

  void give(Dtype* dest, size_t sz) {
    cudaMemcpy(dest, ddata, sizeof(Dtype)*sz,
      cudaMemcpyDeviceToHost);
  }

  void dump(void) {
    Dtype* data = (Dtype*)malloc(sizeof(Dtype)*shape);
    give(data, shape);
    for (size_t i = 0; i < shape; i++)
      cout << data[i];
    free(data);
  }

  void scal_(Dtype c) {
    cuda_sscal<<<1,10>>>(c, ddata, shape);
  }

};

int
main(void)
{
  float* a_h = (float*)malloc(sizeof(float)*10);
  for (size_t i = 0; i < 10; i++) a_h[i] = 2.;
  float* b_h = (float*)malloc(sizeof(float)*10);

  for (size_t i = 0; i < 10; i++) cout << a_h[i] << endl;
  cout << endl;

  float* a_d = nullptr;
  cudaMalloc((void**)&a_d, sizeof(float)*10);
  cudaMemcpy(a_d, a_h, sizeof(float)*10, cudaMemcpyHostToDevice);
  
  cuda_saxpy<<<1, 256>>>(1., a_d, a_d, 10);

  cudaMemcpy(b_h, a_d, sizeof(float)*10, cudaMemcpyDeviceToHost);

  
  for (size_t i = 0; i < 10; i++) cout << b_h[i] << endl;
  cout << endl;

  cudaFree(a_d);
  free(a_h); free(b_h);

  cout << "cudatensor" << endl;

  CudaTensor<float> x (10);
  x.dump();
  x.scal_(0.5);
  x.dump();

  return 0;
}
