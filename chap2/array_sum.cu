// std
#include <stdio.h>
#include <time.h>
// lib
#include <cuda_runtime.h>

// error checking macro
#define CHECK(call)\
{\
  const cudaError_t error = call;\
  if (error != cudaSuccess) {\
    printf("Error:%s:%d, ", __FILE__, __LINE__);\
    printf("code:%d, reason:%s\n", error, cudaGetErrorString(error));\
    exit(1);\
  }\
}

void initial_data(float *ip, int size) {
  // create random seed
  time_t t;
  srand((unsigned int) time(&t));

  for (int i = 0; i < size; i++)
    ip[i] = (float)(rand() & 0xFF) / 10.0f;
  return;
}

void sum_array_on_host(float* a, float* b, float* c, const int n)
{
  for (int i = 0; i < n; i++)
    c[i] = a[i] + b[i];
}

// this function is called by all thread parallely
__global__ void sum_array_on_gpu(float* a, float* b, float* c)
{
  int i = threadIdx.x;
  c[i] = a[i] + b[i];
}

bool check_result(float* host_ref, float* gpu_ref, const int n)
{
  double epsilon = 1.0e-8;
  for (int i = 0; i < n; i++)
    if (abs(host_ref[i] - gpu_ref[i]) > epsilon)
      return false;
  return true;
}

int main() {
  float *data_a, *data_b, *data_c;
  float *host_a, *host_b, *host_ref, *gpu_ref;
  int n_element = 1024;
  size_t n_bytes = n_element * sizeof(float); 
  
  // allocate memory for host
  host_a = (float *)malloc(n_bytes);
  host_b = (float *)malloc(n_bytes);
  host_ref = (float *)malloc(n_bytes);
  gpu_ref = (float *)malloc(n_bytes);

  // allocate memory for cuda device
  cudaMalloc((float**)&data_a, n_bytes);
  cudaMalloc((float**)&data_b, n_bytes);
  cudaMalloc((float**)&data_c, n_bytes);
  
  // create data
  initial_data(host_a, n_element);
  initial_data(host_b, n_element);

  // copy data to the gpu
  cudaMemcpy(data_a, host_a, n_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(data_b, host_b, n_bytes, cudaMemcpyHostToDevice);

  sum_array_on_host(host_a, host_b, host_ref, n_element);
  // execute kernel
  dim3 block(n_element);
  dim3 grid(1);
  sum_array_on_gpu<<<grid, block>>>(data_a, data_b, data_c);
  // copy the result
  cudaMemcpy(gpu_ref, data_c, n_bytes, cudaMemcpyDeviceToHost);

  auto result = check_result(host_ref, gpu_ref, n_element);
  if (result)
    printf("correct answer.\n");
  else
    printf("wrong answer.\n");

  // free host memory
  free(host_a);
  free(host_b);
  free(host_ref);
  free(gpu_ref);

  // free cuda memory
  cudaFree(data_a);
  cudaFree(data_b);
  cudaFree(data_c);
}