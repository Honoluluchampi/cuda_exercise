#include <stdio.h>

__global__ void hello_from_GPU()
{
  printf("Hello World from GPU!\n");
}

int main()
{
  printf("Hello World from CPU!\n");
  hello_from_GPU<<<1, 10>>>();
  cudaDeviceReset();
}