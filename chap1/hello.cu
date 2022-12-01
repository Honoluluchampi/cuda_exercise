#include <stdio.h>
#include "../utils/utils.hpp"

__global__ void hello_from_GPU()
{
  printf("Hello World from GPU!\n");
}

int main()
{
  scope_timer timer("hello function");

  printf("Hello World from CPU!\n");
  hello_from_GPU<<<1, 10>>>();
  cudaDeviceReset();
}