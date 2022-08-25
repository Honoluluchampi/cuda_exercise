#include <cuda_runtime.h>
#include <stdio.h>

__global__ void check_index(void)
{
  printf("thread idx:(%d, %d, %d) block idx:(%d, %d, %d) block dim(%d, %d, %d) grid dim:(%d, %d, %d)\n", 
    threadIdx.x, threadIdx.y, threadIdx.z,
    blockIdx.x, blockIdx.y, blockIdx.z,
    blockDim.x, blockDim.y, blockDim.z,
    gridDim.x, gridDim.y, gridDim.z);
}

int main()
{
  // data count
  int n_elem = 6;
  
  // define a structure of grids and blocks
  dim3 block(3); // consits of three threads
  dim3 grid((n_elem + block.x - 1) / block.x);

  // check the sizes of grids and blocks from cpu
  printf("grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z);
  printf("block.x %d block.y %d block.z %d\n", block.x, block.y, block.z);

  // check from gpu
  check_index<<<grid, block>>> ();

  cudaDeviceReset();
}