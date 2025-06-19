/** Modifed version of knn-CUDA from https://github.com/vincentfpgarcia/kNN-CUDA
 * The modifications are
 *      removed texture memory usage
 *      removed split query KNN computation
 *      added feature extraction with bilinear interpolation
 *
 * Last modified by Christopher B. Choy <chrischoy@ai.stanford.edu> 12/23/2016
 */

// Includes
#include "cuda.h"
#include <cstdio>
#include <sys/time.h>
#include <time.h>
// Constants used by the program
#define BLOCK_DIM 16

#include "../../../common/cupti_add.h"
#include "../../../common/cpu_timestamps.h"

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

using namespace nvcuda::experimental;

#define PREFETCH_COUNT 2

#define DIM_THREAD_BLOCK 64

#ifndef SIZE
#define SIZE 4096
#endif

void initCPU(float *a, uint64_t N)
{
  srand((unsigned)2);
  // fill the arrays 'a' on the CPU
  for (uint64_t i = 0; i < (N * N); i++)
  {
    a[i] = ((rand() % 10) + 1);
  }
}

void initGPU(float *a_dev, uint64_t N)
{
  srand((unsigned)2);
  for (uint64_t i = 0; i < (N * N); i++)
  {
    a_dev[i] = ((rand() % 10) + 1);
  }
}

// __global__ void lud_kernel(float *a, uint64_t N)
// {
//   cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
//   pipeline pipe;
//   // extern __shared__ float pivot[];
//   __shared__ float pivot[SIZE * PREFETCH_COUNT];

//   uint64_t fetch = 0;
//   uint64_t end_tile = fetch + N;

//   for (uint64_t compute = fetch; compute < end_tile; compute++)
// 	{
// 		for (; fetch < end_tile && fetch < compute + PREFETCH_COUNT; fetch++) 
// 		{
//       uint64_t tid = threadIdx.x;
//       uint64_t bid = blockIdx.x;
//       uint64_t block_size = blockDim.x;

//       if (tid == 0 && bid == 0)
//       {
//         uint64_t start = (compute * N + compute);
//         uint64_t end = (compute * N + N);

//         for (uint64_t i = start + 1; i < end; i++)
//           a[i] = (a[i] / a[start]);
//       }
//       block.sync();

//       if (tid == 0)
//       {
//         for (uint64_t i = compute; i < N; i++)
//           memcpy_async(pivot[(fetch % PREFETCH_COUNT) * N + i], a[(compute * N) + i], pipe);
//       }
//       // block.sync();
//       pipe.commit();
//     }
//     if (fetch == end_tile)
//     {
//       for (uint64_t i = 0; i < PREFETCH_COUNT - 1; ++i)
//       {
//         pipe.commit();
//       }
//       ++fetch;
//     }
//     pipe.wait_prior<PREFETCH_COUNT - 1>();
//     block.sync();

//     uint64_t tid = threadIdx.x;
//     uint64_t bid = blockIdx.x;
//     uint64_t block_size = blockDim.x;

//     uint64_t pivot_row = (compute * N);
//     uint64_t my_row = (((block_size * bid) + tid) * N);
//     uint64_t start = my_row + compute;
//     uint64_t end = my_row + N;

//     if (my_row > pivot_row)
//     {
//       for (uint64_t i = start + 1; i < end; i++)
//       {
//         a[i] = a[i] - (a[start] * pivot[(compute % PREFETCH_COUNT) * N + (i - my_row)]);
//       }
//     }
//     block.sync();
//   }
// }

__global__ void lud_kernel(float *a, uint64_t N)
{
  cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
  pipeline pipe;
  // extern __shared__ float pivot[];
  __shared__ float pivot[SIZE * PREFETCH_COUNT];

  uint64_t batches = N / SIZE;
  for (uint64_t b = 0; b < batches; b++) {
    uint64_t tile_start = b * SIZE;
    uint64_t tile_end = (b + 1) * SIZE;

    uint64_t fetch = tile_start;
    uint64_t end_tile = fetch + SIZE;

    for (uint64_t compute = fetch; compute < end_tile; compute++)
    {
      for (; fetch < end_tile && fetch < compute + PREFETCH_COUNT; fetch++) 
      {
        uint64_t tid = threadIdx.x;
        uint64_t bid = blockIdx.x;
        uint64_t block_size = blockDim.x;

        if (tid == 0 && bid == 0)
        {
          uint64_t start = (compute * N + compute);
          uint64_t end = (compute * N + end_tile);

          for (uint64_t i = start + 1; i < end; i++)
            a[i] = (a[i] / a[start]);
        }
        block.sync();

        if (tid == 0)
        {
          for (uint64_t i = compute; i < end_tile; i++)
            memcpy_async(pivot[(fetch % PREFETCH_COUNT) * SIZE + i % SIZE], a[(compute * N) + i], pipe);
        }
        pipe.commit();
      }
      if (fetch == end_tile)
      {
        for (uint64_t i = 0; i < PREFETCH_COUNT - 1; ++i)
        {
          pipe.commit();
        }
        ++fetch;
      }
      pipe.wait_prior<PREFETCH_COUNT - 1>();
      block.sync();

      uint64_t tid = threadIdx.x;
      uint64_t bid = blockIdx.x;
      uint64_t block_size = blockDim.x;

      uint64_t pivot_row = (compute * N);
      uint64_t my_row = (((block_size * bid) + tid) * N);
      uint64_t start = my_row + compute;
      uint64_t end = my_row + end_tile;

      if (my_row > pivot_row)
      {
        for (uint64_t i = start + 1; i < end; i++)
        {
          a[i] = a[i] - (a[start] * pivot[(compute % PREFETCH_COUNT) * SIZE + (i - my_row) % SIZE]);
        }
      }
      block.sync();
    }
    block.sync();
  }
}

int main(int argc, char *argv[])
{
  uint64_t start_tsc = rdtsc();
  uint64_t start_tsp = rdtsp();
  printf("start_tsc %lu start_tsp %lu\n", start_tsc, start_tsp);
  float *a;
  float *c;
  float error;
  uint64_t N;
  uint64_t flag = 0;

  float **result;
  float **a_ref;
  uint64_t blocks;

  float *dev_a;
  uint64_t i;
  uint64_t j;
  uint64_t k;
  float l1;
  float u1;

  N = atoi(argv[1]);
  // allocate memory on CPU
  // a = (float *)malloc(sizeof(float) * N * N);
  // c = (float *)malloc(sizeof(float) * N * N);
  cudaMallocHost((void **)&a, N * N * sizeof(float));
  cudaMallocHost((void **)&c, N * N * sizeof(float));

  result = (float **)malloc(N *sizeof(float *));
  a_ref = (float **)malloc(N * sizeof(float *));

  for (i = 0; i < N; i++)
  {
    result[i] = (float *)malloc(N * sizeof(float));
    a_ref[i] = (float *)malloc(N * sizeof(float));
  }
  initCPU(a, N);
  
  GPU_argv_init();
  initTrace();
  startCPU();
  // allocate the memory on the GPU
  cudaMalloc((void **)&dev_a, N * N * sizeof(float));

  cudaMemcpy(dev_a, a, N * N * sizeof(float), cudaMemcpyHostToDevice); // copy array to device memory

  blocks = ((N / DIM_THREAD_BLOCK));
  // lud_kernel<<<blocks, DIM_THREAD_BLOCK, N * sizeof(float) * PREFETCH_COUNT>>>(dev_a, N);
  lud_kernel<<<blocks, DIM_THREAD_BLOCK>>>(dev_a, N);
  cudaDeviceSynchronize();
  /*LU decomposition ends here*/

  cudaMemcpy(c, dev_a, N * N * sizeof(float), cudaMemcpyDeviceToHost); // copy array back to host

  // free the memory allocated on the GPU
  cudaFree(dev_a);
  endCPU();
  finiTrace();

  /*copy the result matrix into explicit 2D matrix for verification*/
  for (i = 0; i < N; i++)
  {
    for (j = 0; j < N; j++)
    {
      result[i][j] = c[i * N + j];
    }
  }

  printf("=======================================================");
  printf("\n Performing inplace verification \n");

  for (i = 0; i < N; i++)
  {
    for (j = 0; j < N; j++)
    {
      a_ref[i][j] = 0;
      for (k = 0; k < N; k++)
      {
        if (i >= k)
          l1 = result[i][k];
        else
          l1 = 0;

        if (k == j)
          u1 = 1;
        else if (k < j)
          u1 = result[k][j]; // figured it out
        else
          u1 = 0.0;

        a_ref[i][j] = a_ref[i][j] + (l1 * u1);
      }
    }
  }

  return 0;
}
