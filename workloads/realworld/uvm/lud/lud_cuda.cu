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

#define DIM_THREAD_BLOCK 256

#ifndef SIZE
#define SIZE 4096
#endif

__global__ void add(float *a, float *b, float *c)
{
  int tid = blockIdx.x; // Handle the data at the index

  c[tid] = a[tid] + b[tid];
}

__global__ void scale(float *a, int size, int index)
{
  int i;
  int start = (index * size + index);
  int end = (index * size + size);

  for (i = start + 1; i < end; i++)
  {
    a[i] = (a[i] / a[start]);
  }
}

__global__ void reduce(float *a, int size, int index, int b_size)
{
  extern __shared__ float pivot[SIZE];
  int i;

  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int block_size = b_size;

  int pivot_start = (index * size + index);
  int pivot_end = (index * size + size);

  int start;
  int end;
  int pivot_row;
  int my_row;

  if (tid == 0)
  {
    for (i = index; i < size; i++)
      pivot[i] = a[(index * size) + i];
  }

  __syncthreads();

  pivot_row = (index * size);
  my_row = (((block_size * bid) + tid) * size);
  start = my_row + index;
  end = my_row + size;

  if (my_row > pivot_row)
  {
    for (i = start + 1; i < end; i++)
    {
      a[i] = a[i] - (a[start] * pivot[(i - my_row)]);
    }
  }
}

#define GPU_DEVICE 6

void GPU_argv_init()
{
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
  printf("setting device %d with name %s\n", GPU_DEVICE, deviceProp.name);
  cudaSetDevice(GPU_DEVICE);
}

void initCPU(float *a, int N)
{
  srand((unsigned)2);
  // fill the arrays 'a' on the CPU
  for (int i = 0; i < (N * N); i++)
  {
    a[i] = ((rand() % 10) + 1);
    // a[i] = 1.0f;
  }
}

void initGPU(float *a_dev, float *a, int N)
{
  for (int i = 0; i < (N * N); i++)
  {
    a_dev[i] = a[i];
    // a_dev[i] = 1.0f;
  }
}

__global__ void lud_kernel(float *a, int N)
{
  cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
  // extern __shared__ float pivot[];
  __shared__ float pivot[SIZE];

  for (int tile = 0; tile < N; tile += 1) {  
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int block_size = blockDim.x;

    if (tid == 0 && bid == 0) {
      int start = (tile * N + tile);
      int end = (tile * N + N);

      for (int i = start + 1; i < end; i++)
        a[i] = (a[i] / a[start]);
    }
    block.sync();

    if (tid == 0)
    {      
      for (int i = tile; i < N; i++)
        pivot[i] = a[(tile * N) + i];
    }
    block.sync();

    int pivot_row = (tile * N);
    int my_row = (((block_size * bid) + tid) * N);
    int start = my_row + tile;
    int end = my_row + N;

    if (my_row > pivot_row)
    {
      for (int i = start + 1; i < end; i++)
      {
        a[i] = a[i] - (a[start] * pivot[(i - my_row)]);
      }
    }
    block.sync();
  }
}

extern inline __attribute__((always_inline)) unsigned long rdtsc()
{
  unsigned long a, d;

  __asm__ volatile("rdtsc"
                   : "=a"(a), "=d"(d));

  return (a | (d << 32));
}

extern inline __attribute__((always_inline)) unsigned long rdtsp()
{
  struct timespec tms;
  if (clock_gettime(CLOCK_REALTIME, &tms))
  {
    return -1;
  }
  unsigned long ns = tms.tv_sec * 1000000000;
  ns += tms.tv_nsec;
  return ns;
}

int main(int argc, char *argv[])
{
  uint64_t start_tsc = rdtsc();
  uint64_t start_tsp = rdtsp();
  printf("start_tsc %lu start_tsp %lu\n", start_tsc, start_tsp);
  float *a;
  float *a_gpu;
  float *c;
  float error;
  int N;
  int flag = 0;

  float **result;
  float **a_ref;
  int blocks;

  int i;
  int j;
  int k;
  float l1;
  float u1;

  N = SIZE;
  // allocate memory on CPU
  a = (float *)malloc(sizeof(float) * N * N);
  c = (float *)malloc(sizeof(float) * N * N);

  result = (float **)malloc(sizeof(float *) * N);
  a_ref = (float **)malloc(sizeof(float *) * N);

  for (i = 0; i < N; i++)
  {
    result[i] = (float *)malloc(sizeof(float) * N);
    a_ref[i] = (float *)malloc(sizeof(float) * N);
  }
  initCPU(a, N);

  GPU_argv_init();
  initTrace();
  startCPU();
  // allocate the memory on the GPU
  // cudaMalloc((void **)&dev_a, N * N * sizeof(float));

  cudaMallocManaged(&a_gpu, N * N * sizeof(float));
  memcpy(a_gpu, a, N * N * sizeof(float));

  // cudaMemcpy(dev_a, a, N * N * sizeof(float), cudaMemcpyHostToDevice); // copy array to device memory

  /*Perform LU Decomposition*/
  // for (i = 0; i < N; i++)
  // {
  //   scale<<<1, 1>>>(a_gpu, N, i);
  //   // blocks= ((N-i-1)/512)+1;
  //   blocks = ((N / DIM_THREAD_BLOCK));
  //   //	printf("Number of blocks rxd : %d \n",blocks);
  //   reduce<<<blocks, DIM_THREAD_BLOCK>>>(a_gpu, N, i, DIM_THREAD_BLOCK);
  //   cudaDeviceSynchronize();
  // }
  blocks = ((N / DIM_THREAD_BLOCK));
  lud_kernel<<<blocks, DIM_THREAD_BLOCK>>>(a_gpu, N);
  cudaDeviceSynchronize();
  /*LU decomposition ends here*/

  // cudaMemcpy(c, dev_a, N * N * sizeof(float), cudaMemcpyDeviceToHost); // copy array back to host
  memcpy(c, a_gpu, N * N * sizeof(float));
  // free the memory allocated on the GPU
  cudaFree(a_gpu);

  endCPU();
  finiTrace();

  /*copy the result matrix into explicit 2D matrix for verification*/
  for (i = 0; i < N; i++)
  {
    for (j = 0; j < N; j++)
    {
      // c[i * N + j] = a_gpu[i * N + j];
      result[i][j] = c[i * N + j];
      // printf("result %d %d Error is %lf \n ", i, j, result[i][j]);
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

  // for (i = 0; i < N; i++)
  // {
  //   for (j = 0; j < N; j++)
  //   {
  //     error = abs(a[(i * N + j)] - a_ref[i][j]);
  //     if (error > 1)
  //     {
  //       // printf("No match occured at %d %d Error is %lf \n ", i, j, abs(a[(i * N + j)] - a_ref[i][j]));
  //       // printf("No match occured at %d %d Error is %lf, %lf \n ", i, j, a[(i * N + j)], a_ref[i][j]);
  //       flag = flag + 1;
  //     }
  //   }
  // }

  // if (flag == 0)
  //   printf("Match \n");
  // else
  //   printf("No Matchs %d \n", flag);

  

  return 0;
}
