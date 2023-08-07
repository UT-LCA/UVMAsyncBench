#include <cuda.h>
#include <stdio.h>

#ifdef RD_WG_SIZE_0_0
#define BLOCK_SIZE RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
#define BLOCK_SIZE RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
#define BLOCK_SIZE RD_WG_SIZE
#else
#define BLOCK_SIZE 16
#endif

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

using namespace nvcuda::experimental;

#define PREFETCH_COUNT 2

__global__ void
lud_diagonal(float *m, int matrix_dim, int offset, int batch_size)
{
  cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
  int i, j;
  __shared__ float shadow[BLOCK_SIZE][BLOCK_SIZE];

  for (int b = 0; b < batch_size; b++)
  {
    int bx = b + blockIdx.x;
    int by = b + blockIdx.y;
    int tmp_offset = offset + b * BLOCK_SIZE;

    // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x < BLOCK_SIZE && threadIdx.y == 0)

    int array_offset = tmp_offset * matrix_dim + tmp_offset;
    // int array_offset = (tmp_offset + b * BLOCK_SIZE) * matrix_dim + tmp_offset + b * BLOCK_SIZE;
    for (i = 0; i < BLOCK_SIZE; i++)
    {
      shadow[i][threadIdx.x] = m[array_offset + threadIdx.x];
      array_offset += matrix_dim;
    }
    block.sync();
    for (i = 0; i < BLOCK_SIZE - 1; i++)
    {
      if (threadIdx.x > i)
      {
        for (j = 0; j < i; j++)
          shadow[threadIdx.x][i] -= shadow[threadIdx.x][j] * shadow[j][i];
        shadow[threadIdx.x][i] /= shadow[i][i];
      }

      block.sync();
      if (threadIdx.x > i)
      {
        for (j = 0; j < i + 1; j++)
          shadow[i + 1][threadIdx.x] -= shadow[i + 1][j] * shadow[j][threadIdx.x];
      }
      block.sync();
    }

    /*
       The first row is not modified, it
       is no need to write it back to the
       global memory
     */
    array_offset = (tmp_offset + 1) * matrix_dim + tmp_offset;
    // array_offset = (tmp_offset + by * BLOCK_SIZE + 1) * matrix_dim + tmp_offset + bx * BLOCK_SIZE;
    for (i = 1; i < BLOCK_SIZE; i++)
    {
      m[array_offset + threadIdx.x] = shadow[i][threadIdx.x];
      array_offset += matrix_dim;
    }
    block.sync();
  }
}

__global__ void
lud_perimeter(float *m, int matrix_dim, int offset, int batch_size)
{
  cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
  __shared__ float dia[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float peri_row[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float peri_col[BLOCK_SIZE][BLOCK_SIZE];

  int i, j, array_offset;
  int idx;

  for (int b = 0; b < batch_size; b++)
  {
    // int bx = b % batch_size_x;
    // int by = b / batch_size_x;
    int bx = b + blockIdx.x;
    if (bx < gridDim.x)
    {
      int tmp_offset = offset + b * BLOCK_SIZE;
      if (threadIdx.x < BLOCK_SIZE)
      {
        idx = threadIdx.x;

        array_offset = tmp_offset * matrix_dim + tmp_offset;
        // array_offset = (tmp_offset + b * BLOCK_SIZE) * matrix_dim + tmp_offset + b * BLOCK_SIZE;
        for (i = 0; i < BLOCK_SIZE / 2; i++)
        {
          dia[i][idx] = m[array_offset + idx];
          array_offset += matrix_dim;
        }

        array_offset = tmp_offset * matrix_dim + tmp_offset;
        // array_offset = (tmp_offset + b * BLOCK_SIZE) * matrix_dim + tmp_offset + b * BLOCK_SIZE;
        for (i = 0; i < BLOCK_SIZE; i++)
        {
          peri_row[i][idx] = m[array_offset + (blockIdx.x + 1) * BLOCK_SIZE + idx];
          array_offset += matrix_dim;
        }
      }
      else
      {
        idx = threadIdx.x - BLOCK_SIZE;

        array_offset = (tmp_offset + BLOCK_SIZE / 2) * matrix_dim + tmp_offset;
        // array_offset = (tmp_offset + by * BLOCK_SIZE + BLOCK_SIZE / 2) * matrix_dim + tmp_offset + bx * BLOCK_SIZE;

        for (i = BLOCK_SIZE / 2; i < BLOCK_SIZE; i++)
        {
          dia[i][idx] = m[array_offset + idx];
          array_offset += matrix_dim;
        }

        array_offset = (tmp_offset + (blockIdx.x + 1) * BLOCK_SIZE) * matrix_dim + tmp_offset;
        // array_offset = (tmp_offset + by * BLOCK_SIZE + (blockIdx.x + 1) * BLOCK_SIZE * batch_size_x) * matrix_dim + tmp_offset + bx * BLOCK_SIZE;
        for (i = 0; i < BLOCK_SIZE; i++)
        {
          peri_col[i][idx] = m[array_offset + idx];
          array_offset += matrix_dim;
        }
      }
      block.sync();

      if (threadIdx.x < BLOCK_SIZE)
      { // peri-row
        idx = threadIdx.x;
        for (i = 1; i < BLOCK_SIZE; i++)
        {
          for (j = 0; j < i; j++)
            peri_row[i][idx] -= dia[i][j] * peri_row[j][idx];
        }
      }
      else
      { // peri-col
        idx = threadIdx.x - BLOCK_SIZE;
        for (i = 0; i < BLOCK_SIZE; i++)
        {
          for (j = 0; j < i; j++)
            peri_col[idx][i] -= peri_col[idx][j] * dia[j][i];
          peri_col[idx][i] /= dia[i][i];
        }
      }
      block.sync();

      if (threadIdx.x < BLOCK_SIZE)
      { // peri-row
        idx = threadIdx.x;
        array_offset = (tmp_offset + 1) * matrix_dim + tmp_offset;
        // array_offset = (tmp_offset + by * BLOCK_SIZE + 1) * matrix_dim + tmp_offset + bx * BLOCK_SIZE;
        for (i = 1; i < BLOCK_SIZE; i++)
        {
          m[array_offset + (blockIdx.x + 1) * BLOCK_SIZE + idx] = peri_row[i][idx];
          array_offset += matrix_dim;
        }
      }
      else
      { // peri-col
        idx = threadIdx.x - BLOCK_SIZE;
        array_offset = (tmp_offset + (blockIdx.x + 1) * BLOCK_SIZE) * matrix_dim + tmp_offset;
        // array_offset = (tmp_offset + by * BLOCK_SIZE + (blockIdx.x + 1) * BLOCK_SIZE * batch_size_x) * matrix_dim + tmp_offset + bx * BLOCK_SIZE;
        for (i = 0; i < BLOCK_SIZE; i++)
        {
          m[array_offset + idx] = peri_col[i][idx];
          array_offset += matrix_dim;
        }
      }
    }
  }
}

__global__ void
lud_internal(float *m, int matrix_dim, int offset, int batch_size)
{
  cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
  __shared__ float peri_row[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float peri_col[BLOCK_SIZE][BLOCK_SIZE];

  int i;
  float sum;

  for (int b = 0; b < batch_size; b++)
  {
    int bx = b + blockIdx.x;
    int by = b + blockIdx.y;
    if (bx < gridDim.x && by < gridDim.y)
    {
      int tmp_offset = offset + b * BLOCK_SIZE;
      int global_row_id = tmp_offset + (blockIdx.y + 1) * BLOCK_SIZE;
      int global_col_id = tmp_offset + (blockIdx.x + 1) * BLOCK_SIZE;

      peri_row[threadIdx.y][threadIdx.x] = m[(tmp_offset + threadIdx.y) * matrix_dim + global_col_id + threadIdx.x];
      peri_col[threadIdx.y][threadIdx.x] = m[(global_row_id + threadIdx.y) * matrix_dim + tmp_offset + threadIdx.x];

      block.sync();

      sum = 0;
      for (i = 0; i < BLOCK_SIZE; i++)
        sum += peri_col[threadIdx.y][i] * peri_row[i][threadIdx.x];
      m[(global_row_id + threadIdx.y) * matrix_dim + global_col_id + threadIdx.x] -= sum;
    }
  }
}

__global__ void
lud_all(float *m, int matrix_dim, int offset, int batch_size)
{
  cooperative_groups::thread_block block = cooperative_groups::this_thread_block();

  __shared__ float shadow[BLOCK_SIZE][BLOCK_SIZE];

  __shared__ float dia[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float peri_row[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float peri_col[BLOCK_SIZE][BLOCK_SIZE];

  __shared__ float peri_row_internal[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float peri_col_internal[BLOCK_SIZE][BLOCK_SIZE];

  int i, j, idx, array_offset;
  float sum;

  // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0)
  //   printf("gridDim.x is %d, gridDim.y is %d, blockDim.x is %d, blockDim.y is %d\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y);

  // if (threadIdx.x < BLOCK_SIZE)
  // {
  //   dia[threadIdx.x][threadIdx.y] = 0.0f;
  //   peri_row[threadIdx.x][threadIdx.y] = 0.0f;
  //   peri_col[threadIdx.x][threadIdx.y] = 0.0f;
  //   peri_row_internal[threadIdx.x][threadIdx.y] = 0.0f;
  //   peri_col_internal[threadIdx.x][threadIdx.y] = 0.0f;
  // }
  block.sync();

  for (int b = 0; b < batch_size; b++)
  {
    int bx = b + blockIdx.x;
    int by = b + blockIdx.y;
    int tmp_offset = offset + b * BLOCK_SIZE;

    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.y == 0)
    {
      array_offset = tmp_offset * matrix_dim + tmp_offset;
      for (i = 0; i < BLOCK_SIZE; i++)
      {
        shadow[i][threadIdx.x] = m[array_offset + threadIdx.x];
        array_offset += matrix_dim;
      }
      block.sync();

      for (i = 0; i < BLOCK_SIZE - 1; i++)
      {
        if (threadIdx.x > i)
        {
          for (j = 0; j < i; j++)
            shadow[threadIdx.x][i] -= shadow[threadIdx.x][j] * shadow[j][i];
          shadow[threadIdx.x][i] /= shadow[i][i];
        }

        block.sync();
        if (threadIdx.x > i)
        {
          for (j = 0; j < i + 1; j++)
            shadow[i + 1][threadIdx.x] -= shadow[i + 1][j] * shadow[j][threadIdx.x];
        }
        block.sync();
      }
    }
    block.sync();

    /*
       The first row is not modified, it
       is no need to write it back to the
       global memory
     */
    array_offset = (tmp_offset + 1) * matrix_dim + tmp_offset;
    for (i = 1; i < BLOCK_SIZE; i++)
    {
      m[array_offset + threadIdx.x] = shadow[i][threadIdx.x];
      array_offset += matrix_dim;
    }
    block.sync();

    // lud_perimeter
    if (bx < gridDim.x && blockIdx.y == 0 && threadIdx.y == 0)
    // if (bx < gridDim.x)
    {
      idx = threadIdx.x;

      array_offset = tmp_offset * matrix_dim + tmp_offset;
      for (i = 0; i < BLOCK_SIZE / 2; i++)
      {
        dia[i][idx] = m[array_offset + idx];
        array_offset += matrix_dim;
      }
      block.sync();

      array_offset = tmp_offset * matrix_dim + tmp_offset;
      for (i = 0; i < BLOCK_SIZE; i++)
      {
        peri_row[i][idx] = m[array_offset + (blockIdx.x + 1) * BLOCK_SIZE + idx];
        array_offset += matrix_dim;
      }
      block.sync();

      // idx = threadIdx.x-BLOCK_SIZE;
      array_offset = (tmp_offset + BLOCK_SIZE / 2) * matrix_dim + tmp_offset;
      for (i = BLOCK_SIZE / 2; i < BLOCK_SIZE; i++)
      {
        dia[i][idx] = m[array_offset + idx];
        array_offset += matrix_dim;
      }
      block.sync();

      array_offset = (tmp_offset + (blockIdx.x + 1) * BLOCK_SIZE) * matrix_dim + tmp_offset;
      for (i = 0; i < BLOCK_SIZE; i++)
      {
        peri_col[i][idx] = m[array_offset + idx];
        array_offset += matrix_dim;
      }
      block.sync();

      // idx = threadIdx.x;
      for (i = 1; i < BLOCK_SIZE; i++)
      {
        for (j = 0; j < i; j++)
          peri_row[i][idx] -= dia[i][j] * peri_row[j][idx];
      }
      block.sync();

      // idx = threadIdx.x-BLOCK_SIZE;
      for (i = 0; i < BLOCK_SIZE; i++)
      {
        for (j = 0; j < i; j++)
          peri_col[idx][i] -= peri_col[idx][j] * dia[j][i];
        peri_col[idx][i] /= dia[i][i];
      }
      block.sync();

      // idx = threadIdx.x;
      array_offset = (tmp_offset + 1) * matrix_dim + tmp_offset;
      for (i = 1; i < BLOCK_SIZE; i++)
      {
        m[array_offset + (blockIdx.x + 1) * BLOCK_SIZE + idx] = peri_row[i][idx];
        array_offset += matrix_dim;
      }
      block.sync();

      // idx = threadIdx.x-BLOCK_SIZE;
      array_offset = (tmp_offset + (blockIdx.x + 1) * BLOCK_SIZE) * matrix_dim + tmp_offset;
      for (i = 0; i < BLOCK_SIZE; i++)
      {
        m[array_offset + idx] = peri_col[i][idx];
        array_offset += matrix_dim;
      }

      block.sync();
    }
    // __syncthreads();

    // lud_internal
    if (bx < gridDim.x && by < gridDim.y)
    {
      int global_row_id = tmp_offset + (blockIdx.y + 1) * BLOCK_SIZE;
      int global_col_id = tmp_offset + (blockIdx.x + 1) * BLOCK_SIZE;

      peri_row_internal[threadIdx.y][threadIdx.x] = m[(tmp_offset + threadIdx.y) * matrix_dim + global_col_id + threadIdx.x];
      peri_col_internal[threadIdx.y][threadIdx.x] = m[(global_row_id + threadIdx.y) * matrix_dim + tmp_offset + threadIdx.x];

      block.sync();

      sum = 0;
      for (i = 0; i < BLOCK_SIZE; i++)
        sum += peri_col_internal[threadIdx.y][i] * peri_row_internal[i][threadIdx.x];
      m[(global_row_id + threadIdx.y) * matrix_dim + global_col_id + threadIdx.x] -= sum;
      block.sync();
    }
    // __syncthreads();
  }
}

void lud_cuda(float *m, int matrix_dim, int batch_size)
{
  int i = 0;
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  float *m_debug = (float *)malloc(matrix_dim * matrix_dim * sizeof(float));

  // printf("BLOCK_SIZE is %d, matrix_dim is %d, batch_size is %d\n", BLOCK_SIZE, matrix_dim, batch_size);

  for (i = 0; i < matrix_dim - BLOCK_SIZE; i += (BLOCK_SIZE * batch_size))
  {
    int griddim = (matrix_dim - i) / BLOCK_SIZE - 1;
    if (i + BLOCK_SIZE * batch_size >= matrix_dim)
      batch_size = batch_size - 1;
    // printf("i is %d, griddim is %d, batch_size is %d\n", i, griddim, batch_size);

    dim3 dimGrid(griddim, griddim);
    dim3 dimBlock_all(BLOCK_SIZE, BLOCK_SIZE);

    lud_diagonal<<<1, BLOCK_SIZE>>>(m, matrix_dim, i, batch_size);
    lud_perimeter<<<griddim, BLOCK_SIZE * 2>>>(m, matrix_dim, i, batch_size);
    lud_internal<<<dimGrid, dimBlock>>>(m, matrix_dim, i, batch_size);

    // lud_all<<<dimGrid, dimBlock_all>>>(m, matrix_dim, i, batch_size);
    // cudaDeviceSynchronize();
  }
  lud_diagonal<<<1, BLOCK_SIZE>>>(m, matrix_dim, i, 1);
  cudaDeviceSynchronize();
}