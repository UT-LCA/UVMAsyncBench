
#include "needle.h"
#include <stdio.h>

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

using namespace nvcuda::experimental;

#define PREFETCH_COUNT 2

#define SDATA(index) CUT_BANK_CHECKER(sdata, index)

__device__ __host__ int
maximum(int a,
        int b,
        int c)
{

  int k;
  if (a <= b)
    k = b;
  else
    k = a;

  if (k <= c)
    return (c);
  else
    return (k);
}

__global__ void
needle_cuda_shared_1(int *referrence,
                     int *matrix_cuda,
                     int cols,
                     int penalty,
                     int i,
                     int block_width,
                     int block_size)
{
  cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
  pipeline pipe;
  int bx = blockIdx.x;
  int tx = threadIdx.x;

  int b_index_x = bx;
  int b_index_y = i - 1 - bx;

  __shared__ int temp[PREFETCH_COUNT * (BLOCK_SIZE + 1)][BLOCK_SIZE + 1];
  __shared__ int ref[PREFETCH_COUNT * BLOCK_SIZE][BLOCK_SIZE];

  int tile_dim_x = cols / BLOCK_SIZE;

  int total_tiles = tile_dim_x * tile_dim_x;
  int tiles_this_block = (block_size / BLOCK_SIZE) * (block_size / BLOCK_SIZE);
  int tiles_this_block_x = (block_size / BLOCK_SIZE);

  int base_tile = (b_index_y * gridDim.x + b_index_x) * tiles_this_block;
  int fetch = base_tile;
  int end_tile = fetch + tiles_this_block;

  for (int compute = fetch; compute < end_tile; compute++)
  {
    for (; fetch < end_tile && fetch < compute + PREFETCH_COUNT; fetch++)
    {
      int offset = fetch - base_tile;
      int block_id = fetch / tiles_this_block;
      int b_index_x = block_id % gridDim.x * tiles_this_block_x + offset % tiles_this_block_x;
      int b_index_y = block_id / gridDim.x * tiles_this_block_x + offset / tiles_this_block_x;

      int index = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + (cols + 1);
      int index_n = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + (1);
      int index_w = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + (cols);
      int index_nw = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x;

      if (tx == 0)
        memcpy_async(temp[(fetch % PREFETCH_COUNT) * (BLOCK_SIZE + 1) + tx][0], matrix_cuda[index_nw], pipe);

      for (int ty = 0; ty < BLOCK_SIZE; ty++)
        memcpy_async(ref[(fetch % PREFETCH_COUNT) * (BLOCK_SIZE) + ty][tx], referrence[index + cols * ty], pipe);
      block.sync();

      memcpy_async(temp[(fetch % PREFETCH_COUNT) * (BLOCK_SIZE + 1) + tx + 1][0], matrix_cuda[index_w + cols * tx], pipe);
      block.sync();

      memcpy_async(temp[(fetch % PREFETCH_COUNT) * (BLOCK_SIZE + 1) + 0][tx + 1], matrix_cuda[index_n], pipe);
      pipe.commit();
    }
    if (fetch == end_tile)
    {
      for (int i = 0; i < PREFETCH_COUNT - 1; ++i)
      {
        pipe.commit();
      }
      ++fetch;
    }
    pipe.wait_prior<PREFETCH_COUNT - 1>();
    block.sync();

    int b_index_x = compute % tile_dim_x;
    int b_index_y = compute / tile_dim_x;

    int index = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + (cols + 1);

    for (int m = 0; m < BLOCK_SIZE; m++)
    {
      if (tx <= m)
      {
        int t_index_x = tx + 1;
        int t_index_y = (compute % PREFETCH_COUNT) * (BLOCK_SIZE + 1) + m - tx + 1;

        temp[t_index_y][t_index_x] = maximum(temp[t_index_y - 1][t_index_x - 1] + ref[(compute % PREFETCH_COUNT) * BLOCK_SIZE + m - tx][t_index_x - 1],
                                             temp[t_index_y][t_index_x - 1] - penalty,
                                             temp[t_index_y - 1][t_index_x] - penalty);
      }
      block.sync();
    }

    for (int m = BLOCK_SIZE - 2; m >= 0; m--)
    {
      if (tx <= m)
      {
        int t_index_x = tx + BLOCK_SIZE - m;
        int t_index_y = (compute % PREFETCH_COUNT) * (BLOCK_SIZE + 1) + BLOCK_SIZE - tx;
        temp[t_index_y][t_index_x] = maximum(temp[t_index_y - 1][t_index_x - 1] + ref[(compute % PREFETCH_COUNT) * BLOCK_SIZE + m - tx][t_index_x - 1],
                                             temp[t_index_y][t_index_x - 1] - penalty,
                                             temp[t_index_y - 1][t_index_x] - penalty);
      }
      block.sync();
    }

    for (int ty = 0; ty < BLOCK_SIZE; ty++)
      matrix_cuda[index + ty * cols] = temp[(compute % PREFETCH_COUNT) * (BLOCK_SIZE + 1) + ty + 1][tx + 1];
  }
}

__global__ void
needle_cuda_shared_2(int *referrence,
                     int *matrix_cuda,
                     int cols,
                     int penalty,
                     int i,
                     int block_width,
                     int block_size)
{
  cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
  pipeline pipe;
  int bx = blockIdx.x;
  int tx = threadIdx.x;

  int b_index_x = bx + block_width - i;
  int b_index_y = block_width - bx - 1;

  __shared__ int temp[PREFETCH_COUNT * (BLOCK_SIZE + 1)][BLOCK_SIZE + 1];
  __shared__ int ref[PREFETCH_COUNT * BLOCK_SIZE][BLOCK_SIZE];

  int tile_dim_x = cols / BLOCK_SIZE;

  int total_tiles = tile_dim_x * tile_dim_x;
  int tiles_this_block = (block_size / BLOCK_SIZE) * (block_size / BLOCK_SIZE);
  int tiles_this_block_x = (block_size / BLOCK_SIZE);

  int base_tile = (b_index_y * gridDim.x + b_index_x) * tiles_this_block;
  int fetch = base_tile;
  int end_tile = fetch + tiles_this_block;

  for (int compute = fetch; compute < end_tile; compute++)
  {
    for (; fetch < end_tile && fetch < compute + PREFETCH_COUNT; fetch++)
    {
      int offset = fetch - base_tile;
      int block_id = fetch / tiles_this_block;
      int b_index_x = block_id % gridDim.x * tiles_this_block_x + offset % tiles_this_block_x;
      int b_index_y = block_id / gridDim.x * tiles_this_block_x + offset / tiles_this_block_x;

      int index = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + (cols + 1);
      int index_n = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + (1);
      int index_w = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + (cols);
      int index_nw = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x;

      for (int ty = 0; ty < BLOCK_SIZE; ty++)
        ref[(fetch % PREFETCH_COUNT) * BLOCK_SIZE + ty][tx] = referrence[index + cols * ty];
      block.sync();

      if (tx == 0)
        temp[(fetch % PREFETCH_COUNT) * (BLOCK_SIZE + 1) + tx][0] = matrix_cuda[index_nw];
      temp[(fetch % PREFETCH_COUNT) * (BLOCK_SIZE + 1) + tx + 1][0] = matrix_cuda[index_w + cols * tx];
      block.sync();

      temp[(fetch % PREFETCH_COUNT) * (BLOCK_SIZE + 1) + 0][tx + 1] = matrix_cuda[index_n];
      block.sync();
    }
    if (fetch == end_tile)
    {
      for (int i = 0; i < PREFETCH_COUNT - 1; ++i)
      {
        pipe.commit();
      }
      ++fetch;
    }
    pipe.wait_prior<PREFETCH_COUNT - 1>();
    block.sync();

    int b_index_x = compute % tile_dim_x;
    int b_index_y = compute / tile_dim_x;

    int index = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + (cols + 1);

    for (int m = 0; m < BLOCK_SIZE; m++)
    {
      if (tx <= m)
      {
        int t_index_x = tx + 1;
        int t_index_y = (compute % PREFETCH_COUNT) * (BLOCK_SIZE + 1) + m - tx + 1;
        temp[t_index_y][t_index_x] = maximum(temp[t_index_y - 1][t_index_x - 1] + ref[(compute % PREFETCH_COUNT) * BLOCK_SIZE + m - tx][t_index_x - 1],
                                             temp[t_index_y][t_index_x - 1] - penalty,
                                             temp[t_index_y - 1][t_index_x] - penalty);
      }
      block.sync();
    }

    for (int m = BLOCK_SIZE - 2; m >= 0; m--)
    {
      if (tx <= m)
      {
        int t_index_x = tx + BLOCK_SIZE - m;
        int t_index_y = (compute % PREFETCH_COUNT) * (BLOCK_SIZE + 1) + BLOCK_SIZE - tx;

        temp[t_index_y][t_index_x] = maximum(temp[t_index_y - 1][t_index_x - 1] + ref[(compute % PREFETCH_COUNT) * BLOCK_SIZE + BLOCK_SIZE - tx - 1][t_index_x - 1],
                                             temp[t_index_y][t_index_x - 1] - penalty,
                                             temp[t_index_y - 1][t_index_x] - penalty);
      }
      block.sync();
    }

    for (int ty = 0; ty < BLOCK_SIZE; ty++)
      matrix_cuda[index + ty * cols] = temp[(compute % PREFETCH_COUNT) * (BLOCK_SIZE + 1) + ty + 1][tx + 1];
  }
}
