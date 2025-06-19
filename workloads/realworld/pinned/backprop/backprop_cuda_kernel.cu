

#ifndef _BACKPROP_CUDA_KERNEL_H_
#define _BACKPROP_CUDA_KERNEL_H_

#include <stdio.h>
#include "backprop.h"
#include "math.h"
#include "cuda.h"

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

__global__ void
bpnn_layerforward_CUDA(float *input_cuda,
                       float *output_hidden_cuda,
                       float *input_hidden_cuda,
                       float *hidden_partial_sum,
                       int in,
                       int hid,
                       int tile_size)
{
   cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
   int by = blockIdx.y;
   int tx = threadIdx.x;
   int ty = threadIdx.y;

   int batches = tile_size / WIDTH;

   __shared__ float input_node[HEIGHT];
   __shared__ float weight_matrix[HEIGHT * WIDTH];

   for (int b = 0; b < batches; b++)
   {
      int index = (hid + 1) * HEIGHT * (batches * by + b) + (hid + 1) * ty + tx + 1 + (hid + 1);

      int index_in = HEIGHT * (batches * by + b) + ty + 1;

      if (tx == 0)
         input_node[ty] = input_cuda[index_in];

      block.sync();

      weight_matrix[ty * WIDTH + tx] = input_hidden_cuda[index];

      block.sync();

      weight_matrix[ty * WIDTH + tx] = weight_matrix[ty * WIDTH + tx] * input_node[ty];

      block.sync();

      for (int i = 1; i <= __log2f(HEIGHT); i++)
      {

         int power_two = __powf(2, i);

         if (ty % power_two == 0)
            weight_matrix[ty * WIDTH + tx] += weight_matrix[(ty + power_two / 2) * WIDTH + tx];

         block.sync();
      }

      input_hidden_cuda[index] = weight_matrix[ty * WIDTH + tx];

      block.sync();

      if (tx == 0)
      {
         hidden_partial_sum[(batches * by + b) * hid + ty] = weight_matrix[tx * WIDTH + ty];
      }
   }
}

__global__ void bpnn_adjust_weights_cuda(float *delta,
                                         int hid,
                                         float *ly,
                                         int in,
                                         float *w,
                                         float *oldw,
                                         int tile_size)
{
   cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
   int by = blockIdx.y;

   int tx = threadIdx.x;
   int ty = threadIdx.y;

   int batches = tile_size / WIDTH;

   for (int b = 0; b < batches; b++)
   {
      int index = (hid + 1) * HEIGHT * (batches * by + b) + (hid + 1) * ty + tx + 1 + (hid + 1);
      int index_y = HEIGHT * (batches * by + b) + ty + 1;
      int index_x = tx + 1;
      // eta = 0.3;
      // momentum = 0.3;

      w[index] += ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));
      oldw[index] = ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));

      block.sync();

      if (ty == 0 && by == 0 && b == 0)
      {
         w[index_x] += ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
         oldw[index_x] = ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
      }
   }
}
#endif 
