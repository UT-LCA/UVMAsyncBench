

#ifndef _BACKPROP_CUDA_KERNEL_H_
#define _BACKPROP_CUDA_KERNEL_H_

#include <stdio.h>
#include "backprop.h"
#include "math.h"
#include "cuda.h"

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

using namespace nvcuda::experimental;

#define PREFETCH_COUNT 2

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
   pipeline pipe;

   int by = blockIdx.y;
   int tx = threadIdx.x;
   int ty = threadIdx.y;

   __shared__ float input_node[HEIGHT * PREFETCH_COUNT];
   __shared__ float weight_matrix[HEIGHT * WIDTH * PREFETCH_COUNT];

   int batches = tile_size / WIDTH;

   int fetch = batches * by;
   int end_tile = fetch + batches;

   for (int compute = fetch; compute < end_tile; compute++)
   {
      for (; fetch < end_tile && fetch < compute + PREFETCH_COUNT; fetch++)
      {
         int fetch_index = (hid + 1) * HEIGHT * fetch + (hid + 1) * ty + tx + 1 + (hid + 1);
         int index_in = HEIGHT * fetch + ty + 1;

         if (tx == 0)
            memcpy_async(input_node[(fetch % PREFETCH_COUNT) * HEIGHT + ty], input_cuda[index_in], pipe);

         memcpy_async(weight_matrix[(fetch % PREFETCH_COUNT) * HEIGHT * WIDTH + ty * WIDTH + tx], input_hidden_cuda[fetch_index], pipe);
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



      int compute_index = (hid + 1) * HEIGHT * compute + (hid + 1) * ty + tx + 1 + (hid + 1);
      weight_matrix[(compute % PREFETCH_COUNT) * HEIGHT * WIDTH + ty * WIDTH + tx] *= input_node[(compute % PREFETCH_COUNT) * HEIGHT + ty];
      block.sync();

      for (int i = 1; i <= __log2f(HEIGHT); i++) {
         int power_two = __powf(2, i);
         if (ty % power_two == 0)
            weight_matrix[(compute % PREFETCH_COUNT) * HEIGHT * WIDTH + ty * WIDTH + tx] += weight_matrix[(compute % PREFETCH_COUNT) * HEIGHT * WIDTH + (ty + power_two / 2) * WIDTH + tx];
         block.sync();
      }

      input_hidden_cuda[compute_index] = weight_matrix[(compute % PREFETCH_COUNT) * HEIGHT * WIDTH + ty * WIDTH + tx];
      block.sync();

      if (tx == 0)
      {
         hidden_partial_sum[compute * hid + ty] = weight_matrix[(compute % PREFETCH_COUNT) * HEIGHT * WIDTH + tx * WIDTH + ty];
      }
   }
}

// __global__ void
// bpnn_layerforward_CUDA(float *input_cuda,
//                        float *output_hidden_cuda,
//                        float *input_hidden_cuda,
//                        float *hidden_partial_sum,
//                        int in,
//                        int hid,
//                        int tile_size)
// {
//    int by = blockIdx.y;
//    int tx = threadIdx.x;
//    int ty = threadIdx.y;

//    int batches = tile_size / WIDTH;

//    __shared__ float input_node[HEIGHT];
//    __shared__ float weight_matrix[HEIGHT * WIDTH];

//    for (int b = 0; b < batches; b++)
//    {
//       int index = (hid + 1) * HEIGHT * (batches * by + b) + (hid + 1) * ty + tx + 1 + (hid + 1);

//       int index_in = HEIGHT * (batches * by + b) + ty + 1;

//       if (tx == 0)
//          input_node[ty] = input_cuda[index_in];

//       __syncthreads();

//       weight_matrix[ty * WIDTH + tx] = input_hidden_cuda[index];

//       __syncthreads();

//       weight_matrix[ty * WIDTH + tx] = weight_matrix[ty * WIDTH + tx] * input_node[ty];

//       __syncthreads();

//       for (int i = 1; i <= __log2f(HEIGHT); i++)
//       {

//          int power_two = __powf(2, i);

//          if (ty % power_two == 0)
//             weight_matrix[ty * WIDTH + tx] += weight_matrix[(ty + power_two / 2) * WIDTH + tx];

//          __syncthreads();
//       }

//       input_hidden_cuda[index] = weight_matrix[ty * WIDTH + tx];

//       __syncthreads();

//       if (tx == 0)
//       {
//          hidden_partial_sum[(batches * by + b) * hid + ty] = weight_matrix[tx * WIDTH + ty];
//       }
//    }
// }

__global__ void bpnn_adjust_weights_cuda(float * delta,   
										 int hid,         
										 float * ly,      
										 int in,          
										 float * w,       
										 float * oldw,
                               int tile_size)  									
{
   int by = blockIdx.y;

   int tx = threadIdx.x;
   int ty = threadIdx.y;

   int batches = tile_size / WIDTH;

   for (int b = 0; b < batches; b++) {
      int index = (hid + 1) * HEIGHT * (batches * by + b) + (hid + 1) * ty + tx + 1 + (hid + 1);
      int index_y = HEIGHT * (batches * by + b) + ty + 1;
      int index_x = tx + 1;
      // eta = 0.3;
      // momentum = 0.3;

      w[index] += ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));
      oldw[index] = ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));

      __syncthreads();

      if (ty == 0 && by == 0 && b == 0)
      {
         w[index_x] += ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
         oldw[index_x] = ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
      }
   }   
}
#endif 
