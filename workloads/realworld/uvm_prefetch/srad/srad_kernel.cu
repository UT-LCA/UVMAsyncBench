#include "srad.h"
#include <stdio.h>

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

using namespace nvcuda::experimental;

#define PREFETCH_COUNT 2

__global__ void
srad_cuda_1(
    float *E_C,
    float *W_C,
    float *N_C,
    float *S_C,
    float *J_cuda,
    float *C_cuda,
    int cols,
    int rows,
    float q0sqr,
    int block_size)
{
    cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
    // shared memory allocation
    __shared__ float temp[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float temp_result[BLOCK_SIZE * BLOCK_SIZE];

    __shared__ float north[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float south[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float east[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float west[BLOCK_SIZE * BLOCK_SIZE];

    int tile_dim_x = cols / BLOCK_SIZE;

    int total_tiles = tile_dim_x * tile_dim_x;
    int tiles_this_block = (block_size / BLOCK_SIZE) * (block_size / BLOCK_SIZE);
    int tiles_this_block_x = (block_size / BLOCK_SIZE);

    int base_tile = (blockIdx.y * gridDim.x + blockIdx.x) * tiles_this_block;
    int tile = base_tile;
    int end_tile = tile + tiles_this_block;

    for (; tile < end_tile; tile += 1)
    {
        // block id
        int offset = tile - base_tile;
        int block_id = tile / tiles_this_block;
        int bx = block_id % gridDim.x * tiles_this_block_x + offset % tiles_this_block_x;
        int by = block_id / gridDim.x * tiles_this_block_x + offset / tiles_this_block_x;

        // thread id
        int tx = threadIdx.x;
        int ty = threadIdx.y;

        // indices
        int index = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + tx;
        int index_n = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + tx - cols;
        int index_s = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * BLOCK_SIZE + tx;
        int index_w = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty - 1;
        int index_e = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + BLOCK_SIZE;

        if (index_n < 0) index_n = 0;
        if (index_s >= (cols * rows)) index_s = cols * rows - 1;
        if (index_w < 0) index_w = 0;
        if (index_e >= (cols * rows)) index_e = cols * rows - 1;

        float n, w, e, s, jc, g2, l, num, den, qsqr, c;

        // load data to shared memory
        north[ty * BLOCK_SIZE + tx] = J_cuda[index_n];
        south[ty * BLOCK_SIZE + tx] = J_cuda[index_s];
        if (by == 0)
        {
            north[ty * BLOCK_SIZE + tx] = J_cuda[BLOCK_SIZE * bx + tx];
        }
        else if (by == tile_dim_x - 1)
        {
            south[ty * BLOCK_SIZE + tx] = J_cuda[cols * BLOCK_SIZE * (tile_dim_x - 1) + BLOCK_SIZE * bx + cols * (BLOCK_SIZE - 1) + tx];
        }
        block.sync();

        west[ty * BLOCK_SIZE + tx] = J_cuda[index_w];
        east[ty * BLOCK_SIZE + tx] = J_cuda[index_e];

        if (bx == 0)
        {
            west[ty * BLOCK_SIZE + tx] = J_cuda[cols * BLOCK_SIZE * by + cols * ty];
        }
        else if (bx == tile_dim_x - 1)
        {
            east[ty * BLOCK_SIZE + tx] = J_cuda[cols * BLOCK_SIZE * by + BLOCK_SIZE * (tile_dim_x - 1) + cols * ty + BLOCK_SIZE - 1];
        }

        block.sync();
        temp[ty * BLOCK_SIZE + tx] = J_cuda[index];

        block.sync();

        jc = temp[ty * BLOCK_SIZE + tx];

        if (ty == 0 && tx == 0)
        { // nw
            n = north[ty * BLOCK_SIZE + tx] - jc;
            s = temp[(ty + 1) * BLOCK_SIZE + tx] - jc;
            w = west[ty * BLOCK_SIZE + tx] - jc;
            e = temp[ty * BLOCK_SIZE + tx + 1] - jc;
        }
        else if (ty == 0 && tx == BLOCK_SIZE - 1)
        { // ne
            n = north[ty * BLOCK_SIZE + tx] - jc;
            s = temp[(ty + 1) * BLOCK_SIZE + tx] - jc;
            w = temp[ty * BLOCK_SIZE + tx - 1] - jc;
            e = east[ty * BLOCK_SIZE + tx] - jc;
        }
        else if (ty == BLOCK_SIZE - 1 && tx == BLOCK_SIZE - 1)
        { // se
            n = temp[(ty - 1) * BLOCK_SIZE + tx] - jc;
            s = south[ty * BLOCK_SIZE + tx] - jc;
            w = temp[ty * BLOCK_SIZE + tx - 1] - jc;
            e = east[ty * BLOCK_SIZE + tx] - jc;
        }
        else if (ty == BLOCK_SIZE - 1 && tx == 0)
        { // sw
            n = temp[(ty - 1) * BLOCK_SIZE + tx] - jc;
            s = south[ty * BLOCK_SIZE + tx] - jc;
            w = west[ty * BLOCK_SIZE + tx] - jc;
            e = temp[ty * BLOCK_SIZE + tx + 1] - jc;
        }

        else if (ty == 0)
        { // n
            n = north[ty * BLOCK_SIZE + tx] - jc;
            s = temp[(ty + 1) * BLOCK_SIZE + tx] - jc;
            w = temp[ty * BLOCK_SIZE + tx - 1] - jc;
            e = temp[ty * BLOCK_SIZE + tx + 1] - jc;
        }
        else if (tx == BLOCK_SIZE - 1)
        { // e
            n = temp[(ty - 1) * BLOCK_SIZE + tx] - jc;
            s = temp[(ty + 1) * BLOCK_SIZE + tx] - jc;
            w = temp[ty * BLOCK_SIZE + tx - 1] - jc;
            e = east[ty * BLOCK_SIZE + tx] - jc;
        }
        else if (ty == BLOCK_SIZE - 1)
        { // s
            n = temp[(ty - 1) * BLOCK_SIZE + tx] - jc;
            s = south[ty * BLOCK_SIZE + tx] - jc;
            w = temp[ty * BLOCK_SIZE + tx - 1] - jc;
            e = temp[ty * BLOCK_SIZE + tx + 1] - jc;
        }
        else if (tx == 0)
        { // w
            n = temp[(ty - 1) * BLOCK_SIZE + tx] - jc;
            s = temp[(ty + 1) * BLOCK_SIZE + tx] - jc;
            w = west[ty * BLOCK_SIZE + tx] - jc;
            e = temp[ty * BLOCK_SIZE + tx + 1] - jc;
        }
        else
        { // the data elements which are not on the borders
            n = temp[(ty - 1) * BLOCK_SIZE + tx] - jc;
            s = temp[(ty + 1) * BLOCK_SIZE + tx] - jc;
            w = temp[ty * BLOCK_SIZE + tx - 1] - jc;
            e = temp[ty * BLOCK_SIZE + tx + 1] - jc;
        }

        g2 = (n * n + s * s + w * w + e * e) / (jc * jc);

        l = (n + s + w + e) / jc;

        num = (0.5 * g2) - ((1.0 / 16.0) * (l * l));
        den = 1 + (.25 * l);
        qsqr = num / (den * den);

        // diffusion coefficent (equ 33)
        den = (qsqr - q0sqr) / (q0sqr * (1 + q0sqr));
        c = 1.0 / (1.0 + den);

        // saturate diffusion coefficent
        if (c < 0)
        {
            temp_result[ty * BLOCK_SIZE + tx] = 0;
        }
        else if (c > 1)
        {
            temp_result[ty * BLOCK_SIZE + tx] = 1;
        }
        else
        {
            temp_result[ty * BLOCK_SIZE + tx] = c;
        }

        block.sync();

        C_cuda[index] = temp_result[ty * BLOCK_SIZE + tx];
        E_C[index] = e;
        W_C[index] = w;
        S_C[index] = s;
        N_C[index] = n;
    }
}

__global__ void
srad_cuda_2(
    float *E_C,
    float *W_C,
    float *N_C,
    float *S_C,
    float *J_cuda,
    float *C_cuda,
    int cols,
    int rows,
    float lambda,
    float q0sqr,
    int block_size)
{
    cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
    // shared memory allocation
    __shared__ float south_c[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float east_c[BLOCK_SIZE * BLOCK_SIZE];

    __shared__ float c_cuda_temp[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float c_cuda_result[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float temp[BLOCK_SIZE * BLOCK_SIZE];

    int tile_dim_x =  cols / BLOCK_SIZE;

    int total_tiles = tile_dim_x * tile_dim_x;
    int tiles_this_block = (block_size / BLOCK_SIZE) * (block_size / BLOCK_SIZE);

    int base_tile = (blockIdx.y * gridDim.x + blockIdx.x) * tiles_this_block;
    int tile = base_tile;
    int end_tile = tile + tiles_this_block;

    for (; tile < end_tile; tile += 1)
    {
        //block id
        int bx = tile % tile_dim_x;
        int by = tile / tile_dim_x;

        //thread id
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        
        // indices
        int index = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + tx;
        int index_s = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * BLOCK_SIZE + tx;
        int index_e = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + BLOCK_SIZE;

        if (index_s >= (cols * rows)) index_s = cols * rows - 1;
        if (index_e >= (cols * rows)) index_e = cols * rows - 1;

        float cc, cn, cs, ce, cw, d_sum;

        // load data to shared memory
        temp[ty * BLOCK_SIZE + tx] = J_cuda[index];
        block.sync();

        south_c[ty * BLOCK_SIZE + tx] = C_cuda[index_s];
        if (by == tile_dim_x - 1)
        {
            south_c[ty * BLOCK_SIZE + tx] = C_cuda[cols * BLOCK_SIZE * (tile_dim_x - 1) + BLOCK_SIZE * bx + cols * (BLOCK_SIZE - 1) + tx];
        }
        block.sync();

        east_c[ty * BLOCK_SIZE + tx] = C_cuda[index_e];
        if (bx == tile_dim_x - 1)
        {
            east_c[ty * BLOCK_SIZE + tx] = C_cuda[cols * BLOCK_SIZE * by + BLOCK_SIZE * (tile_dim_x - 1) + cols * ty + BLOCK_SIZE - 1];
        }
        block.sync();

        c_cuda_temp[ty * BLOCK_SIZE + tx] = C_cuda[index];
        block.sync();
        cc = c_cuda_temp[ty * BLOCK_SIZE + tx];

        if (ty == BLOCK_SIZE - 1 && tx == BLOCK_SIZE - 1)
        { // se
            cn = cc;
            cs = south_c[ty * BLOCK_SIZE + tx];
            cw = cc;
            ce = east_c[ty * BLOCK_SIZE + tx];
        }
        else if (tx == BLOCK_SIZE - 1)
        { // e
            cn = cc;
            cs = c_cuda_temp[(ty + 1) * BLOCK_SIZE + tx];
            cw = cc;
            ce = east_c[ty * BLOCK_SIZE + tx];
        }
        else if (ty == BLOCK_SIZE - 1)
        { // s
            cn = cc;
            cs = south_c[ty * BLOCK_SIZE + tx];
            cw = cc;
            ce = c_cuda_temp[ty * BLOCK_SIZE + tx + 1];
        }
        else
        { // the data elements which are not on the borders
            cn = cc;
            cs = c_cuda_temp[(ty + 1) * BLOCK_SIZE + tx];
            cw = cc;
            ce = c_cuda_temp[ty * BLOCK_SIZE + tx + 1];
        }

        // divergence (equ 58)
        d_sum = cn * N_C[index] + cs * S_C[index] + cw * W_C[index] + ce * E_C[index];

        // image update (equ 61)
        c_cuda_result[ty * BLOCK_SIZE + tx] = temp[ty * BLOCK_SIZE + tx] + 0.25 * lambda * d_sum;

        block.sync();

        J_cuda[index] = c_cuda_result[ty * BLOCK_SIZE + tx];
    }
}
