#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <cupti.h>
#include "../../../common/cupti_add.h"
#include "../../../common/cpu_timestamps.h"

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

using namespace nvcuda::experimental;

#define PREFETCH_COUNT 2

#ifdef TIMING
#include "timing.h"

struct timeval tv;
struct timeval tv_total_start, tv_total_end;
struct timeval tv_h2d_start, tv_h2d_end;
struct timeval tv_d2h_start, tv_d2h_end;
struct timeval tv_kernel_start, tv_kernel_end;
struct timeval tv_mem_alloc_start, tv_mem_alloc_end;
struct timeval tv_close_start, tv_close_end;
float init_time = 0, mem_alloc_time = 0, h2d_time = 0, kernel_time = 0,
      d2h_time = 0, close_time = 0, total_time = 0;
#endif

#define BLOCK_SIZE 256
#define STR_SIZE 256
#define DEVICE 0
#define HALO 1 // halo width along one direction when advancing to the next iteration

// #define BENCH_PRINT

void run(int argc, char **argv);

int rows, cols;
int *data;
int **wall;
int *result;
#define M_SEED 9
int pyramid_height;
int nblocks;

void init(int argc, char **argv)
{
    if (argc == 5)
    {
        cols = atoi(argv[1]);
        rows = atoi(argv[2]);
        pyramid_height = atoi(argv[3]);
        nblocks = atoi(argv[4]);
    }
    else
    {
        printf("Usage: dynproc row_len col_len pyramid_height\n");
        exit(0);
    }
    data = new int[rows * cols];
    wall = new int *[rows];
    for (int n = 0; n < rows; n++)
        wall[n] = data + cols * n;
    result = new int[cols];

    int seed = M_SEED;
    srand(seed);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            wall[i][j] = rand() % 10;
        }
    }
#ifdef BENCH_PRINT
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%d ", wall[i][j]);
        }
        printf("\n");
    }
#endif
}

void fatal(char *s)
{
    fprintf(stderr, "error: %s\n", s);
}

#define IN_RANGE(x, min, max) ((x) >= (min) && (x) <= (max))
#define CLAMP_RANGE(x, min, max) x = (x < (min)) ? min : ((x > (max)) ? max : x)
#define MIN(a, b) ((a) <= (b) ? (a) : (b))

__global__ void dynproc_kernel(
    int iteration,
    int *gpuWall,
    int *gpuSrc,
    int *gpuResults,
    int cols,
    int rows,
    int startStep,
    int border,
    int small_block_cols,
    int tile_size,
    int batches)
{
    cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
    __shared__ int prev[BLOCK_SIZE];
    __shared__ int result[BLOCK_SIZE];

    int bx = blockIdx.x;
    int tx = threadIdx.x;

    for (int b = 0; b < batches; b++)
    {
        // each block finally computes result for a small block
        // after N iterations.
        // it is the non-overlapping small blocks that cover
        // all the input data

        // calculate the boundary for the block according to
        // the boundary of its small block
        int blkX = bx * tile_size + small_block_cols * b - border;
        int blkXmax = blkX + BLOCK_SIZE - 1;

        // calculate the global thread coordination
        int xidx = blkX + tx;

        // effective range within this block that falls within
        // the valid range of the input data
        // used to rule out computation outside the boundary.
        int validXmin = (blkX < 0) ? -blkX : 0;
        int validXmax = (blkXmax > cols - 1) ? BLOCK_SIZE - 1 - (blkXmax - cols + 1) : BLOCK_SIZE - 1;

        int W = tx - 1;
        int E = tx + 1;

        W = (W < validXmin) ? validXmin : W;
        E = (E > validXmax) ? validXmax : E;

        bool isValid = IN_RANGE(tx, validXmin, validXmax);

        if (IN_RANGE(xidx, 0, cols - 1))
        {
            prev[tx] = gpuSrc[xidx];
        }
        block.sync(); // [Ronny] Added sync to avoid race on prev Aug. 14 2012
        bool computed;
        for (int i = 0; i < iteration; i++)
        {
            computed = false;
            if (IN_RANGE(tx, i + 1, BLOCK_SIZE - i - 2) &&
                isValid)
            {
                computed = true;
                int left = prev[W];
                int up = prev[tx];
                int right = prev[E];
                int shortest = MIN(left, up);
                shortest = MIN(shortest, right);
                int index = cols * (startStep + i) + xidx;
                result[tx] = shortest + gpuWall[index];
            }
            block.sync();
            if (i == iteration - 1)
                break;
            if (computed) // Assign the computation range
                prev[tx] = result[tx];
            block.sync(); // [Ronny] Added sync to avoid race on prev Aug. 14 2012
        }

        // update the global memory
        // after the last iteration, only threads coordinated within the
        // small block perform the calculation and switch on ``computed''
        if (computed)
        {
            gpuResults[xidx] = result[tx];
        }
    }

    
}

/*
   compute N time steps
*/
int calc_path(int *gpuWall, int *gpuResult[2], int rows, int cols,
              int pyramid_height, int blockCols, int borderCols, int tile_size, int batches)
{
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(nblocks);

    int src = 1, dst = 0;
    for (int t = 0; t < rows - 1; t += pyramid_height)
    {
        int temp = src;
        src = dst;
        dst = temp;

        int iteration = MIN(pyramid_height, rows - t - 1);
        int small_block_cols = BLOCK_SIZE - iteration * HALO * 2;
        dynproc_kernel<<<dimGrid, dimBlock>>>(
            iteration, gpuWall, gpuResult[src], gpuResult[dst],
            cols, rows, t, borderCols, small_block_cols, tile_size, batches);

        // for the measurement fairness
        cudaDeviceSynchronize();
    }
    return dst;
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

#define GPU_DEVICE 6

void GPU_argv_init()
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
    printf("setting device %d with name %s\n", GPU_DEVICE, deviceProp.name);
    cudaSetDevice(GPU_DEVICE);
}

int main(int argc, char *argv[])
{
    uint64_t start_tsc = rdtsc();
    uint64_t start_tsp = rdtsp();
    printf("start_tsc %lu start_tsp %lu\n", start_tsc, start_tsp);
    GPU_argv_init();

    run(argc, argv);

    return EXIT_SUCCESS;
}

void run(int argc, char **argv)
{
    init(argc, argv);

    /* --------------- pyramid parameters --------------- */
    int borderCols = (pyramid_height)*HALO;
    int smallBlockCol = BLOCK_SIZE - (pyramid_height)*HALO * 2;
    int blockCols = cols / smallBlockCol + ((cols % smallBlockCol == 0) ? 0 : 1);

    //ruihao
    int cols_per_block = cols / nblocks;
    if (cols_per_block < BLOCK_SIZE) cols_per_block = BLOCK_SIZE;
    int batches = cols_per_block / smallBlockCol + ((cols_per_block % smallBlockCol == 0) ? 0 : 1);

    // printf("pyramidHeight: %d\ngridSize: [%d]\nborder:[%d]\nblockSize: %d\nblockGrid:[%d]\ntargetBlock:[%d]\n",
    //        pyramid_height, cols, borderCols, BLOCK_SIZE, blockCols, smallBlockCol);
    printf("pyramidHeight: %d\ngridSize: [%d]\nborder:[%d]\nblockSize: %d\nblockGrid:[%d]\ntargetBlock:[%d]\n",
           pyramid_height, cols, borderCols, BLOCK_SIZE, nblocks, smallBlockCol);

    int *gpuWall, *gpuResult[2];
    int size = rows * cols;

    initTrace();
    startCPU();

    cudaMalloc((void **)&gpuResult[0], sizeof(int) * cols);
    cudaMalloc((void **)&gpuResult[1], sizeof(int) * cols);
    cudaMemcpy(gpuResult[0], data, sizeof(int) * cols, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&gpuWall, sizeof(int) * (size - cols));
    cudaMemcpy(gpuWall, data + cols, sizeof(int) * (size - cols), cudaMemcpyHostToDevice);

#ifdef TIMING
    gettimeofday(&tv_kernel_start, NULL);
#endif

    // int final_ret = calc_path(gpuWall, gpuResult, rows, cols,
    //                           pyramid_height, blockCols, borderCols);
    int final_ret = calc_path(gpuWall, gpuResult, rows, cols,
                              pyramid_height, blockCols, borderCols, cols_per_block, batches);

#ifdef TIMING
    gettimeofday(&tv_kernel_end, NULL);
    tvsub(&tv_kernel_end, &tv_kernel_start, &tv);
    kernel_time += tv.tv_sec * 1000.0 + (float)tv.tv_usec / 1000.0;
#endif

    cudaMemcpy(result, gpuResult[final_ret], sizeof(int) * cols, cudaMemcpyDeviceToHost);

#ifdef BENCH_PRINT
    for (int i = 0; i < cols; i++)
        printf("%d ", data[i]);
    printf("\n");
    for (int i = 0; i < cols; i++)
        printf("%d ", result[i]);
    printf("\n");
#endif

    cudaFree(gpuWall);
    cudaFree(gpuResult[0]);
    cudaFree(gpuResult[1]);

    endCPU();
    finiTrace();

    delete[] data;
    delete[] wall;
    delete[] result;

#ifdef TIMING
    printf("Exec: %f\n", kernel_time);
#endif
}
