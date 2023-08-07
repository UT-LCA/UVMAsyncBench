/**
 * 2DConvolution.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <cuda.h>

#include "../../../common/cupti_add.h"
#include "../../../common/cpu_timestamps.h"

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

using namespace nvcuda::experimental;

#define PREFETCH_COUNT 2

#define SMALL_FLOAT_VAL 0.00000001f

double rtclock()
{
	struct timezone Tzp;
	struct timeval Tp;
	uint64_t stat;
	stat = gettimeofday(&Tp, &Tzp);
	if (stat != 0)
		printf("Error return from gettimeofday: %d", stat);
	return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

float absVal(float a)
{
	if (a < 0)
	{
		return (a * -1);
	}
	else
	{
		return a;
	}
}

float percentDiff(double val1, double val2)
{
	if ((absVal(val1) < 0.01) && (absVal(val2) < 0.01))
	{
		return 0.0f;
	}

	else
	{
		return 100.0f * (absVal(absVal(val1 - val2) / absVal(val1 + SMALL_FLOAT_VAL)));
	}
}

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 5

/* Problem size */
#define SIZE 4096
#define NBLOCKS 32
#define BATCH_SIZE 4

uint64_t NI;
uint64_t NJ;
uint64_t nblocks;


/* Thread block dimensions */
#define KERNEL 3
#define DIM_THREAD_BLOCK 8

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void conv2D(DATA_TYPE* A, DATA_TYPE* B)
{
	uint64_t i, j;
	DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

	c11 = +0.2;  c21 = +0.5;  c31 = -0.8;
	c12 = -0.3;  c22 = +0.6;  c32 = -0.9;
	c13 = +0.4;  c23 = +0.7;  c33 = +0.10;

	for (i = 1; i < NI - 1; ++i) // 0
	{
		for (j = 1; j < NJ - 1; ++j) // 1
		{
			B[i*NJ + j] = c11 * A[(i - 1)*NJ + (j - 1)]  +  c12 * A[(i + 0)*NJ + (j - 1)]  +  c13 * A[(i + 1)*NJ + (j - 1)]
				+ c21 * A[(i - 1)*NJ + (j + 0)]  +  c22 * A[(i + 0)*NJ + (j + 0)]  +  c23 * A[(i + 1)*NJ + (j + 0)] 
				+ c31 * A[(i - 1)*NJ + (j + 1)]  +  c32 * A[(i + 0)*NJ + (j + 1)]  +  c33 * A[(i + 1)*NJ + (j + 1)];
		}
	}
}

void initGPU(DATA_TYPE* A_gpu)
{
	uint64_t i, j;

	for (i = 0; i < NI; ++i) {
		for (j = 0; j < NJ; ++j) {
			A_gpu[i * NJ + j] = ((DATA_TYPE)i * j) / NI;
		}
    }
}

void initCPU(DATA_TYPE* A)
{
	uint64_t i, j;

	for (i = 0; i < NI; ++i) {
		for (j = 0; j < NJ; ++j) {
			A[i * NJ + j] = ((DATA_TYPE)i * j) / NI;
		}
    }
}


void compareResults(DATA_TYPE* B, DATA_TYPE* B_outputFromGpu)
{
	uint64_t i, j, fail;
	fail = 0;
	
	// Compare a and b
	for (i=1; i < (NI-1); i++) 
	{
		for (j=1; j < (NJ-1); j++) 
		{
			if (percentDiff(B[i*NJ + j], B_outputFromGpu[i*NJ + j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
			{
				printf("%d, %d, CPU is %f, GPU is %f.\n", i, j, B[i * NJ + j], B_outputFromGpu[i * NJ + j]);
				fail++;
			}
		}
	}
	
	// Print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
	
}


void GPU_argv_init()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
	printf("setting device %d with name %s\n",GPU_DEVICE,deviceProp.name);
	cudaSetDevice( GPU_DEVICE );
}

__global__ void Convolution2D_kernel(DATA_TYPE *A, DATA_TYPE *B, uint64_t NI, uint64_t NJ, uint64_t block_size)
{
	cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
	uint64_t tile_dim_x = (NJ + DIM_THREAD_BLOCK - 1) / (DIM_THREAD_BLOCK * BATCH_SIZE);

	__shared__ DATA_TYPE tmp_A[DIM_THREAD_BLOCK * BATCH_SIZE + KERNEL - 1][DIM_THREAD_BLOCK * BATCH_SIZE + KERNEL - 1];
	__shared__ DATA_TYPE tmp_B[DIM_THREAD_BLOCK * BATCH_SIZE][DIM_THREAD_BLOCK * BATCH_SIZE];

	uint64_t total_tiles = tile_dim_x * tile_dim_x;

	uint64_t tiles_this_block_x = (block_size / (DIM_THREAD_BLOCK * BATCH_SIZE));
	uint64_t tiles_this_block = tiles_this_block_x * tiles_this_block_x;

	uint64_t base_tile = (blockIdx.y * gridDim.x + blockIdx.x) * tiles_this_block;
	uint64_t tile = base_tile;
	uint64_t end_tile = tile + tiles_this_block;

	// DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

	// c11 = +0.2;  c21 = +0.5;  c31 = -0.8;
	// c12 = -0.3;  c22 = +0.6;  c32 = -0.9;
	// c13 = +0.4;  c23 = +0.7;  c33 = +0.10;

	DATA_TYPE c[KERNEL][KERNEL];

	c[0][0] = +0.2;
	c[1][0] = +0.5;
	c[2][0] = -0.8;
	c[0][1] = -0.3;
	c[1][1] = +0.6;
	c[2][1] = -0.9;
	c[0][2] = +0.4;
	c[1][2] = +0.7;
	c[2][2] = +0.10;

	for (; tile < end_tile; tile += 1)
	{
		// block id
		uint64_t offset = tile - base_tile;
		uint64_t block_id = tile / tiles_this_block;
		uint64_t bx = block_id % gridDim.x * tiles_this_block_x + offset % tiles_this_block_x;
		uint64_t by = block_id / gridDim.x * tiles_this_block_x + offset / tiles_this_block_x;

		uint64_t batch_size = DIM_THREAD_BLOCK * BATCH_SIZE;

		// thread id
		uint64_t tx = threadIdx.x;
		uint64_t ty = threadIdx.y;

		uint64_t index_B_y = DIM_THREAD_BLOCK * BATCH_SIZE * by + BATCH_SIZE * ty + 1;
		uint64_t index_B_x = DIM_THREAD_BLOCK * BATCH_SIZE * bx + BATCH_SIZE * tx + 1;

		uint64_t index_A_y = DIM_THREAD_BLOCK * BATCH_SIZE * by + BATCH_SIZE * ty;
		uint64_t index_A_x = DIM_THREAD_BLOCK * BATCH_SIZE * bx + BATCH_SIZE * tx;

		uint64_t index_A_y_start = DIM_THREAD_BLOCK * BATCH_SIZE * by;
		uint64_t index_A_x_start = DIM_THREAD_BLOCK * BATCH_SIZE * bx;

		uint64_t index_A_y_bound = DIM_THREAD_BLOCK * BATCH_SIZE * by + BATCH_SIZE * DIM_THREAD_BLOCK;
		uint64_t index_A_x_bound = DIM_THREAD_BLOCK * BATCH_SIZE * bx + BATCH_SIZE * DIM_THREAD_BLOCK;

		// fetch A
		for (uint64_t i = 0; i < BATCH_SIZE; i++) {
			for (uint64_t j = 0; j < BATCH_SIZE; j++) {
				if ((index_A_y + i) < NI && (index_A_x + j) < NJ) {
					tmp_A[ty * BATCH_SIZE + i][tx * BATCH_SIZE + j] = A[(index_A_y + i) * NJ + index_A_x + j];
					tmp_B[ty * BATCH_SIZE + i][tx * BATCH_SIZE + j] = 0;
				}
			}
		}

		// fetch A -- padding
		for (uint64_t i = 0; i < KERNEL - 1; i++) {
			for (uint64_t j = 0; j < BATCH_SIZE * DIM_THREAD_BLOCK + KERNEL - 1; j++) {
				if ((index_A_y_bound + i) < NI && (index_A_x_start + j) < NJ) {
					tmp_A[DIM_THREAD_BLOCK * BATCH_SIZE + i][j] = A[(index_A_y_bound + i) * NJ + index_A_x_start + j];
				}
			}
		}

		// fetch A -- padding
		for (uint64_t i = 0; i < BATCH_SIZE * DIM_THREAD_BLOCK + KERNEL - 1; i++) {
			for (uint64_t j = 0; j < KERNEL - 1; j++) {
				if ((index_A_y_start + i) < NI && (index_A_x_bound + j) < NJ) {
					tmp_A[i][DIM_THREAD_BLOCK * BATCH_SIZE + j] = A[(index_A_y_start + i) * NJ + index_A_x_bound + j];
				}
			}
		}
		block.sync();

		// Computation
		for (uint64_t i = 0; i < BATCH_SIZE; i++) {
			for (uint64_t j = 0; j < BATCH_SIZE; j++) {
				for (uint64_t m = 0; m < KERNEL; m++) {
					for (uint64_t n = 0; n < KERNEL; n++) {
						tmp_B[ty * BATCH_SIZE + i][tx * BATCH_SIZE + j] += tmp_A[ty * BATCH_SIZE + i + m][tx * BATCH_SIZE + j + n] * c[n][m];
					}
				}
			}
		}
		block.sync();

		// Store B
		for (uint64_t i = 0; i < BATCH_SIZE; i++) {
			for (uint64_t j = 0; j < BATCH_SIZE; j++) {
				if ((index_B_y + i + 1) < NI && (index_B_x + j + 1) < NJ) {
					B[(index_B_y + i) * NJ + index_B_x + j] = tmp_B[ty * BATCH_SIZE + i][tx * BATCH_SIZE + j];
				}
			}
		}
		block.sync();
	}
}

void convolution2DCuda(DATA_TYPE *A_gpu, DATA_TYPE *B_gpu)
{
	double t_start, t_end;

	uint64_t output_width = NI - KERNEL + 1;
	uint64_t output_height = NJ - KERNEL + 1;

	dim3 block(DIM_THREAD_BLOCK, DIM_THREAD_BLOCK);
	dim3 grid(nblocks, nblocks);

	uint64_t block_size = (NJ + (nblocks - 1)) / nblocks;

	// t_start = rtclock();

	cudaStream_t stream1;
	cudaStream_t stream2;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);

	cudaMemPrefetchAsync(A_gpu, NI * NJ * sizeof(DATA_TYPE), GPU_DEVICE, stream1);
	cudaStreamSynchronize(stream1);
	cudaMemPrefetchAsync(B_gpu, NI * NJ * sizeof(DATA_TYPE), GPU_DEVICE, stream2);
	cudaStreamSynchronize(stream2);
	Convolution2D_kernel<<<grid, block, 0, stream2>>>(A_gpu, B_gpu, NI, NJ, block_size);
	cudaDeviceSynchronize();

	// Convolution2D_kernel<<<grid, block>>>(A_gpu, B_gpu, NI, NJ, block_size);
	// cudaDeviceSynchronize();

	// t_end = rtclock();
	// fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start); //);
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
	if (argc >= 4) {
		NI = atoll(argv[1]);
		NJ = atoll(argv[2]);
		nblocks = atoi(argv[3]);
	} else {
		NI = SIZE;
		NJ = SIZE;
		nblocks = NBLOCKS;
	}
	double t_start, t_end;

	DATA_TYPE* A;
	DATA_TYPE* B;
	DATA_TYPE *A_gpu;
	DATA_TYPE *B_gpu;

	A = (DATA_TYPE*)malloc(NI*NJ*sizeof(DATA_TYPE));
	B = (DATA_TYPE*)malloc(NI*NJ*sizeof(DATA_TYPE));
	initCPU(A);
	GPU_argv_init();

	initTrace();
	startCPU();

	cudaMallocManaged(&A_gpu, sizeof(DATA_TYPE) * NI * NJ);
	cudaMallocManaged(&B_gpu, sizeof(DATA_TYPE) * NI * NJ);

	// initGPU(A_gpu);
	// overlapStartCPU();
	memcpy(A_gpu, A, NI*NJ*sizeof(DATA_TYPE));
	// overlapEndCPU();

	convolution2DCuda(A_gpu, B_gpu);

	memcpy(B, B_gpu, NI * NJ * sizeof(DATA_TYPE));

	cudaFree(A_gpu);
	cudaFree(B_gpu);
	endCPU();
	finiTrace();
	
	// t_start = rtclock();
	// conv2D(A, B);
	// t_end = rtclock();
	// fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);//);

	// compareResults(B, B_gpu);
	free(A);
	free(B);
	
	return 0;
}
