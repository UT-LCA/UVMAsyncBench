/**
 * 3DConvolution.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 5

/* Problem size */
#define SIZE 4096
#define NBLOCKS 2
#define BATCH_SIZE 3

uint64_t NI;
uint64_t NJ;
uint64_t NK;
uint64_t nblocks;

/* Thread block dimensions */
#define DIM_THREAD_BLOCK 4

#define KERNEL 3

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;



void conv3D(DATA_TYPE* A, DATA_TYPE* B)
{
	uint64_t i, j, k;
	DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

	c11 = +2;  c21 = +5;  c31 = -8;
	c12 = -3;  c22 = +6;  c32 = -9;
	c13 = +4;  c23 = +7;  c33 = +10;

	for (i = 1; i < NI - 1; ++i) // 0
	{
		for (j = 1; j < NJ - 1; ++j) // 1
		{
			for (k = 1; k < NK -1; ++k) // 2
			{
				B[i*(NK * NJ) + j*NK + k] = c11 * A[(i - 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]  +  c13 * A[(i + 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]
					     +   c21 * A[(i - 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]  +  c23 * A[(i + 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]
					     +   c31 * A[(i - 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]  +  c33 * A[(i + 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]
					     +   c12 * A[(i + 0)*(NK * NJ) + (j - 1)*NK + (k + 0)]  +  c22 * A[(i + 0)*(NK * NJ) + (j + 0)*NK + (k + 0)]   
					     +   c32 * A[(i + 0)*(NK * NJ) + (j + 1)*NK + (k + 0)]  +  c11 * A[(i - 1)*(NK * NJ) + (j - 1)*NK + (k + 1)]  
					     +   c13 * A[(i + 1)*(NK * NJ) + (j - 1)*NK + (k + 1)]  +  c21 * A[(i - 1)*(NK * NJ) + (j + 0)*NK + (k + 1)]  
					     +   c23 * A[(i + 1)*(NK * NJ) + (j + 0)*NK + (k + 1)]  +  c31 * A[(i - 1)*(NK * NJ) + (j + 1)*NK + (k + 1)]  
					     +   c33 * A[(i + 1)*(NK * NJ) + (j + 1)*NK + (k + 1)];
			}
		}
	}
}

void initGPU(DATA_TYPE *A_gpu)
{
	uint64_t i, j, k;

	for (i = 0; i < NI; ++i)
	{
		for (j = 0; j < NJ; ++j)
		{
			for (k = 0; k < NK; ++k)
			{
				A_gpu[i * (NK * NJ) + j * NK + k] = i % 12 + 2 * (j % 7) + 3 * (k % 13);
			}
		}
	}
}

void initCPU(DATA_TYPE *A)
{
	uint64_t i, j, k;

	for (i = 0; i < NI; ++i)
    {
		for (j = 0; j < NJ; ++j)
		{
			for (k = 0; k < NK; ++k)
			{
				A[i*(NK * NJ) + j*NK + k] = i % 12 + 2 * (j % 7) + 3 * (k % 13);
			}
		}
	}
}


void compareResults(DATA_TYPE* B, DATA_TYPE* B_outputFromGpu)
{
	uint64_t i, j, k, fail;
	fail = 0;
	
	// Compare result from cpu and gpu...
	for (i = 1; i < NI - 1; ++i) // 0
	{
		for (j = 1; j < NJ - 1; ++j) // 1
		{
			for (k = 1; k < NK - 1; ++k) // 2
			{
				if (percentDiff(B[i*(NK * NJ) + j*NK + k], B_outputFromGpu[i*(NK * NJ) + j*NK + k]) > PERCENT_DIFF_ERROR_THRESHOLD)
				{
					printf("%d, %d, %d, CPU is %f, GPU is %f.\n", i, j, k, B[i * (NK * NJ) + j * NK + k], B_outputFromGpu[i * (NK * NJ) + j * NK + k]);
					fail++;
				}
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

__global__ void convolution3D_kernel(DATA_TYPE *A, DATA_TYPE *B, uint64_t NI, uint64_t NJ, uint64_t NK, uint64_t block_size)
{
	cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
	pipeline pipe;
	DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

	c11 = +2;
	c21 = +5;
	c31 = -8;
	c12 = -3;
	c22 = +6;
	c32 = -9;
	c13 = +4;
	c23 = +7;
	c33 = +10;

	uint64_t tile_dim_x = (NJ + DIM_THREAD_BLOCK - 1) / (DIM_THREAD_BLOCK * BATCH_SIZE);

	__shared__ DATA_TYPE tmp_A[PREFETCH_COUNT][DIM_THREAD_BLOCK * BATCH_SIZE + KERNEL - 1][DIM_THREAD_BLOCK * BATCH_SIZE + KERNEL - 1][DIM_THREAD_BLOCK * BATCH_SIZE + KERNEL - 1];
	__shared__ DATA_TYPE tmp_B[DIM_THREAD_BLOCK * BATCH_SIZE][DIM_THREAD_BLOCK * BATCH_SIZE][DIM_THREAD_BLOCK * BATCH_SIZE];

	// uint64_t total_tiles = tile_dim_x * tile_dim_x * tile_dim_x;

	uint64_t tiles_this_block_x = (block_size / (DIM_THREAD_BLOCK * BATCH_SIZE));
	uint64_t tiles_this_block = tiles_this_block_x * tiles_this_block_x * tiles_this_block_x;

	uint64_t base_tile = (blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x + blockIdx.x) * tiles_this_block;
	uint64_t fetch = base_tile;
	uint64_t end_tile = fetch + tiles_this_block;

	// printf("block_size is %d, tile_dim_x is %d, tiles_this_block_x is %d.\n", block_size, tile_dim_x, tiles_this_block_x);

	for (uint64_t compute = fetch; compute < end_tile; compute++)
	{
		for (; fetch < end_tile && fetch < compute + PREFETCH_COUNT; fetch++)
		{
			// block id
			uint64_t offset = fetch - base_tile;
			uint64_t block_id = fetch / tiles_this_block;

			uint64_t bz = block_id / (gridDim.y * gridDim.x) * tiles_this_block_x + offset / (tiles_this_block_x * tiles_this_block_x);
			uint64_t by = block_id % (gridDim.y * gridDim.x) / gridDim.x * tiles_this_block_x + offset % (tiles_this_block_x * tiles_this_block_x) / tiles_this_block_x;
			uint64_t bx = block_id % (gridDim.y * gridDim.x) % gridDim.x * tiles_this_block_x + offset % (tiles_this_block_x * tiles_this_block_x) % tiles_this_block_x;

			// thread id
			uint64_t tx = threadIdx.x;
			uint64_t ty = threadIdx.y;
			uint64_t tz = threadIdx.z;

			uint64_t index_A_z = DIM_THREAD_BLOCK * BATCH_SIZE * bz + BATCH_SIZE * tz;
			uint64_t index_A_y = DIM_THREAD_BLOCK * BATCH_SIZE * by + BATCH_SIZE * ty;
			uint64_t index_A_x = DIM_THREAD_BLOCK * BATCH_SIZE * bx + BATCH_SIZE * tx;

			uint64_t index_A_z_start = DIM_THREAD_BLOCK * BATCH_SIZE * bz;
			uint64_t index_A_y_start = DIM_THREAD_BLOCK * BATCH_SIZE * by;
			uint64_t index_A_x_start = DIM_THREAD_BLOCK * BATCH_SIZE * bx;

			uint64_t index_A_z_bound = DIM_THREAD_BLOCK * BATCH_SIZE * bz + BATCH_SIZE * DIM_THREAD_BLOCK;
			uint64_t index_A_y_bound = DIM_THREAD_BLOCK * BATCH_SIZE * by + BATCH_SIZE * DIM_THREAD_BLOCK;
			uint64_t index_A_x_bound = DIM_THREAD_BLOCK * BATCH_SIZE * bx + BATCH_SIZE * DIM_THREAD_BLOCK;

			// fetch A
			for (uint64_t i = 0; i < BATCH_SIZE; i++)
			{
				for (uint64_t j = 0; j < BATCH_SIZE; j++)
				{
					for (uint64_t k = 0; k < BATCH_SIZE; k++)
					{
						if ((index_A_z + i) < NI && (index_A_y + j) < NJ && (index_A_x + k) < NK)
						{
							memcpy_async(tmp_A[fetch % PREFETCH_COUNT][tz * BATCH_SIZE + i][ty * BATCH_SIZE + j][tx * BATCH_SIZE + k], A[(index_A_z + i) * NJ * NK + (index_A_y + j) * NK + index_A_x + k], pipe);
						}
					}
				}
			}

			// fetch A -- padding
			for (uint64_t i = 0; i < KERNEL - 1; i++)
			{
				for (uint64_t j = 0; j < BATCH_SIZE * DIM_THREAD_BLOCK + KERNEL - 1; j++)
				{
					for (uint64_t k = 0; k < BATCH_SIZE * DIM_THREAD_BLOCK + KERNEL - 1; k++)
					{
						if ((index_A_z_bound + i) < NI && (index_A_y_start + j) < NJ && (index_A_x_start + k) < NK)
						{
							memcpy_async(tmp_A[fetch % PREFETCH_COUNT][DIM_THREAD_BLOCK * BATCH_SIZE + i][j][k], A[(index_A_z_bound + i) * NJ * NK + (index_A_y_start + j) * NK + index_A_x_start + k], pipe);
						}
					}
				}
			}

			// fetch A -- padding
			for (uint64_t i = 0; i < BATCH_SIZE * DIM_THREAD_BLOCK + KERNEL - 1; i++)
			{
				for (uint64_t j = 0; j < KERNEL - 1; j++)
				{
					for (uint64_t k = 0; k < BATCH_SIZE * DIM_THREAD_BLOCK + KERNEL - 1; k++)
					{
						if ((index_A_z_start + i) < NI && (index_A_y_bound + j) < NJ && (index_A_x_start + k) < NK)
						{
							memcpy_async(tmp_A[fetch % PREFETCH_COUNT][i][DIM_THREAD_BLOCK * BATCH_SIZE + j][k], A[(index_A_z_start + i) * NJ * NK + (index_A_y_bound + j) * NK + index_A_x_start + k], pipe);
						}
					}
				}
			}

			// fetch A -- padding
			for (uint64_t i = 0; i < BATCH_SIZE * DIM_THREAD_BLOCK + KERNEL - 1; i++)
			{
				for (uint64_t j = 0; j < BATCH_SIZE * DIM_THREAD_BLOCK + KERNEL - 1; j++)
				{
					for (uint64_t k = 0; k < KERNEL - 1; k++)
					{
						if ((index_A_z_start + i) < NI && (index_A_y_start + j) < NJ && (index_A_x_bound + k) < NK)
						{
							memcpy_async(tmp_A[fetch % PREFETCH_COUNT][i][j][DIM_THREAD_BLOCK * BATCH_SIZE + k], A[(index_A_z_start + i) * NJ * NK + (index_A_y_start + j) * NK + index_A_x_bound + k], pipe);
						}
					}
				}
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

		// block id
		uint64_t offset = compute - base_tile;
		uint64_t block_id = compute / tiles_this_block;

		uint64_t bz = block_id / (gridDim.y * gridDim.x) * tiles_this_block_x + offset / (tiles_this_block_x * tiles_this_block_x);
		uint64_t by = block_id % (gridDim.y * gridDim.x) / gridDim.x * tiles_this_block_x + offset % (tiles_this_block_x * tiles_this_block_x) / tiles_this_block_x;
		uint64_t bx = block_id % (gridDim.y * gridDim.x) % gridDim.x * tiles_this_block_x + offset % (tiles_this_block_x * tiles_this_block_x) % tiles_this_block_x;

		// thread id
		uint64_t tx = threadIdx.x;
		uint64_t ty = threadIdx.y;
		uint64_t tz = threadIdx.z;

		uint64_t index_B_z = DIM_THREAD_BLOCK * BATCH_SIZE * bz + BATCH_SIZE * tz + 1;
		uint64_t index_B_y = DIM_THREAD_BLOCK * BATCH_SIZE * by + BATCH_SIZE * ty + 1;
		uint64_t index_B_x = DIM_THREAD_BLOCK * BATCH_SIZE * bx + BATCH_SIZE * tx + 1;

		// Computation
		for (uint64_t i = 0; i < BATCH_SIZE; i++)
		{
			for (uint64_t j = 0; j < BATCH_SIZE; j++)
			{
				for (uint64_t k = 0; k < BATCH_SIZE; k++)
				{
					tmp_B[tz * BATCH_SIZE + i][ty * BATCH_SIZE + j][tx * BATCH_SIZE + k] = 0;
				}
			}
		}
		block.sync();

		for (uint64_t i = 0; i < BATCH_SIZE; i++)
		{
			for (uint64_t j = 0; j < BATCH_SIZE; j++)
			{
				for (uint64_t k = 0; k < BATCH_SIZE; k++)
				{
					tmp_B[tz * BATCH_SIZE + i][ty * BATCH_SIZE + j][tx * BATCH_SIZE + k] =
						c11 * tmp_A[compute % PREFETCH_COUNT][tz * BATCH_SIZE + i][ty * BATCH_SIZE + j][tx * BATCH_SIZE + k] + c13 * tmp_A[compute % PREFETCH_COUNT][tz * BATCH_SIZE + i + 2][ty * BATCH_SIZE + j][tx * BATCH_SIZE + k] + c21 * tmp_A[compute % PREFETCH_COUNT][tz * BATCH_SIZE + i][ty * BATCH_SIZE + j][tx * BATCH_SIZE + k] + c23 * tmp_A[compute % PREFETCH_COUNT][tz * BATCH_SIZE + i + 2][ty * BATCH_SIZE + j][tx * BATCH_SIZE + k] + c31 * tmp_A[compute % PREFETCH_COUNT][tz * BATCH_SIZE + i][ty * BATCH_SIZE + j][tx * BATCH_SIZE + k] + c33 * tmp_A[compute % PREFETCH_COUNT][tz * BATCH_SIZE + i + 2][ty * BATCH_SIZE + j][tx * BATCH_SIZE + k] + c12 * tmp_A[compute % PREFETCH_COUNT][tz * BATCH_SIZE + i + 1][ty * BATCH_SIZE + j][tx * BATCH_SIZE + k + 1] + c22 * tmp_A[compute % PREFETCH_COUNT][tz * BATCH_SIZE + i + 1][ty * BATCH_SIZE + j + 1][tx * BATCH_SIZE + k + 1] + c32 * tmp_A[compute % PREFETCH_COUNT][tz * BATCH_SIZE + i + 1][ty * BATCH_SIZE + j + 2][tx * BATCH_SIZE + k + 1] + c11 * tmp_A[compute % PREFETCH_COUNT][tz * BATCH_SIZE + i][ty * BATCH_SIZE + j][tx * BATCH_SIZE + k + 2] + c13 * tmp_A[compute % PREFETCH_COUNT][tz * BATCH_SIZE + i + 2][ty * BATCH_SIZE + j][tx * BATCH_SIZE + k + 2] + c21 * tmp_A[compute % PREFETCH_COUNT][tz * BATCH_SIZE + i][ty * BATCH_SIZE + j + 1][tx * BATCH_SIZE + k + 2] + c23 * tmp_A[compute % PREFETCH_COUNT][tz * BATCH_SIZE + i + 2][ty * BATCH_SIZE + j + 1][tx * BATCH_SIZE + k + 2] + c31 * tmp_A[compute % PREFETCH_COUNT][tz * BATCH_SIZE + i][ty * BATCH_SIZE + j + 2][tx * BATCH_SIZE + k + 2] + c33 * tmp_A[compute % PREFETCH_COUNT][tz * BATCH_SIZE + i + 2][ty * BATCH_SIZE + j + 2][tx * BATCH_SIZE + k + 2];
				}
			}
		}
		block.sync();

		// Store B
		for (uint64_t i = 0; i < BATCH_SIZE; i++)
		{
			for (uint64_t j = 0; j < BATCH_SIZE; j++)
			{
				for (uint64_t k = 0; k < BATCH_SIZE; k++)
				{
					if ((index_B_z + i + 1) < NI && (index_B_y + j + 1) < NJ && (index_B_x + k + 1) < NK)
					{
						B[(index_B_z + i) * NJ * NK + (index_B_y + j) * NK + index_B_x + k] = tmp_B[tz * BATCH_SIZE + i][ty * BATCH_SIZE + j][tx * BATCH_SIZE + k];
					}
				}
			}
		}
		block.sync();
	}
}

void convolution3DCuda(DATA_TYPE* A_gpu, DATA_TYPE* B_gpu)
{
	double t_start, t_end;

	dim3 block(DIM_THREAD_BLOCK, DIM_THREAD_BLOCK, DIM_THREAD_BLOCK);
	dim3 grid(nblocks, nblocks, nblocks);

	uint64_t block_size = (NI + (nblocks - 1)) / nblocks;

	// t_start = rtclock();

	cudaStream_t stream1;
	cudaStream_t stream2;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);

	cudaMemPrefetchAsync(A_gpu, NI * NJ * NK * sizeof(DATA_TYPE), GPU_DEVICE, stream1);
	cudaStreamSynchronize(stream1);
	cudaMemPrefetchAsync(B_gpu, NI * NJ * NK * sizeof(DATA_TYPE), GPU_DEVICE, stream2);
	cudaStreamSynchronize(stream2);
	convolution3D_kernel<<<grid, block, 0, stream2>>>(A_gpu, B_gpu, NI, NJ, NK, block_size);
	cudaDeviceSynchronize();

	// convolution3D_kernel<<<grid, block>>>(A_gpu, B_gpu, NI, NJ, NK, block_size);
	// cudaDeviceSynchronize();
	// t_end = rtclock();
	// fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

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
	if (argc >= 5) {
		NI = atoll(argv[1]);
		NJ = atoll(argv[2]);
		NK = atoll(argv[3]);
		nblocks = atoi(argv[4]);
	} else {
		NI = SIZE;
		NJ = SIZE;
		NK = SIZE;
		nblocks = NBLOCKS;
	}
	double t_start, t_end;

	DATA_TYPE* A;
	DATA_TYPE* B;
	DATA_TYPE *A_gpu;
	DATA_TYPE *B_gpu;

	A = (DATA_TYPE*)malloc(NI*NJ*NK*sizeof(DATA_TYPE));
	B = (DATA_TYPE*)malloc(NI*NJ*NK*sizeof(DATA_TYPE));
	initCPU(A);
	GPU_argv_init();

	initTrace();
	startCPU();

	cudaMallocManaged(&A_gpu, sizeof(DATA_TYPE) * NI * NJ * NK);
	cudaMallocManaged(&B_gpu, sizeof(DATA_TYPE) * NI * NJ * NK);

	// initGPU(A_gpu);
	// overlapStartCPU();
	memcpy(A_gpu, A, NI * NJ * NK * sizeof(DATA_TYPE));
	// overlapEndCPU();

	convolution3DCuda(A_gpu, B_gpu);
	memcpy(B, B_gpu, NI * NJ * NK * sizeof(DATA_TYPE));

	cudaFree(A_gpu);
	cudaFree(B_gpu);

	endCPU();
	finiTrace();

	// t_start = rtclock();
	// conv3D(A, B);
	// t_end = rtclock();
	// fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);
	
	// compareResults(B, B_gpu);

	free(A);
	free(B);

    return 0;
}
