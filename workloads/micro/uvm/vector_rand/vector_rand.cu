/**
 * gemm.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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
#include <curand_kernel.h>
#include "../../../common/cupti_add.h"
#include "../../../common/cpu_timestamps.h"

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

#define GPU_DEVICE 5

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.005

/* Problem size */
#define SIZE 4096000
#define ITER 100
uint64_t NI;

/* Thread block dimensions */
#define DIM_THREAD_BLOCK 256

#define BATCH_SIZE 16

#define LCG_A 1.1f
#define LCG_B 1.1f

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;
// typedef uint64_t DATA_TYPE;

void saxpy(DATA_TYPE *A, uint64_t iterations)
{	
	for (uint64_t i = 0; i < NI; i++) {
		for (uint64_t iter = 0; iter < iterations; iter++) {
			A[i] = LCG_A * A[i] + LCG_B;
		}
	}
}

void initGPU(DATA_TYPE *A_gpu)
{
	for (uint64_t i = 0; i < NI; i++)
	{
		A_gpu[i] = ((DATA_TYPE)i) / NI;
	}
    	
}

void initCPU(DATA_TYPE *A)
{
	for (uint64_t i = 0; i < NI; i++)
	{
		A[i] = ((DATA_TYPE)i) / NI;
	}
    	
}

void compareResults(DATA_TYPE* A, DATA_TYPE* A_outputFromGpu)
{
	uint64_t fail = 0;
	
	// Compare C1 and C2
	for (uint64_t i = 0; i < NI; i++) {
		// printf("%lld, GPU is %f, CPU is %f.\n", i, A[i], A_outputFromGpu[i]);
		if (percentDiff(A[i], A_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD) {
			fail++;
			printf("%lld, GPU is %f, CPU is %f.\n", i, A[i], A_outputFromGpu[i]);
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

__global__ void vector_rand_kernel(DATA_TYPE *a, uint64_t NI, uint64_t iterations, uint64_t block_size, size_t seed)
{
	// Compute each thread's global row and column index
	const uint64_t mem_size = DIM_THREAD_BLOCK * BATCH_SIZE;

	__shared__ DATA_TYPE tmp[mem_size];

	curandState_t randState;
	size_t tx = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(seed, tx, 0, &randState);
	size_t idx = 0;

	uint64_t total_tiles = NI / mem_size;
	uint64_t base_tiles = total_tiles / gridDim.x;

	uint64_t tiles_this_block = block_size / mem_size;

	uint64_t tile = base_tiles * blockIdx.x;
	uint64_t end_tile = tile + tiles_this_block;

	for (; tile < end_tile; tile += 1)
	{
		for (uint64_t i = threadIdx.x; i < mem_size; i += blockDim.x)
		{
			// tmp[i] = a[tile * mem_size + i];

			idx = curand(&randState);
			idx <<= 32;
			idx |= curand(&randState);
			tmp[i] = a[tile * mem_size + idx % mem_size];
		}

		__syncthreads();

		for (uint64_t i = threadIdx.x; i < mem_size; i += blockDim.x)
		{
			for (uint64_t iter = 0; iter < iterations; iter++)
			{
				tmp[i] = LCG_A * tmp[i] + LCG_B;
			}
		}

		for (uint64_t i = threadIdx.x; i < mem_size; i += blockDim.x)
		{
			a[tile * mem_size + idx % mem_size] = tmp[i];
		}
	}
}

void saxpyCuda(DATA_TYPE *A, DATA_TYPE *A_gpu, uint64_t iterations, uint64_t block_size)
{
	double t_start, t_end;
	if (block_size <= DIM_THREAD_BLOCK)
		block_size = DIM_THREAD_BLOCK;

	dim3 block(DIM_THREAD_BLOCK);
	dim3 grid(NI / block_size);

	//t_start = rtclock();
	vector_rand_kernel<<<grid, block>>>(A_gpu, NI, iterations, block_size, 832945);
	cudaDeviceSynchronize();
	//t_end = rtclock();

	//fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);
}

extern inline __attribute__((always_inline)) unsigned long rdtsc()
{
           unsigned long a, d;

              __asm__ volatile("rdtsc" : "=a" (a), "=d" (d));

                 return (a | (d << 32));
}

extern inline __attribute__((always_inline)) unsigned long rdtsp() {
                struct timespec tms;
                    if (clock_gettime(CLOCK_REALTIME, &tms)) {
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

	uint64_t iterations = ITER;
	uint64_t block_size = DIM_THREAD_BLOCK * BATCH_SIZE;
	if (argc >= 4) {
		NI = atoll(argv[1]);
		iterations = atoi(argv[2]);
		block_size = atoi(argv[3]);
	}
	else {
		NI = SIZE;
		iterations = ITER;
		block_size = DIM_THREAD_BLOCK * BATCH_SIZE;
	}

	int nblocks = NI / block_size;
	if (nblocks > 64)
	{
		nblocks = 64;
		block_size = NI / nblocks;
	}

	double t_start, t_end;

	DATA_TYPE *A;
	DATA_TYPE *A_gpu;

	A = (DATA_TYPE *)malloc(NI * sizeof(DATA_TYPE));
	initCPU(A);
	GPU_argv_init();

	initTrace();
	startCPU();

	cudaMallocManaged(&A_gpu, sizeof(DATA_TYPE) * NI);

	// initGPU(A_gpu);
	// overlapStartCPU();
	memcpy(A_gpu, A, NI * sizeof(DATA_TYPE));
	// overlapEndCPU();

	saxpyCuda(A, A_gpu, iterations, block_size);
	memcpy(A, A_gpu, NI * sizeof(DATA_TYPE));

	// t_start = rtclock();
	//saxpy(A, iterations);
	// t_end = rtclock();
	// fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

	//compareResults(A_gpu, A);
	cudaFree(A_gpu);
	endCPU();
	finiTrace();

	free(A);
	return 0;
}
