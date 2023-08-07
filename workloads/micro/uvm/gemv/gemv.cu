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

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

/* Problem size */
#define SIZE 40960
uint64_t NI;
uint64_t NJ;

/* Thread block dimensions */
#define DIM_THREAD_BLOCK 256

#define BATCH_SIZE 16

/* Declared constant values for ALPHA and BETA (same as values in PolyBench 2.0) */
#define ALPHA 1.1f
#define BETA 1.1f

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;
// typedef uint64_t DATA_TYPE;

void gemv(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C)
{
	uint64_t i, j;

	for (i = 0; i < NI; i++)
	{
		C[i] *= BETA;
		for (j = 0; j < NJ; j++)
		{
			C[i] += ALPHA * A[i * NJ + j] * B[j];
		}
	}
}

void initGPU(DATA_TYPE *A_gpu, DATA_TYPE *B_gpu, DATA_TYPE *C_gpu)
{
	uint64_t i, j;

	for (i = 0; i < NI; i++)
	{
		for (j = 0; j < NJ; j++)
		{
			A_gpu[i * NJ + j] = ((DATA_TYPE)i * j) / NI;
		}
	}

	for (j = 0; j < NJ; j++)
	{
		B_gpu[j] = ((DATA_TYPE)j + 1) / NJ;
	}

	for (i = 0; i < NI; i++)
	{
		C_gpu[i] = ((DATA_TYPE)i + 2) / NI;
	}
}

void initCPU(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C)
{
	uint64_t i, j;

	for (i = 0; i < NI; i++)
	{
		for (j = 0; j < NJ; j++)
		{
			A[i * NJ + j] = ((DATA_TYPE)i * j) / NI;
		}
	}

	for (j = 0; j < NJ; j++)
	{
		B[j] = ((DATA_TYPE)j + 1) / NJ;
	}

	for (i = 0; i < NI; i++)
	{
		C[i] = ((DATA_TYPE)i + 2) / NI;
	}
}

void compareResults(DATA_TYPE *C, DATA_TYPE *C_outputFromGpu)
{
	uint64_t i, fail;
	fail = 0;

	// Compare C1 and C2
	for (i = 0; i < NI; i++)
	{
		if (percentDiff(C[i], C_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
		{
			fail++;
			printf("%d, GPU is %f, CPU is %f.\n", i, C[i], C_outputFromGpu[i]);
		}
	}

	// Print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void GPU_argv_init()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
	printf("setting device %d with name %s\n", GPU_DEVICE, deviceProp.name);
	cudaSetDevice(GPU_DEVICE);
}

__global__ void gemv_kernel(DATA_TYPE *a, DATA_TYPE *b, DATA_TYPE *c, uint64_t NI, uint64_t NJ)
{
	uint64_t row = blockIdx.x * blockDim.x + threadIdx.x;
	uint64_t tx = threadIdx.x;

	__shared__ DATA_TYPE s_b[DIM_THREAD_BLOCK][BATCH_SIZE];

	DATA_TYPE tmp = BETA * c[row];
	__syncthreads();

	uint64_t tile = 0;
	uint64_t end_tile = NJ / BATCH_SIZE;

	for (; tile < end_tile; tile += 1)
	{
		uint64_t base_index = tile * BATCH_SIZE;
		for (uint64_t k = 0; k < BATCH_SIZE; k++)
		{
			s_b[tx][k] = b[base_index + k];
		}
		__syncthreads();

		for (uint64_t k = 0; k < BATCH_SIZE; k++)
		{
			tmp += ALPHA * a[row * NJ + base_index + k] * s_b[tx][k];
		}
		__syncthreads();
	}
	c[row] = tmp;
}

void gemvCuda(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C, DATA_TYPE *A_gpu, DATA_TYPE *B_gpu, DATA_TYPE *C_gpu)
{
	double t_start, t_end;

	dim3 block(DIM_THREAD_BLOCK);
	dim3 grid(NI / (DIM_THREAD_BLOCK));

	// t_start = rtclock();
	gemv_kernel<<<grid, block>>>(A_gpu, B_gpu, C_gpu, NI, NJ);
	cudaDeviceSynchronize();
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

	if (argc >= 3)
	{
		NI = atoll(argv[1]);
		NJ = atoll(argv[2]);
	}
	else
	{
		NI = SIZE;
		NJ = SIZE;
	}

	double t_start, t_end;

	DATA_TYPE *A;
	DATA_TYPE *B;
	DATA_TYPE *C;

	DATA_TYPE *A_gpu;
	DATA_TYPE *B_gpu;
	DATA_TYPE *C_gpu;

	A = (DATA_TYPE *)malloc(NI * NJ * sizeof(DATA_TYPE));
	B = (DATA_TYPE *)malloc(NJ * sizeof(DATA_TYPE));
	C = (DATA_TYPE *)malloc(NI * sizeof(DATA_TYPE));

	initCPU(A, B, C);
	GPU_argv_init();

	initTrace();
	startCPU();

	cudaMallocManaged(&A_gpu, sizeof(DATA_TYPE) * NI * NJ);
	cudaMallocManaged(&B_gpu, sizeof(DATA_TYPE) * NJ);
	cudaMallocManaged(&C_gpu, sizeof(DATA_TYPE) * NI);

	// initGPU(A_gpu, B_gpu, C_gpu);
	// overlapStartCPU();
	memcpy(A_gpu, A, NI * NJ * sizeof(DATA_TYPE));
	memcpy(B_gpu, B, NJ * sizeof(DATA_TYPE));
	memcpy(C_gpu, C, NI * sizeof(DATA_TYPE));
	// overlapEndCPU();

	gemvCuda(A, B, C, A_gpu, B_gpu, C_gpu);
	memcpy(C, C_gpu, NI * sizeof(DATA_TYPE));

	// t_start = rtclock();
	// gemv(A, B, C);
	// t_end = rtclock();
	// fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

	// compareResults(C_gpu, C);
	cudaFree(A_gpu);
	cudaFree(B_gpu);
	cudaFree(C_gpu);
	endCPU();
	finiTrace();

	free(A);
	free(B);
	free(C);
	return 0;
}
