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

#define GPU_DEVICE 5

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

/* Problem size */
#define SIZE 4096
uint64_t NI;
uint64_t NJ;
uint64_t NK;

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 32


/* Declared constant values for ALPHA and BETA (same as values in PolyBench 2.0) */
#define ALPHA 1.1f
#define BETA 1.1f

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;
// typedef uint64_t DATA_TYPE;

void gemm(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C)
{
	uint64_t i,j,k;
	
	for (i = 0; i < NI; i++) {
    	for (j = 0; j < NJ; j++) {
			C[i*NJ + j] *= BETA;
			for (k = 0; k < NK; ++k) {
	  			C[i*NJ + j] += ALPHA * A[i*NK + k] * B[k*NJ + j];
			}
      	}
	}
}

void initGPU(DATA_TYPE *A_gpu, DATA_TYPE *B_gpu, DATA_TYPE *C_gpu)
{
	uint64_t i, j;

  	for (i = 0; i < NI; i++) {
		for (j = 0; j < NK; j++) {
			A_gpu[i * NK + j] = ((DATA_TYPE)i * j) / NI;
		}
	}
    	

  	for (i = 0; i < NK; i++) {
		for (j = 0; j < NJ; j++) {
			B_gpu[i * NJ + j] = ((DATA_TYPE)i * j + 1) / NJ;
		}
	}
    	

  	for (i = 0; i < NI; i++) {
		for (j = 0; j < NJ; j++) {
			C_gpu[i * NJ + j] = ((DATA_TYPE)i * j + 2) / NJ;
		}
	} 
    	
}

void initCPU(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C)
{
	uint64_t i, j;

  	for (i = 0; i < NI; i++) {
		for (j = 0; j < NK; j++) {
			A[i * NK + j] = ((DATA_TYPE)i * j) / NI;
		}
	}
    	

  	for (i = 0; i < NK; i++) {
		for (j = 0; j < NJ; j++) {
			B[i * NJ + j] = ((DATA_TYPE)i * j + 1) / NJ;
		}
	}
    	

  	for (i = 0; i < NI; i++) {
		for (j = 0; j < NJ; j++) {
			C[i * NJ + j] = ((DATA_TYPE)i * j + 2) / NJ;
		}
	} 
    	
}


void compareResults(DATA_TYPE* C, DATA_TYPE* C_outputFromGpu)
{
	uint64_t i, j, fail;
	fail = 0;
	
	// Compare C1 and C2
	for (i=0; i < NI; i++) 
	{
		for (j=0; j < NJ; j++) 
		{
			// printf("%d, %d, GPU is %f, CPU is %f.\n", i, j, C[i*NJ + j], C_outputFromGpu[i*NJ + j]);
			if (percentDiff(C[i*NJ + j], C_outputFromGpu[i*NJ + j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
			{
				printf("%d, %d, GPU is %f, CPU is %f.\n", i, j, C[i*NJ + j], C_outputFromGpu[i*NJ + j]);
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

__global__ void gemm_kernel(DATA_TYPE *a, DATA_TYPE *b, DATA_TYPE *c, uint64_t NI, uint64_t NK, uint64_t NJ)
{
	cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
	pipeline pipe;

	uint64_t row = blockIdx.y * blockDim.y + threadIdx.y;
	uint64_t col = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ DATA_TYPE s_a[DIM_THREAD_BLOCK_X * DIM_THREAD_BLOCK_Y * PREFETCH_COUNT];
	__shared__ DATA_TYPE s_b[DIM_THREAD_BLOCK_X * DIM_THREAD_BLOCK_Y * PREFETCH_COUNT];

	DATA_TYPE tmp = BETA * c[row * NJ + col];

	uint64_t base_tiles = 0;
	uint64_t end_tile = base_tiles + NK / blockDim.x;

	uint64_t fetch = base_tiles;
	uint64_t tile_size = DIM_THREAD_BLOCK_X;
	uint64_t mem_size = DIM_THREAD_BLOCK_X * DIM_THREAD_BLOCK_Y;

	for (uint64_t compute = fetch; compute < end_tile; compute++)
	{
		for (; fetch < end_tile && fetch < compute + PREFETCH_COUNT; fetch++)
		{
			memcpy_async(s_a[(fetch % PREFETCH_COUNT) * mem_size + (threadIdx.y * blockDim.x + threadIdx.x)], a[row * NK + fetch * tile_size + threadIdx.x], pipe);
			memcpy_async(s_b[(fetch % PREFETCH_COUNT) * mem_size + (threadIdx.y * blockDim.x + threadIdx.x)], b[(fetch * tile_size + threadIdx.y) * NJ + col], pipe);

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

		for (uint64_t k = 0; k < blockDim.x; k++)
		{
			tmp += ALPHA * s_a[(compute % PREFETCH_COUNT) * mem_size + (threadIdx.y * blockDim.x + k)] * s_b[(compute % PREFETCH_COUNT) * mem_size + (k * blockDim.x + threadIdx.x)];
		}
		block.sync();
	}

	c[row * NJ + col] = tmp;
}

void gemmCuda(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C, DATA_TYPE *A_gpu, DATA_TYPE *B_gpu, DATA_TYPE *C_gpu)
{
	double t_start, t_end;

	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid((size_t)(ceil( ((float)NI)/ ((float)block.x) )),(size_t)(ceil( ((float)NJ)/ ((float)block.y) )));

	//t_start = rtclock();
	cudaStream_t stream1;
	cudaStream_t stream2;
	cudaStream_t stream3;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);
	cudaStreamCreate(&stream3);

	cudaMemPrefetchAsync(A_gpu, NI * NK * sizeof(DATA_TYPE), GPU_DEVICE, stream1);
	cudaStreamSynchronize(stream1);
	cudaMemPrefetchAsync(B_gpu, NK * NJ * sizeof(DATA_TYPE), GPU_DEVICE, stream2);
	cudaStreamSynchronize(stream2);
	cudaMemPrefetchAsync(C_gpu, NI * NJ * sizeof(DATA_TYPE), GPU_DEVICE, stream3);
	cudaStreamSynchronize(stream3);
	gemm_kernel<<<grid, block, 0, stream3>>>(A_gpu, B_gpu, C_gpu, NI, NK, NJ);
	cudaDeviceSynchronize();


	// gemm_kernel<<< grid, block >>>(A_gpu, B_gpu, C_gpu, NI, NK, NJ);
	// cudaDeviceSynchronize();
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

	if (argc >= 4) {
		NI = atoll(argv[1]);
		NK = atoll(argv[2]);
		NJ = atoll(argv[3]);
	} else {
		NI = SIZE;
		NK = SIZE;
		NJ = SIZE;
	}

	double t_start, t_end;

	DATA_TYPE* A;
	DATA_TYPE* B;  
	DATA_TYPE* C;

	DATA_TYPE *A_gpu;
	DATA_TYPE *B_gpu;
	DATA_TYPE *C_gpu; 

	A = (DATA_TYPE*)malloc(NI*NK*sizeof(DATA_TYPE)); 
	B = (DATA_TYPE*)malloc(NK*NJ*sizeof(DATA_TYPE));   
	C = (DATA_TYPE*)malloc(NI*NJ*sizeof(DATA_TYPE));

	initCPU(A,B,C);
	GPU_argv_init();

	initTrace();
	startCPU();

	cudaMallocManaged(&A_gpu, sizeof(DATA_TYPE) * NI * NK);
	cudaMallocManaged(&B_gpu, sizeof(DATA_TYPE) * NK * NJ);
	cudaMallocManaged(&C_gpu, sizeof(DATA_TYPE) * NI * NJ);

	// initGPU(A_gpu, B_gpu, C_gpu);
	// overlapStartCPU();
	memcpy(A_gpu, A, NI * NK * sizeof(DATA_TYPE));
	memcpy(B_gpu, B, NK * NJ * sizeof(DATA_TYPE));
	memcpy(C_gpu, C, NI * NJ * sizeof(DATA_TYPE));
	// overlapEndCPU();

	gemmCuda(A, B, C, A_gpu, B_gpu, C_gpu);
	memcpy(C, C_gpu, NI * NJ * sizeof(DATA_TYPE));

	// t_start = rtclock();	
	// gemm(A, B, C); // needed to keep benchmark accurate
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

