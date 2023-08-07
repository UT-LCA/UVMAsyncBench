
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>

#include "gemm.h"
#include "utils.h"
#include "cuda_dark.h"

#include <sys/time.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

using namespace nvcuda::experimental;

#define PREFETCH_COUNT 2

#define DIM_THREAD_BLOCK_X 16
#define DIM_THREAD_BLOCK_Y 16

double rtclock()
{
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday(&Tp, &Tzp);
    if (stat != 0)
        printf("Error return from gettimeofday: %d", stat);
    return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

__global__ void gemm_kernel(float *a, float *b, float *c, int M, int K, int N, float alpha, float beta)
{
	cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
	// Compute each thread's global row and column index
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;


	// Statically allocated shared memory
	__shared__ float s_a[DIM_THREAD_BLOCK_X * DIM_THREAD_BLOCK_Y];
	__shared__ float s_b[DIM_THREAD_BLOCK_X * DIM_THREAD_BLOCK_Y];
	// Accumulate in temporary variable

	float tmp = 0.0f;
	if (row < M && col < N) {
        
		tmp = beta * c[row * N + col];

        // Sweep tile across matrix
        for (int i = 0; i < K; i += blockDim.x) {
            int left = K - i;

            if ((i + threadIdx.x) < K) 
                s_a[threadIdx.y * blockDim.x + threadIdx.x] = a[row * K + i + threadIdx.x];

            if ((i + threadIdx.y) < K)
                s_b[threadIdx.y * blockDim.x + threadIdx.x] = b[(i + threadIdx.y) * N + col];
            
            block.sync();

            for (int k = 0; k < blockDim.x && k < left ; k++) {
                tmp += alpha * s_a[threadIdx.y * blockDim.x + k] * s_b[k * blockDim.x + threadIdx.x];
            }
            block.sync();
        }
		c[row * N + col] = tmp;
    }
    block.sync();
}

void gemmCuda(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta)
{
    dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
    dim3 grid((size_t)(ceil(((float)N) / ((float)block.y))), (size_t)(ceil(((float)M) / ((float)block.x))));

    gemm_kernel<<<grid, block>>>(A, B, C, M, K, N, alpha, beta);
    check_error(cudaPeekAtLastError());
}

void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA,
                         float *A_gpu, int lda,
                         float *B_gpu, int ldb,
                         float BETA,
                         float *C_gpu, int ldc)
{
    // printf("TA is %d, TB is %d, M is %d, N is %d, K is %d, lda is %d, ldb is %d, ldc is %d.\n", TA, TB, M, N, K, lda, ldb, ldc);
    if (TA == 0 && TB == 0) {
        gemmCuda(A_gpu, B_gpu, C_gpu, M, N, K, ALPHA, BETA);
    } else {
        cublasHandle_t handle = blas_handle();
        cudaError_t status = (cudaError_t) cublasSgemm(handle, (TB ? CUBLAS_OP_T : CUBLAS_OP_N), 
                (TA ? CUBLAS_OP_T : CUBLAS_OP_N), N, M, K, &ALPHA, B_gpu, ldb, A_gpu, lda, &BETA, C_gpu, ldc);
        check_error(status);
    }
}


void time_gpu_random_matrix(int TA, int TB, int m, int k, int n)
{
    float *a;
    if(!TA) a = random_matrix(m,k);
    else a = random_matrix(k,m);
    int lda = (!TA)?k:m;
    float *b;
    if(!TB) b = random_matrix(k,n);
    else b = random_matrix(n,k);
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);
    int i;
    clock_t start = clock(), end;
    for(i = 0; i<32; ++i){
        gemm_gpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
    }
    end = clock();
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf s\n",m,k,k,n, TA, TB, (float)(end-start)/CLOCKS_PER_SEC);
    free(a);
    free(b);
    free(c);
}

void time_gpu(int TA, int TB, int m, int k, int n)
{
    int iter = 10;
    float *a = random_matrix(m,k);
    float *b = random_matrix(k,n);

    int lda = (!TA)?k:m;
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);

    float *a_cl = cuda_make_array(a, m*k);
    float *b_cl = cuda_make_array(b, k*n);
    float *c_cl = cuda_make_array(c, m*n);

    int i;
    clock_t start = clock(), end;
    for(i = 0; i<iter; ++i){
        gemm_gpu(TA,TB,m,n,k,1,a_cl,lda,b_cl,ldb,1,c_cl,n);
        cudaThreadSynchronize();
    }
    double flop = ((double)m)*n*(2.*k + 2.)*iter;
    double gflop = flop/pow(10., 9);
    end = clock();
    double seconds = sec(end-start);
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf s, %lf GFLOPS\n",m,k,k,n, TA, TB, seconds, gflop/seconds);
    cuda_free(a_cl);
    cuda_free(b_cl);
    cuda_free(c_cl);
    free(a);
    free(b);
    free(c);
}


void test_gpu_accuracy(int TA, int TB, int m, int k, int n)
{
    srand(0);
    float *a;
    if(!TA) a = random_matrix(m,k);
    else a = random_matrix(k,m);
    int lda = (!TA)?k:m;
    float *b;
    if(!TB) b = random_matrix(k,n);
    else b = random_matrix(n,k);
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);
    float *c_gpu = random_matrix(m,n);
    memset(c, 0, m*n*sizeof(float));
    memset(c_gpu, 0, m*n*sizeof(float));
    int i;
    //pm(m,k,b);
    gemm_gpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c_gpu,n);
    //printf("GPU\n");
    //pm(m, n, c_gpu);

    gemm_cpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
    //printf("\n\nCPU\n");
    //pm(m, n, c);
    double sse = 0;
    for(i = 0; i < m*n; ++i) {
        //printf("%f %f\n", c[i], c_gpu[i]);
        sse += pow(c[i]-c_gpu[i], 2);
    }
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %g SSE\n",m,k,k,n, TA, TB, sse/(m*n));
    free(a);
    free(b);
    free(c);
    free(c_gpu);
}

int test_gpu_blas()
{
    /*
       test_gpu_accuracy(0,0,10,576,75); 

       test_gpu_accuracy(0,0,17,10,10); 
       test_gpu_accuracy(1,0,17,10,10); 
       test_gpu_accuracy(0,1,17,10,10); 
       test_gpu_accuracy(1,1,17,10,10); 

       test_gpu_accuracy(0,0,1000,10,100); 
       test_gpu_accuracy(1,0,1000,10,100); 
       test_gpu_accuracy(0,1,1000,10,100); 
       test_gpu_accuracy(1,1,1000,10,100); 

       test_gpu_accuracy(0,0,10,10,10); 

       time_gpu(0,0,64,2916,363); 
       time_gpu(0,0,64,2916,363); 
       time_gpu(0,0,64,2916,363); 
       time_gpu(0,0,192,729,1600); 
       time_gpu(0,0,384,196,1728); 
       time_gpu(0,0,256,196,3456); 
       time_gpu(0,0,256,196,2304); 
       time_gpu(0,0,128,4096,12544); 
       time_gpu(0,0,128,4096,4096); 
     */
    time_gpu(0,0,64,75,12544); 
    time_gpu(0,0,64,75,12544); 
    time_gpu(0,0,64,75,12544); 
    time_gpu(0,0,64,576,12544); 
    time_gpu(0,0,256,2304,784); 
    time_gpu(1,1,2304,256,784); 
    time_gpu(0,0,512,4608,196); 
    time_gpu(1,1,4608,512,196); 

    return 0;
}
