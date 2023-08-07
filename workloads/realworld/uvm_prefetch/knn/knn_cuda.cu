/** Modifed version of knn-CUDA from https://github.com/vincentfpgarcia/kNN-CUDA
 * The modifications are
 *      removed texture memory usage
 *      removed split query KNN computation
 *      added feature extraction with bilinear interpolation
 *
 * Last modified by Christopher B. Choy <chrischoy@ai.stanford.edu> 12/23/2016
 */

// Includes
#include "cuda.h"
#include <cstdio>
#include <sys/time.h>
#include <time.h>
// Constants used by the program
#define BLOCK_DIM 16

#include "../../../common/cupti_add.h"
#include "../../../common/cpu_timestamps.h"

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

using namespace nvcuda::experimental;

#define PREFETCH_COUNT 2


#define GPU_DEVICE 6

//-----------------------------------------------------------------------------------------------//
//                                            KERNELS //
//-----------------------------------------------------------------------------------------------//
__global__ void extract_with_interpolation(int nthreads, float *data,
                                           float *n_xy_coords,
                                           float *extracted_data,
                                           int n_max_coord, int channels,
                                           int height, int width) {

  int x0, x1, y0, y1, nc;
  float wx0, wx1, wy0, wy1;
  int n, nd;
  float x, y;

  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < (nthreads);
       index += blockDim.x * gridDim.x) {
    n = (index / n_max_coord);
    nd = n * n_max_coord * channels;
    x = n_xy_coords[index * 2];
    y = n_xy_coords[index * 2 + 1];

    x0 = static_cast<int>(floor(x));
    x1 = x0 + 1;
    y0 = static_cast<int>(floor(y));
    y1 = y0 + 1;

    x0 = x0 <= 0 ? 0 : (x0 >= (width - 1) ? (width - 1) : x0);
    y0 = y0 <= 0 ? 0 : (y0 >= (height - 1) ? (height - 1) : y0);
    x1 = x1 <= 0 ? 0 : (x1 >= (width - 1) ? (width - 1) : x1);
    y1 = y1 <= 0 ? 0 : (y1 >= (height - 1) ? (height - 1) : y1);

    wx0 = static_cast<float>(x1) - x;
    wx1 = x - x0;
    wy0 = static_cast<float>(y1) - y;
    wy1 = y - y0;

    if (x0 == x1) {
      wx0 = 1;
      wx1 = 0;
    }
    if (y0 == y1) {
      wy0 = 1;
      wy1 = 0;
    }
    for (int c = 0; c < channels; c++) {
      nc = (n * channels + c) * height;
      // extracted_data[index * channels + c] = wy0 * wx0 * data[(nc + y0) *
      // width + x0]
      // extracted_data[nd + index % n_max_coord + n_max_coord * c] = index;
      extracted_data[nd + index % n_max_coord + n_max_coord * c] =
          wy0 * wx0 * data[(nc + y0) * width + x0] +
          wy1 * wx0 * data[(nc + y1) * width + x0] +
          wy0 * wx1 * data[(nc + y0) * width + x1] +
          wy1 * wx1 * data[(nc + y1) * width + x1];
    }
  }
}

/**
  * Computes the distance between two matrix A (reference points) and
  * B (query points) containing respectively wA and wB points.
  *
  * @param A     pointer on the matrix A
  * @param wA    width of the matrix A = number of points in A
  * @param B     pointer on the matrix B
  * @param wB    width of the matrix B = number of points in B
  * @param dim   dimension of points = height of matrices A and B
  * @param AB    pointer on the matrix containing the wA*wB distances computed
  */
__global__ void cuComputeDistanceGlobal(float *A, int wA, float *B, int wB,
                                        int dim, float *AB) {
  cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
  // Declaration of the shared memory arrays As and Bs used to store the
  // sub-matrix of A and B
  __shared__ float shared_A[BLOCK_DIM][BLOCK_DIM];
  __shared__ float shared_B[BLOCK_DIM][BLOCK_DIM];

  // Sub-matrix of A (begin, step, end) and Sub-matrix of B (begin, step)
  __shared__ int begin_A;
  __shared__ int begin_B;
  __shared__ int step_A;
  __shared__ int step_B;
  __shared__ int end_A;

  // Thread index
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Other variables
  float tmp;
  float ssd = 0;

  // Loop parameters
  begin_A = BLOCK_DIM * blockIdx.y;
  begin_B = BLOCK_DIM * blockIdx.x;
  step_A = BLOCK_DIM * wA;
  step_B = BLOCK_DIM * wB;
  end_A = begin_A + (dim - 1) * wA;

  // if (blockIdx.x == 0 && blockIdx.y == 0 && tx == 0 && ty == 0)
  //   printf("begin_A is %d, end_A is %d, step_A is %d, begin_B is %d, step_B is %d\n", begin_A, end_A, step_A, begin_B, step_B);

  // Conditions
  int cond0 = (begin_A + tx < wA); // used to write in shared memory
  int cond1 = (begin_B + tx < wB); // used to write in shared memory & to
                                   // computations and to write in output matrix
  int cond2 =
      (begin_A + ty < wA); // used to computations and to write in output matrix
  // Loop over all the sub-matrices of A and B required to compute the block
  // sub-matrix
  for (int a = begin_A, b = begin_B; a <= end_A; a += step_A, b += step_B) {
    // if (blockIdx.x == 0 && blockIdx.y == 0 && tx == 0 && ty == 0)
    //   printf("a is %d, end_A is %d, step_A is %d, b is %d, step_B is %d\n", begin_A, end_A, step_A, begin_B, step_B);
    // Load the matrices from device memory to shared memory; each thread loads
    // one element of each matrix
    if (a / wA + ty < dim) {
      shared_A[ty][tx] = (cond0) ? A[a + wA * ty + tx] : 0;
      shared_B[ty][tx] = (cond1) ? B[b + wB * ty + tx] : 0;
    } else {
      shared_A[ty][tx] = 0;
      shared_B[ty][tx] = 0;
    }

    // Synchronize to make sure the matrices are loaded
    block.sync();
    // Compute the difference between the two matrixes; each thread computes one
    // element of the block sub-matrix
    if (cond2 && cond1) {
      for (int k = 0; k < BLOCK_DIM; ++k) {
        tmp = shared_A[k][ty] - shared_B[k][tx];
        ssd += tmp * tmp;
      }
    }

    // Synchronize to make sure that the preceding computation is done before
    // loading two new sub-matrices of A and B in the next iteration
    block.sync();
  }

  // Write the block sub-matrix to device memory; each thread writes one element
  if (cond2 && cond1) {
    AB[(begin_A + ty) * wB + begin_B + tx] = ssd;
  }
}

/**
  * Gathers k-th smallest distances for each column of the distance matrix in
 * the top.
  *
  * @param dist        distance matrix
  * @param ind         index matrix
  * @param width       width of the distance matrix and of the index matrix
  * @param height      height of the distance matrix and of the index matrix
  * @param k           number of neighbors to consider
  */
__global__ void cuInsertionSort(float *dist, int *ind, int width, int height,
                                int k) {
  // printf("test2\n");
  // Variables
  int l, i, j;
  float *p_dist;
  int *p_ind;
  float curr_dist, max_dist;
  int curr_row, max_row;
  unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;

  if (xIndex < width) {
    // Pointer shift, initialization, and max value
    p_dist = dist + xIndex;
    p_ind = ind + xIndex;
    max_dist = p_dist[0];
    p_ind[0] = 0;

    // Part 1 : sort kth firt elementZ
    for (l = 1; l < k; l++) {
      curr_row = l * width;
      curr_dist = p_dist[curr_row];
      if (curr_dist < max_dist) {
        i = l - 1;
        for (int a = 0; a < l - 1; a++) {
          if (p_dist[a * width] > curr_dist) {
            i = a;
            break;
          }
        }
        for (j = l; j > i; j--) {
          p_dist[j * width] = p_dist[(j - 1) * width];
          p_ind[j * width] = p_ind[(j - 1) * width];
        }
        p_dist[i * width] = curr_dist;
        p_ind[i * width] = l;
      } else {
        p_ind[l * width] = l;
      }
      max_dist = p_dist[curr_row];
    }

    // Part 2 : insert element in the k-th first lines
    max_row = (k - 1) * width;
    for (l = k; l < height; l++) {
      curr_dist = p_dist[l * width];
      if (curr_dist < max_dist) {
        i = k - 1;
        for (int a = 0; a < k - 1; a++) {
          if (p_dist[a * width] > curr_dist) {
            i = a;
            break;
          }
        }
        for (j = k - 1; j > i; j--) {
          p_dist[j * width] = p_dist[(j - 1) * width];
          p_ind[j * width] = p_ind[(j - 1) * width];
        }
        p_dist[i * width] = curr_dist;
        p_ind[i * width] = l;
        max_dist = p_dist[max_row];
      }
    }
  }
}

/**
  * Computes the square root of the first line (width-th first element)
  * of the distance matrix.
  *
  * @param dist    distance matrix
  * @param width   width of the distance matrix
  * @param k       number of neighbors to consider
  */
__global__ void cuParallelSqrt(float *dist, int width, int k) {
  unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
  // printf("test3\n");
  if (xIndex < width && yIndex < k)
    dist[yIndex * width + xIndex] = sqrt(dist[yIndex * width + xIndex]);
}

//-----------------------------------------------------------------------------------------------//
//                                   K-th NEAREST NEIGHBORS //
//-----------------------------------------------------------------------------------------------//

/**
  * Prints the error message return during the memory allocation.
  *
  * @param error        error value return by the memory allocation function
  * @param memorySize   size of memory tried to be allocated
  */
void printErrorMessage(cudaError_t error, int memorySize) {
  printf("==================================================\n");
  printf("MEMORY ALLOCATION ERROR  : %s\n", cudaGetErrorString(error));
  printf("Whished allocated memory : %d\n", memorySize);
  printf("==================================================\n");
}

/**
  * K nearest neighbor algorithm
  * - Initialize CUDA
  * - Allocate device memory
  * - Copy point sets (reference and query points) from host to device memory
  * - Compute the distances + indexes to the k nearest neighbors for each query
 * point
  * - Copy distances from device to host memory
  *
  * @param ref_host      reference points ; pointer to linear matrix
  * @param ref_width     number of reference points ; width of the matrix
  * @param query_host    query points ; pointer to linear matrix
  * @param query_width   number of query points ; width of the matrix
  * @param height        dimension of points ; height of the matrices
  * @param k             number of neighbor to consider
  * @param dist_host     distances to k nearest neighbors ; pointer to linear
 * matrix
  * @param dist_host     indexes of the k nearest neighbors ; pointer to linear
 * matrix
  *
  */
void knn_cuda(float *ref_host, int ref_width, float *query_host,
              int query_width, int height, int k, float *dist_host,
              int *ind_host) {
  // Grids ans threads
  dim3 g_16x16(query_width / 16, ref_width / 16, 1);
  dim3 t_16x16(16, 16, 1);
  if (query_width % 16 != 0)
    g_16x16.x += 1;
  if (ref_width % 16 != 0)
    g_16x16.y += 1;
  //
  dim3 g_256x1(query_width / 256, 1, 1);
  dim3 t_256x1(256, 1, 1);
  if (query_width % 256 != 0)
    g_256x1.x += 1;

  dim3 g_k_16x16(query_width / 16, k / 16, 1);
  dim3 t_k_16x16(16, 16, 1);
  if (query_width % 16 != 0)
    g_k_16x16.x += 1;
  if (k % 16 != 0)
    g_k_16x16.y += 1;

  // printf("ref_width is %d, query_width is %d, height is %d\n", ref_width, query_width, height);

  cudaStream_t stream1;
  cudaStream_t stream2;
  cudaStream_t stream3;
  cudaStream_t stream4;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);
  cudaStreamCreate(&stream3);
  cudaStreamCreate(&stream4);

  // knn_cuda(ref_device, ref_nb, query_device, query_nb, dim, k, dist_device, ind_device);
  // memcpy(ref_device, ref, ref_nb * dim * sizeof(float));
  // memcpy(query_device, query, query_nb * dim * sizeof(float));
  // memcpy(dist, dist_device, query_nb * ref_nb * sizeof(float));
  // memcpy(ind, ind_device, query_nb * k * sizeof(int));

  cudaMemPrefetchAsync(ref_host, ref_width * height * sizeof(float), GPU_DEVICE, stream1);
  cudaStreamSynchronize(stream1);
  cudaMemPrefetchAsync(query_host, query_width * height * sizeof(float), GPU_DEVICE, stream2);
  cudaStreamSynchronize(stream2);
  cudaMemPrefetchAsync(dist_host, query_width * ref_width * sizeof(float), GPU_DEVICE, stream3);
  cudaStreamSynchronize(stream3);
  cudaMemPrefetchAsync(ind_host, query_width * k * sizeof(int), GPU_DEVICE, stream4);
  cudaStreamSynchronize(stream4);

  // Kernel 1: Compute all the distances
  cuComputeDistanceGlobal<<<g_16x16, t_16x16, 0, stream3>>>(ref_host, ref_width, query_host,
                                                query_width, height, dist_host);
  // Kernel 2: Sort each column
  cuInsertionSort<<<g_256x1, t_256x1, 0, stream4>>>(dist_host, ind_host, query_width,
                                        ref_width, k);
  // Kernel 3: Compute square root of k first elements
  cuParallelSqrt<<<g_k_16x16, t_k_16x16, 0, stream4>>>(dist_host, query_width, k);
  cudaDeviceSynchronize();
}

float compute_distance(const float *ref, int ref_nb, const float *query,
                       int query_nb, int dim, int ref_index, int query_index) {
  float sum = 0.f;
  for (int d = 0; d < dim; ++d) {
    const float diff =
        ref[d * ref_nb + ref_index] - query[d * query_nb + query_index];
    sum += diff * diff;
  }
  return sqrtf(sum);
}

void modified_insertion_sort(float *dist, int *index, int length, int k) {

  // Initialise the first index
  index[0] = 0;

  // Go through all points
  for (int i = 1; i < length; ++i) {

    // Store current distance and associated index
    float curr_dist = dist[i];
    int curr_index = i;

    // Skip the current value if its index is >= k and if it's higher the k-th
    // slready sorted mallest value
    if (i >= k && curr_dist >= dist[k - 1]) {
      continue;
    }

    // Shift values (and indexes) higher that the current distance to the right
    int j = min(i, k - 1);
    while (j > 0 && dist[j - 1] > curr_dist) {
      dist[j] = dist[j - 1];
      index[j] = index[j - 1];
      --j;
    }

    // Write the current distance and index at their position
    dist[j] = curr_dist;
    index[j] = curr_index;
  }
}

bool knn_c(const float *ref, int ref_nb, const float *query, int query_nb,
           int dim, int k, float *knn_dist, int *knn_index) {
  // Allocate local array to store all the distances / indexes for a given query
  // point
  float *dist = (float *)malloc(ref_nb * sizeof(float));
  int *index = (int *)malloc(ref_nb * sizeof(int));

  // Allocation checks
  if (!dist || !index) {
    printf("Memory allocation error\n");
    free(dist);
    free(index);
    return false;
  }

  // Process one query point at the time
  for (int i = 0; i < query_nb; ++i) {

    // Compute all distances / indexes
    for (int j = 0; j < ref_nb; ++j) {
      dist[j] = compute_distance(ref, ref_nb, query, query_nb, dim, j, i);
      index[j] = j;
    }

    // Sort distances / indexes
    modified_insertion_sort(dist, index, ref_nb, k);

    // Copy k smallest distances and their associated index
    for (int j = 0; j < k; ++j) {
      knn_dist[j * query_nb + i] = dist[j];
      knn_index[j * query_nb + i] = index[j];
    }
  }

  // Memory clean-up
  free(dist);
  free(index);
  return true;
}

/**
  * Example of use of kNN search CUDA.
  */
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
  // Variables and parameters
  float *ref;           // Pointer to reference point array
  float *query;         // Pointer to query point array
  float *dist, *dist_c; // Pointer to distance array
  int *ind, *ind_c;     // Pointer to index array
  int ref_nb = 4096;    // Reference point number, max=65535
  int query_nb = 4096;  // Query point number,     max=65535
  int dim = 128;       // Dimension of points
  int k = 20;           // Nearest neighbors to consider
  int iterations = 100;

  if (argc >= 4) {
    ref_nb = atoi(argv[1]);
    query_nb = atoi(argv[2]);
    dim = atoi(argv[3]);
	} 

  int c_iterations = 10;
  int i;
  const float precision = 0.001f; // distance error max
  int nb_correct_precisions = 0;
  int nb_correct_indexes = 0;
  float *knn_dist = (float *)malloc(query_nb * k * sizeof(float));
  int *knn_index = (int *)malloc(query_nb * k * sizeof(int));

  // Memory allocation
  ref = (float *)malloc(ref_nb * dim * sizeof(float));
  query = (float *)malloc(query_nb * dim * sizeof(float));
  dist = (float *)malloc(query_nb * ref_nb * sizeof(float));
  ind = (int *)malloc(query_nb * k * sizeof(int));

  dist_c = (float *)malloc(query_nb * k * sizeof(float));
  ind_c = (int *)malloc(query_nb * k * sizeof(float));

  // Init
  srand(time(NULL));
  for (i = 0; i < ref_nb * dim; i++)
    ref[i] = (float)rand() / (float)RAND_MAX;
  for (i = 0; i < query_nb * dim; i++)
    query[i] = (float)rand() / (float)RAND_MAX;

  // printf("Ground truth computation in progress...\n\n");
  // if (!knn_c(ref, ref_nb, query, query_nb, dim, k, knn_dist, knn_index)) {
  //   free(knn_dist);
  //   free(knn_index);
  //   return EXIT_FAILURE;
  // }

  // Variables for duration evaluation
  float elapsed_time;

  // Display informations
  printf("Number of reference points      : %6d\n", ref_nb);
  printf("Number of query points          : %6d\n", query_nb);
  printf("Dimension of points             : %4d\n", dim);
  printf("Number of neighbors to consider : %4d\n", k);
  printf("Processing kNN search           :\n");

  float precision_accuracy = 0.0f;
  float index_accuracy = 0.0f;
  /*
  printf("On CPU: \n");
  struct timeval tic;
  gettimeofday(&tic, NULL);
  for (i = 0; i < c_iterations; i++) {
    knn_c(ref, ref_nb, query, query_nb, dim, k, dist_c, ind_c);
  }

  for (int i = 0; i < query_nb * k; ++i) {
    if (fabs(dist_c[i] - knn_dist[i]) <= precision) {
      nb_correct_precisions++;
    }
    if (ind_c[i] == knn_index[i]) {
      nb_correct_indexes++;
    }
  }

  struct timeval toc;
  gettimeofday(&toc, NULL);
  elapsed_time = toc.tv_sec - tic.tv_sec;
  elapsed_time += (toc.tv_usec - tic.tv_usec) / 1000000.;
  precision_accuracy = nb_correct_precisions / ((float)query_nb * k);
  index_accuracy = nb_correct_indexes / ((float)query_nb * k);
  printf("%f, %f\n", precision_accuracy, index_accuracy);
  printf(" done in %f s for %d iterations (%f s by iteration)\n", elapsed_time,
         c_iterations, elapsed_time / (c_iterations));
  */
  printf("on GPU: \n");

  // Call kNN search CUDA
  GPU_argv_init();

  initTrace();
  startCPU();

  float *ref_device;
  float *query_device;
  float *dist_device;
  int *ind_device;

  cudaMallocManaged(&ref_device, ref_nb * dim * sizeof(float));
  cudaMallocManaged(&query_device, query_nb * dim * sizeof(float));
  cudaMallocManaged(&dist_device, query_nb * ref_nb * sizeof(float));
  cudaMallocManaged(&ind_device, query_nb * k * sizeof(int));

  memcpy(ref_device, ref, ref_nb * dim * sizeof(float));
  memcpy(query_device, query, query_nb * dim * sizeof(float));

  for (i = 0; i < iterations; i++) {
    // knn_cuda(ref, ref_nb, query, query_nb, dim, k, dist, ind);
    knn_cuda(ref_device, ref_nb, query_device, query_nb, dim, k, dist_device, ind_device);
  }

  memcpy(dist, dist_device, query_nb * ref_nb * sizeof(float));
  memcpy(ind, ind_device, query_nb * k * sizeof(int));

  cudaFree(ind_device);
  cudaFree(dist_device);
  cudaFree(query_device);
  cudaFree(ref_device);

  endCPU();
  finiTrace();

  nb_correct_precisions = 0;
  nb_correct_indexes = 0;
  for (int i = 0; i < query_nb * k; ++i) {
    if (fabs(dist[i] - knn_dist[i]) <= precision) {
      nb_correct_precisions++;
    }
    if (ind[i] == knn_index[i]) {
      nb_correct_indexes++;
    }
  }
  
  precision_accuracy = nb_correct_precisions / ((float)query_nb * k);
  index_accuracy = nb_correct_indexes / ((float)query_nb * k);
  printf("%f, %f\n", precision_accuracy, index_accuracy);
  

  // Destroy cuda event object and free memory
  free(ind);
  free(dist);
  free(query);
  free(ref);
  free(dist_c);
  free(ind_c);
}