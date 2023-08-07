

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <sys/time.h>
#include "../../../common/cupti_add.h"
#include "../../../common/cpu_timestamps.h"

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

double t_start, t_end;

// includes, kernels
#include "backprop_cuda_kernel.cu"
#include "backprop.h"

////////////////////////////////////////////////////////////////////////////////

extern "C" void bpnn_layerforward(float *l1, float *l2, float **conn, int n1, int n2);

extern "C" void bpnn_output_error(float *delta, float *target, float *output, int nj, float *err);

extern "C" void bpnn_hidden_error(float *delta_h, int nh, float *delta_o, int no, float **who, float *hidden, float *err);

extern "C" void bpnn_adjust_weights(float *delta, int ndelta, float *ly, int nly, float **w, float **oldw);

extern "C" int setup(int argc, char **argv);

extern "C" float **alloc_2d_dbl(int m, int n);

extern "C" float squash(float x);

double gettime()
{
  struct timeval t;
  gettimeofday(&t, NULL);
  return t.tv_sec + t.tv_usec * 1e-6;
}

unsigned int num_threads = 0;
unsigned int num_blocks = 0;

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
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

  initTrace();
  startCPU();

  num_blocks = atoi(argv[2]);
  setup(argc, argv);
}

extern "C" void bpnn_train_cuda(BPNN *net, float *eo, float *eh)
{
  int in, hid, out;
  float out_err, hid_err;

  in = net->input_n;
  hid = net->hidden_n;
  out = net->output_n;

#ifdef GPU
  int m = 0;
  float *input_hidden_cuda;
  float *input_cuda;
  float *output_hidden_cuda;
  float *partial_sum;
  float *hidden_partial_sum;
  float *hidden_delta_cuda;
  float *input_prev_weights_cuda;
  float sum;
  float *input_weights_one_dim;
  float *input_weights_prev_one_dim;
  // ruihao
  // num_blocks = in / 16;
  // dim3 grid(1, num_blocks);
  // dim3 threads(16, 16);

  int tile_size = in / num_blocks;
  dim3 grid(1, num_blocks);
  dim3 threads(16, 16);
  // ruihao

  input_weights_one_dim = (float *)malloc((in + 1) * (hid + 1) * sizeof(float));
  input_weights_prev_one_dim = (float *)malloc((in + 1) * (hid + 1) * sizeof(float));
  // ruihao
  // partial_sum = (float *) malloc(num_blocks * WIDTH * sizeof(float));
  partial_sum = (float *)malloc(in * sizeof(float));
  // ruihao

  // this preprocessing stage is added to correct the bugs of wrong memcopy using two-dimensional net->inputweights
  for (int k = 0; k <= in; k++)
  {
    for (int j = 0; j <= hid; j++)
    {
      input_weights_one_dim[m] = net->input_weights[k][j];
      input_weights_prev_one_dim[m] = net->input_prev_weights[k][j];
      m++;
    }
  }

  // GPU_argv_init();

  // initTrace();
  // startCPU();

  cudaMallocManaged((void **)&input_cuda, (in + 1) * sizeof(float));
  cudaMallocManaged((void **)&output_hidden_cuda, (hid + 1) * sizeof(float));
  cudaMallocManaged((void **)&input_hidden_cuda, (in + 1) * (hid + 1) * sizeof(float));
  // ruihao
  // cudaMalloc((void**) &hidden_partial_sum, num_blocks * WIDTH * sizeof(float));
  cudaMallocManaged((void **)&hidden_partial_sum, in * sizeof(float));
  // ruihao

#endif

#ifdef CPU

  printf("Performing CPU computation\n");
  bpnn_layerforward(net->input_units, net->hidden_units, net->input_weights, in, hid);

#endif

#ifdef GPU

  //printf("Performing GPU computation\n");

  memcpy(input_cuda, net->input_units, (in + 1) * sizeof(float));
  memcpy(input_hidden_cuda, input_weights_one_dim, (in + 1) * (hid + 1) * sizeof(float));

  // ruihao
  //t_start = rtclock();
  // ruihao
  bpnn_layerforward_CUDA<<<grid, threads>>>(input_cuda,
                                            output_hidden_cuda,
                                            input_hidden_cuda,
                                            hidden_partial_sum,
                                            in,
                                            hid,
                                            tile_size);

  cudaDeviceSynchronize();

  // ruihao
  //t_end = rtclock();
  //fprintf(stdout, "bpnn_layerforward_CUDA GPU Runtime: %0.6lfs\n", t_end - t_start);
  memcpy(partial_sum, hidden_partial_sum, in * sizeof(float));
  // ruihao

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess)
  {
    printf("bpnn kernel error: %s\n", cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }

  for (int j = 1; j <= hid; j++)
  {
    sum = 0.0;
    // ruihao
    // for (int k = 0; k < num_blocks; k++) {
    //   sum += partial_sum[k * hid + j-1] ;
    // }
    for (int k = 0; k < in / WIDTH; k++)
    {
      sum += partial_sum[k * hid + j - 1];
    }
    // ruihao
    sum += net->input_weights[0][j];
    net->hidden_units[j] = float(1.0 / (1.0 + exp(-sum)));
  }
#endif

  bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights, hid, out);
  bpnn_output_error(net->output_delta, net->target, net->output_units, out, &out_err);
  bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out, net->hidden_weights, net->hidden_units, &hid_err);
  bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid, net->hidden_weights, net->hidden_prev_weights);

#ifdef CPU

  bpnn_adjust_weights(net->hidden_delta, hid, net->input_units, in, net->input_weights, net->input_prev_weights);

#endif

#ifdef GPU

  cudaMallocManaged((void **)&hidden_delta_cuda, (hid + 1) * sizeof(float));
  cudaMallocManaged((void **)&input_prev_weights_cuda, (in + 1) * (hid + 1) * sizeof(float));
  //  ruihao
  //t_start = rtclock();
  memcpy(hidden_delta_cuda, net->hidden_delta, (hid + 1) * sizeof(float));
  memcpy(input_prev_weights_cuda, input_weights_prev_one_dim, (in + 1) * (hid + 1) * sizeof(float));
  memcpy(input_hidden_cuda, input_weights_one_dim, (in + 1) * (hid + 1) * sizeof(float));
  // ruihao
  bpnn_adjust_weights_cuda<<<grid, threads>>>(hidden_delta_cuda,
                                              hid,
                                              input_cuda,
                                              in,
                                              input_hidden_cuda,
                                              input_prev_weights_cuda,
                                              tile_size);
  // ruihao
  cudaDeviceSynchronize();
  //t_end = rtclock();
  memcpy(net->input_units, input_cuda, (in + 1) * sizeof(float));
  memcpy(input_weights_one_dim, input_hidden_cuda, (in + 1) * (hid + 1) * sizeof(float));
  //fprintf(stdout, "bpnn_adjust_weights_cuda GPU Runtime: %0.6lfs\n", t_end - t_start);
  // ruihao

  cudaFree(input_cuda);
  cudaFree(output_hidden_cuda);
  cudaFree(input_hidden_cuda);
  cudaFree(hidden_partial_sum);
  cudaFree(input_prev_weights_cuda);
  cudaFree(hidden_delta_cuda);

  endCPU();
  finiTrace();

  free(partial_sum);
  free(input_weights_one_dim);
  free(input_weights_prev_one_dim);

#endif
}
