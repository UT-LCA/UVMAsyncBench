#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "dropout_layer.h"
#include "cuda_dark.h"
#include "utils.h"
}

__global__ void yoloswag420blazeit360noscope(float *input, int size, float *rand, float prob, float scale)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id < size) input[id] = (rand[id] < prob) ? 0 : input[id]*scale;
}

void forward_dropout_layer_gpu(dropout_layer layer, network net)
{
    if (!net.train) return;
    int size = layer.inputs*layer.batch;
    cuda_random(layer.rand_gpu, size);
    /*
    int i;
    for(i = 0; i < size; ++i){
        layer.rand[i] = rand_uniform();
    }
    cuda_push_array(layer.rand_gpu, layer.rand, size);
    */
    cudaStream_t stream1;
    cudaStream_t stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    cudaMemPrefetchAsync(net.input_gpu, size * sizeof(float), GPU_DEVICE, stream1);
    cudaStreamSynchronize(stream1);
    cudaMemPrefetchAsync(layer.rand_gpu, size * sizeof(float), GPU_DEVICE, stream2);
    cudaStreamSynchronize(stream2);

    yoloswag420blazeit360noscope<<<cuda_gridsize(size), BLOCK, 0, stream1>>>(net.input_gpu, size, layer.rand_gpu, layer.probability, layer.scale);
    check_error(cudaPeekAtLastError());
}

void backward_dropout_layer_gpu(dropout_layer layer, network net)
{
    if(!net.delta_gpu) return;
    int size = layer.inputs*layer.batch;

    cudaStream_t stream1;
    cudaStream_t stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    cudaMemPrefetchAsync(net.delta_gpu, size * sizeof(float), GPU_DEVICE, stream1);
    cudaStreamSynchronize(stream1);
    cudaMemPrefetchAsync(layer.rand_gpu, size * sizeof(float), GPU_DEVICE, stream2);
    cudaStreamSynchronize(stream2);

    yoloswag420blazeit360noscope<<<cuda_gridsize(size), BLOCK, 0, stream1>>>(net.delta_gpu, size, layer.rand_gpu, layer.probability, layer.scale);
    check_error(cudaPeekAtLastError());
}
