int gpu_index = 0;

#ifdef GPU

#include "cuda.h"
#include "utils.h"
#include "blas.h"
#include <assert.h>
#include <stdlib.h>
#include <time.h>


#include <cxxabi.h>

void cuda_set_device(int n)
{
    gpu_index = n;
    cudaError_t status = cudaSetDevice(n);
    check_error(status);
}

int cuda_get_device()
{
    int n = 0;
    cudaError_t status = cudaGetDevice(&n);
    check_error(status);
    return n;
}

void check_error(cudaError_t status)
{
    cudaDeviceSynchronize();
    cudaError_t status2 = cudaGetLastError();
    if (status != cudaSuccess)
    {   
        const char *s = cudaGetErrorString(status);
        char buffer[256];
        printf("CUDA Error: %s\n", s);
        assert(0);
        snprintf(buffer, 256, "CUDA Error: %s", s);
        error(buffer);
    } 
    if (status2 != cudaSuccess)
    {   
        const char *s = cudaGetErrorString(status);
        char buffer[256];
        printf("CUDA Error Prev: %s\n", s);
        assert(0);
        snprintf(buffer, 256, "CUDA Error Prev: %s", s);
        error(buffer);
    } 
}

dim3 cuda_gridsize(size_t n){
    size_t k = (n-1) / BLOCK + 1;
    size_t x = k;
    size_t y = 1;
    if(x > 65535){
        x = ceil(sqrt(k));
        y = (n-1)/(x*BLOCK) + 1;
    }
    dim3 d = {x, y, 1};
    //printf("%ld %ld %ld %ld\n", n, x, y, x*y*BLOCK);
    return d;
}

#ifdef CUDNN
cudnnHandle_t cudnn_handle()
{
    static int init[16] = {0};
    static cudnnHandle_t handle[16];
    int i = cuda_get_device();
    if(!init[i]) {
        cudnnCreate(&handle[i]);
        init[i] = 1;
    }
    return handle[i];
}
#endif

cublasHandle_t blas_handle()
{
    static int init[16] = {0};
    static cublasHandle_t handle[16];
    int i = cuda_get_device();
    if(!init[i]) {
        cublasCreate(&handle[i]);
        init[i] = 1;
    }
    return handle[i];
}

float *cuda_make_array(float *x, size_t n)
{
    float *x_gpu;
    size_t size = sizeof(float)*n;
    cudaError_t status = cudaMalloc((void **)&x_gpu, size);
    check_error(status);
    if(x){
        status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
        check_error(status);
    } else {
        fill_gpu(n, 0, x_gpu, 1);
    }
    if(!x_gpu) error("Cuda malloc failed\n");
    return x_gpu;
}

void cuda_random(float *x_gpu, size_t n)
{
    static curandGenerator_t gen[16];
    static int init[16] = {0};
    int i = cuda_get_device();
    if(!init[i]){
        curandCreateGenerator(&gen[i], CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen[i], time(0));
        init[i] = 1;
    }
    curandGenerateUniform(gen[i], x_gpu, n);
    check_error(cudaPeekAtLastError());
}

float cuda_compare(float *x_gpu, float *x, size_t n, char *s)
{
    float *tmp = (float *) calloc(n, sizeof(float));
    cuda_pull_array(x_gpu, tmp, n);
    //int i;
    //for(i = 0; i < n; ++i) printf("%f %f\n", tmp[i], x[i]);
    axpy_cpu(n, -1, x, 1, tmp, 1);
    float err = dot_cpu(n, tmp, 1, tmp, 1);
    printf("Error %s: %f\n", s, sqrt(err/n));
    free(tmp);
    return err;
}

int *cuda_make_int_array(int *x, size_t n)
{
    int *x_gpu;
    size_t size = sizeof(int)*n;
    cudaError_t status = cudaMalloc((void **)&x_gpu, size);
    check_error(status);
    if(x){
        status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
        check_error(status);
    }
    if(!x_gpu) error("Cuda malloc failed\n");
    return x_gpu;
}

void cuda_free(float *x_gpu)
{
    cudaError_t status = cudaFree(x_gpu);
    check_error(status);
}

void cuda_push_array(float *x_gpu, float *x, size_t n)
{
    size_t size = sizeof(float)*n;
    cudaError_t status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
    check_error(status);
}

void cuda_pull_array(float *x_gpu, float *x, size_t n)
{
    size_t size = sizeof(float)*n;
    cudaError_t status = cudaMemcpy(x, x_gpu, size, cudaMemcpyDeviceToHost);
    check_error(status);
}

float cuda_mag_array(float *x_gpu, size_t n)
{
    float *temp = (float *) calloc(n, sizeof(float));
    cuda_pull_array(x_gpu, temp, n);
    float m = mag_array(temp, n);
    free(temp);
    return m;
}

static const char *
getMemcpyKindString(CUpti_ActivityMemcpyKind kind)
{
    switch (kind)
    {
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
        return "HtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
        return "DtoH";
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOA:
        return "HtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOH:
        return "AtoH";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOA:
        return "AtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOD:
        return "AtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOA:
        return "DtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
        return "DtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH:
        return "HtoH";
    default:
        break;
    }

    return "<unknown>";
}

static const char *
getUvmCounterKindString(CUpti_ActivityUnifiedMemoryCounterKind kind)
{
    switch (kind)
    {
    case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD:
        return "BYTES_TRANSFER_HTOD";
    case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH:
        return "BYTES_TRANSFER_DTOH";
    default:
        break;
    }
    return "<unknown>";
}

static void
printActivity(CUpti_Activity *record)
{
    switch (record->kind)
    {
    case CUPTI_ACTIVITY_KIND_KERNEL:
    {
        int status;
        CUpti_ActivityKernel4 *kernel = (CUpti_ActivityKernel4 *)record;
        printf("KERNEL %s, %llu, %llu, %llu\n",
               abi::__cxa_demangle(kernel->name, 0, 0, &status),
               (unsigned long long)(kernel->start),
               (unsigned long long)(kernel->end),
               (unsigned long long)(kernel->end) - (kernel->start));
        break;
    }
    case CUPTI_ACTIVITY_KIND_RUNTIME:
    {
        CUpti_ActivityAPI *api = (CUpti_ActivityAPI *)record;
        const char *callback_name;
        cuptiGetCallbackName(CUPTI_CB_DOMAIN_RUNTIME_API, api->cbid, &callback_name);
        // printf("RUNTIME %s (cbid=%u) [ %llu - %llu ] process %u, thread %u, correlation %u\n",
        //        callback_name, api->cbid,
        //        (unsigned long long)(api->start - startTimestamp),
        //        (unsigned long long)(api->end - startTimestamp),
        //        api->processId, api->threadId, api->correlationId);
        printf("RUNTIME %s (cbid=%u), %llu,%llu,%llu, process %u, thread %u, correlation %u\n",
               callback_name, api->cbid,
               (unsigned long long)(api->start),
               (unsigned long long)(api->end),
               (unsigned long long)(api->end - api->start),
               api->processId, api->threadId, api->correlationId);
        break;
    }
    case CUPTI_ACTIVITY_KIND_MEMCPY:
    {
        CUpti_ActivityMemcpy4 *memcpy = (CUpti_ActivityMemcpy4 *)record;
        printf("MEMCPY %s, size %llu, %llu, %llu, %llu\n",
               getMemcpyKindString((CUpti_ActivityMemcpyKind)memcpy->copyKind),
               (unsigned long long)memcpy->bytes,
               (unsigned long long)(memcpy->start),
               (unsigned long long)(memcpy->end),
               (unsigned long long)(memcpy->end) - (memcpy->start));
        break;
    }
    case CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER:
    {
        CUpti_ActivityUnifiedMemoryCounter2 *uvm = (CUpti_ActivityUnifiedMemoryCounter2 *)record;
        printf("UVM MEMCPY %s, size %llu, %llu, %llu, %llu \n",
               getUvmCounterKindString(uvm->counterKind),
               (unsigned long long)uvm->value,
               (unsigned long long)(uvm->start),
               (unsigned long long)(uvm->end),
               (unsigned long long)(uvm->end - uvm->start));
        break;
    }
    }
}

void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords)
{
    uint8_t *bfr = (uint8_t *)malloc(BUF_SIZE + ALIGN_SIZE);
    if (bfr == NULL)
    {
        printf("Error: out of memory\n");
        exit(-1);
    }

    *size = BUF_SIZE;
    *buffer = ALIGN_BUFFER(bfr, ALIGN_SIZE);
    *maxNumRecords = 0;
}

void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize)
{
    CUptiResult status;
    CUpti_Activity *record = NULL;
    if (validSize > 0)
    {
        do
        {
            status = cuptiActivityGetNextRecord(buffer, validSize, &record);
            if (status == CUPTI_SUCCESS)
            {
                printActivity(record);
            }
            else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED)
                break;
            else
            {
                CUPTI_CALL(status);
            }
        } while (1);

        // report any records dropped from the queue
        size_t dropped;
        CUPTI_CALL(cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped));
        if (dropped != 0)
        {
            printf("Dropped %u activity records\n", (unsigned int)dropped);
        }
    }

    free(buffer);
}

// void initTrace() {
//     return;
// }

// void finiTrace() {
//     return;
// }

void initTrace()
{
    size_t attrValue = 0, attrValueSize = sizeof(size_t);

    CUpti_ActivityUnifiedMemoryCounterConfig config[2];

    // configure unified memory counters
    config[0].scope = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_PROCESS_SINGLE_DEVICE;
    config[0].kind = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD;
    config[0].deviceId = 0;
    config[0].enable = 1;

    config[1].scope = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_PROCESS_SINGLE_DEVICE;
    config[1].kind = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH;
    config[1].deviceId = 0;
    config[1].enable = 1;

    CUptiResult res = cuptiActivityConfigureUnifiedMemoryCounter(config, 2);
    if (res == CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED)
    {
        printf("Test is waived, unified memory is not supported on the underlying platform.\n");
    }
    else if (res == CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_ON_DEVICE)
    {
        printf("Test is waived, unified memory is not supported on the device.\n");
    }
    else if (res == CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_ON_NON_P2P_DEVICES)
    {
        printf("Test is waived, unified memory is not supported on the non-P2P multi-gpu setup.\n");
    }
    else
    {
        CUPTI_CALL(res);
    }

    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER));

    // Register callbacks for buffer requests and for buffers completed by CUPTI.
    CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));

    // Optionally get and set activity attributes.
    // Attributes can be set by the CUPTI client to change behavior of the activity API.
    // Some attributes require to be set before any CUDA context is created to be effective,
    // e.g. to be applied to all device buffer allocations (see documentation).
    CUPTI_CALL(cuptiActivityGetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE, &attrValueSize, &attrValue));
    printf("%s = %llu B\n", "CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE", (long long unsigned)attrValue);
    attrValue *= 2;
    CUPTI_CALL(cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE, &attrValueSize, &attrValue));

    CUPTI_CALL(cuptiActivityGetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT, &attrValueSize, &attrValue));
    printf("%s = %llu\n", "CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT", (long long unsigned)attrValue);
    attrValue *= 2;
    CUPTI_CALL(cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT, &attrValueSize, &attrValue));

    CUPTI_CALL(cuptiGetTimestamp(&startTimestamp));
}

void finiTrace()
{
    // Force flush any remaining activity buffers before termination of the application
    CUPTI_CALL(cuptiActivityFlushAll(1));
}


void GPU_argv_init()
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
    printf("setting device %d with name %s\n", GPU_DEVICE, deviceProp.name);
    cudaSetDevice(GPU_DEVICE);
}
#else
void cuda_set_device(int n){}

#endif
