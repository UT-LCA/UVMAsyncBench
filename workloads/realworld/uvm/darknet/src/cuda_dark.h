#ifndef CUDA_H
#define CUDA_H

#include "darknet.h"

#ifdef GPU

#define BUF_SIZE (32 * 1024)
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align) (((uintptr_t)(buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t)(buffer) & ((align)-1))) : (buffer))

static uint64_t startTimestamp;
// Timestamp at trace initialization time. Used to normalized other
// timestamps

#define CUPTI_CALL(call)                                                         \
    do                                                                           \
    {                                                                            \
        CUptiResult _status = call;                                              \
        if (_status != CUPTI_SUCCESS)                                            \
        {                                                                        \
            const char *errstr;                                                  \
            cuptiGetResultString(_status, &errstr);                              \
            fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", \
                    __FILE__, __LINE__, #call, errstr);                          \
            if (_status == CUPTI_ERROR_LEGACY_PROFILER_NOT_SUPPORTED)            \
                exit(0);                                                         \
            else                                                                 \
                exit(-1);                                                        \
        }                                                                        \
    } while (0)

#include <cupti.h>

#ifdef __cplusplus
extern "C" {
#endif
void check_error(cudaError_t status);
cublasHandle_t blas_handle();
int *cuda_make_int_array(int *x, size_t n);
void cuda_random(float *x_gpu, size_t n);
float cuda_compare(float *x_gpu, float *x, size_t n, char *s);
dim3 cuda_gridsize(size_t n);

void GPU_argv_init();
void initTrace();
void finiTrace();
void startCPU();
void endCPU();
void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords);
void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize);
static void printActivity(CUpti_Activity *record);

#ifdef __cplusplus
}
#endif

#ifdef CUDNN
cudnnHandle_t cudnn_handle();
#endif

#endif
#endif
