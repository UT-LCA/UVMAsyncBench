#include <cupti.h>
#include <stdio.h>
#include <cxxabi.h>
#include <time.h>
#include "gpu_id.h"

#define BUF_SIZE (32 * 1024)
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                            \
  (((uintptr_t) (buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t) (buffer) & ((align)-1))) : (buffer))

static uint64_t startTimestamp;
// Timestamp at trace initialization time. Used to normalized other
// timestamps

#define CUPTI_CALL(call)                                                \
  do {                                                                  \
    CUptiResult _status = call;                                         \
    if (_status != CUPTI_SUCCESS) {                                     \
      const char *errstr;                                               \
      cuptiGetResultString(_status, &errstr);                           \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", \
              __FILE__, __LINE__, #call, errstr);                       \
      if(_status == CUPTI_ERROR_LEGACY_PROFILER_NOT_SUPPORTED)          \
          exit(0);                                                      \
      else                                                              \
          exit(-1);                                                     \
    }                                                                   \
  } while (0)


extern inline __attribute__((always_inline)) unsigned long rdtsc() {
  unsigned long a, d;

  __asm__ volatile("rdtsc" : "=a"(a), "=d"(d));

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

void initTrace();
void finiTrace();
void GPU_argv_init();
void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords);
void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize);
static void printActivity(CUpti_Activity *record);
