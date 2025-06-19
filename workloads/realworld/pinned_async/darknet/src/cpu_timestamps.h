#ifndef CPU_TIMESTAMP_
#define CPU_TIMESTAMP_

#include <time.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
static uint64_t startCPUTime;
static uint64_t endCPUTime;

void startCPU();
void endCPU();
#ifdef __cplusplus
}
#endif

#endif
