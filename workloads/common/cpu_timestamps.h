#ifndef CPU_TIMESTAMP_
#define CPU_TIMESTAMP_

#include <time.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdint.h>
#include <error.h>

static uint64_t startCPUTime;
static uint64_t endCPUTime;

static uint64_t overlapStartCPUTime = 0;
static uint64_t overlapEndCPUTime = 0;

void startCPU();
void endCPU();

void overlapStartCPU();
void overlapEndCPU();

#endif
