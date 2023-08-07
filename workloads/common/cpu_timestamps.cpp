#include "cpu_timestamps.h"

void startCPU() {
  struct timespec tv;
  if(clock_gettime(CLOCK_REALTIME, &tv))
    printf("error clock_gettime\n");
  startCPUTime = (tv.tv_sec * 1.0e9 + tv.tv_nsec);
}



void endCPU() {
  struct timespec tv;
  if(clock_gettime(CLOCK_REALTIME, &tv))
    printf("error clock_gettime\n");

  endCPUTime =  (tv.tv_sec * 1.0e9 + tv.tv_nsec);
  //endCPUTimestamp1 = std::chrono::system_clock::now();
  printf("CPU_Times,%lu,%lu,%lu\n", startCPUTime, endCPUTime, endCPUTime-startCPUTime);
  printf("Overlap_Times,%lu,%lu,%lu\n", overlapStartCPUTime, overlapEndCPUTime, overlapEndCPUTime - overlapStartCPUTime);
}

void overlapStartCPU()
{
  struct timespec tv;
  if (clock_gettime(CLOCK_REALTIME, &tv))
    printf("error clock_gettime\n");
  overlapStartCPUTime = (tv.tv_sec * 1.0e9 + tv.tv_nsec);
}

void overlapEndCPU()
{
  struct timespec tv;
  if (clock_gettime(CLOCK_REALTIME, &tv))
    printf("error clock_gettime\n");

  overlapEndCPUTime = (tv.tv_sec * 1.0e9 + tv.tv_nsec);
}
