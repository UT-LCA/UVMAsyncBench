include ../../../common/make.config

NVCCCFLAGS = -I$(CUPTI_INCLUDE) -L$(CUPTI_LIB_DIR) -std=c++11 -lcuda -lcupti -arch=sm_80 -O3
NVCC = $(CUDA_DIR)/bin/nvcc

EXECUTABLE := lud
CUFILES := lud_cuda.cu $(CUPTI_ADD_COMMON)/cupti_add.cpp $(CUPTI_ADD_COMMON)/cpu_timestamps.cpp

all:
	$(NVCC) ${NVCCCFLAGS} ${CUFILES} ${DEF} -o ${EXECUTABLE}
clean:
	rm -f *.o lud

