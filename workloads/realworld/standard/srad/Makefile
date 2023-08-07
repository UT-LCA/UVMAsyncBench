include ../../../common/make.config

CC := $(CUDA_DIR)/bin/nvcc

INCLUDE := $(CUDA_DIR)/include -I$(CUPTI_INCLUDE) -std=c++11 -arch=sm_80 -O3

SRC = srad.cu $(CUPTI_ADD_COMMON)/cupti_add.cpp $(CUPTI_ADD_COMMON)/cpu_timestamps.cpp 

EXE = srad

release: $(SRC)
	$(CC) $(KERNEL_DIM) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -L$(CUPTI_LIB_DIR) -lcuda -lcupti

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result.txt
