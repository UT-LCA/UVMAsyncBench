include ./make.config

NVCC = $(CUDA_DIR)/bin/nvcc
NVCC_FLAGS = -I$(CUDA_DIR)/include -I$(CUPTI_INCLUDE)

all: cpu_timestapms.o cupti_add.o

cpu_timestapms.o: cpu_timestamps.cpp
	$(NVCC) $(NVCC_FLAGS) -c cpu_timestamps.cpp 

cupti_add.o: cupti_add.cpp
	$(NVCC) $(NVCC_FLAGS) -c cupti_add.cpp 
