include ../../../common/make.config

# C compiler
CC = gcc
CC_FLAGS = -g -fopenmp -O2 

# CUDA compiler
NVCC = $(CUDA_DIR)/bin/nvcc
NVCC_FLAGS = -I$(CUDA_DIR)/include -I$(CUPTI_INCLUDE)

# 'make dbg=1' enables NVCC debugging


# 'make emu=1' compiles the CUDA kernels for emulation
ifeq ($(emu),1)
	NVCC_FLAGS += -deviceemu
endif


kmeans: cluster.o getopt.o kmeans.o kmeans_clustering.o kmeans_cuda.o rmse.o $(CUPTI_ADD_COMMON)/cpu_timestamps.o
	$(CC) $(CC_FLAGS) cluster.o getopt.o kmeans.o kmeans_clustering.o kmeans_cuda.o rmse.o $(CUPTI_ADD_COMMON)/cpu_timestamps.o $(CUPTI_ADD_COMMON)/cupti_add.cpp -o kmeans $(NVCC_FLAGS) -L$(CUDA_LIB_DIR) -L$(CUPTI_LIB_DIR) -lcuda -lcudart -lcupti -lm -lstdc++

kmeans.o: kmeans.c
	$(CC) $(CC_FLAGS) $< -c $(NVCC_FLAGS) -I$(CUPTI_INCLUDE) -L$(CUDA_LIB_DIR) -L$(CUPTI_LIB_DIR) -lcuda -lcupti

%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c -I$(CUDA_DIR)/include -I$(CUPTI_INCLUDE) -L$(CUDA_LIB_DIR) -L$(CUPTI_LIB_DIR) -lcuda -lcudart -lcupti -lstdc++

kmeans_cuda.o: kmeans_cuda.cu $(CUPTI_ADD_COMMON)/cupti_add.cpp
	$(NVCC) -O2 -c kmeans_cuda.cu $(CUPTI_ADD_COMMON)/cupti_add.cpp $(NVCC_FLAGS) -I$(CUPTI_INCLUDE) -L$(CUDA_LIB_DIR) -L$(CUPTI_LIB_DIR) -lcuda -lcupti 

clean:
	rm -f *.o *~ kmeans kmeans_cuda.linkinfo
