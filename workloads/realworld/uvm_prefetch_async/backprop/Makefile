include ../../../common/make.config

# C compiler
CC = gcc
CC_FLAGS = -g  -O2

# CUDA compiler
NVCC = $(CUDA_DIR)/bin/nvcc
NVCC_FLAGS = -I$(CUDA_DIR)/include -I$(CUPTI_INCLUDE) -arch=sm_80 

# 'make dbg=1' enables NVCC debugging
ifeq ($(dbg),1)
	NVCC_FLAGS += -g -O0
else
	NVCC_FLAGS += -O2
endif

# 'make emu=1' compiles the CUDA kernels for emulation
ifeq ($(emu),1)
	NVCC_FLAGS += -deviceemu
endif


backprop: backprop.o facetrain.o imagenet.o backprop_cuda.o $(CUPTI_ADD_COMMON)/cupti_add.cpp $(CUPTI_ADD_COMMON)/cpu_timestamps.cpp
	$(NVCC) $(NVCC_FLAGS) backprop.o facetrain.o imagenet.o backprop_cuda.o $(CUPTI_ADD_COMMON)/cupti_add.cpp $(CUPTI_ADD_COMMON)/cpu_timestamps.cpp -o backprop $(NVCC_FLAGS) -L$(CUDA_LIB_DIR) -L$(CUPTI_LIB_DIR) -lcuda -lcudart -lcupti -lm -lstdc++

# backprop: backprop.o facetrain.o imagenet.o backprop_cuda.o $(CUPTI_ADD_COMMON)/cupti_add.cpp $(CUPTI_ADD_COMMON)/cpu_timestamps.cpp
# 	$(CC) $(CC_FLAGS) backprop.o facetrain.o imagenet.o backprop_cuda.o $(CUPTI_ADD_COMMON)/cupti_add.cpp $(CUPTI_ADD_COMMON)/cpu_timestamps.cpp -o backprop -L$(CUDA_LIB_DIR) -L$(CUPTI_LIB_DIR) -lcuda -lcudart -lcupti -lm -lstdc++

%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c

facetrain.o: facetrain.c backprop.h
	$(CC) $(CC_FLAGS) facetrain.c -c
	
backprop.o: backprop.c backprop.h
	$(CC) $(CC_FLAGS) backprop.c -c

backprop_cuda.o: backprop_cuda.cu backprop.h $(CUPTI_ADD_COMMON)/cupti_add.h $(CUPTI_ADD_COMMON)/cupti_add.cpp $(CUPTI_ADD_COMMON)/cpu_timestamps.cpp
	$(NVCC) $(NVCC_FLAGS) -c backprop_cuda.cu $(CUPTI_ADD_COMMON)/cupti_add.cpp $(CUPTI_ADD_COMMON)/cpu_timestamps.cpp

imagenet.o: imagenet.c backprop.h
	$(CC) $(CC_FLAGS) imagenet.c -c


clean:
	rm -f *.o *~ backprop backprop_cuda.linkinfo
