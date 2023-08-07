include ../../../../common/make.config
CC = gcc
NVCC = nvcc

DEFS += \
		-DGPU_TIMER \
		$(SPACE)

NVCCFLAGS += -I../common \
-I$(CUPTI_INCLUDE) -I$(CUDA_DIR)/include \
			 -O3 \
			 -use_fast_math \
			 -arch=sm_80 \
			 -lm \
			 $(SPACE)

CFLAGS += -I../common \
-I$(CUPTI_INCLUDE) -I$(CUDA_DIR)/include \
					-I/usr/include/cuda \
		  -O3 \
		  -Wall \
		  $(SPACE)

# Add source files here
EXECUTABLE  := lud_cuda
# Cuda source files (compiled with cudacc)
CUFILES     := lud_kernel.cu
# C/C++ source files (compiled with gcc / c++)
CCFILES     := lud.c lud_cuda.c ../common/common.c

OBJS = ../common/common.o lud.o lud_kernel.o

.PHONY: all clean 
all : $(EXECUTABLE)

.c.o : 
	$(NVCC) $(KERNEL_DIM) $(NVCCFLAGS) -I$(CUPTI_INCLUDE) -I$(CUDA_DIR)/include -L$(CUPTI_LIB_DIR) -lcupti $(DEFS) -o $@ -c $<

%.o:	%.cu 
	$(NVCC) $(KERNEL_DIM) $(NVCCFLAGS) -I$(CUPTI_INCLUDE) -I$(CUDA_DIR)/include -L$(CUPTI_LIB_DIR) -lcupti $(DEFS) -o $@ -c $<

clang: $(SRC)
	clang++ lud.cu lud_kernel.cu ../common/common.c -o $(EXECUTABLE) \
		-I../common -I../../util -I$(CUPTI_INCLUDE) -I$(CUDA_DIR)/include --cuda-gpu-arch=sm_80 \
		-L/usr/local/cuda/lib64 -L$(CUPTI_LIB_DIR) -L$(CUDA_DIR)/lib64 -lcudart_static -lcupti -ldl -lrt -pthread -DTIMING

$(EXECUTABLE) : $(OBJS)
	$(NVCC) $(NVCCFLAGS) -I$(CUPTI_INCLUDE) -I$(CUDA_DIR)/include -L$(CUPTI_LIB_DIR) -L$(CUDA_DIR)/lib64 -lcupti -o $@ $? $(CUPTI_ADD_COMMON)/cpu_timestamps.o $(CUPTI_ADD_COMMON)/cupti_add.cpp

clean:
	rm -f $(EXECUTABLE) $(OBJS) *.linkinfo
