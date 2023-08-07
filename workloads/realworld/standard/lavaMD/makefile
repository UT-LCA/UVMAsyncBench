include ../../../common/make.config

# Example
# target: dependencies
	# command 1
	# command 2
          # .
          # .
          # .
	# command n
ifdef OUTPUT
override OUTPUT = -DOUTPUT
endif

C_C = gcc
OMP_LIB = -lgomp
OMP_FLAG = -fopenmp

CUD_C = $(CUDA_DIR)/bin/nvcc
# OMP_FLAG = 	-Xcompiler paste_one_here
CUDA_FLAG = -arch sm_80

# link objects (binaries) together
a.out:		main.o \
			./kernel/kernel_gpu_cuda_wrapper.o \
			./util/num/num.o \
			./util/timer/timer.o \
			./util/device/device.o
	$(C_C)	$(KERNEL_DIM) main.o \
			$(CUPTI_ADD_COMMON)/cupti_add.cpp \
			$(CUPTI_ADD_COMMON)/cpu_timestamps.o \
			./kernel/kernel_gpu_cuda_wrapper.o \
			./util/num/num.o \
			./util/timer/timer.o \
			./util/device/device.o \
			-lm \
			-I$(CUDA_DIR)/include \
			-I$(CUPTI_INCLUDE) \
			-L$(CUDA_DIR)/lib64 \
			-L$(CUPTI_LIB_DIR) \
			-lcuda -lcudart -lcupti -lstdc++ \
			$(OMP_LIB) \
			-o lavaMD

# compile function files into objects (binaries)
main.o:		main.h \
			main.c \
			./kernel/kernel_gpu_cuda_wrapper.h \
			./kernel/kernel_gpu_cuda_wrapper.cu \
			./util/num/num.h \
			./util/num/num.c \
			./util/timer/timer.h \
			./util/timer/timer.c \
			./util/device/device.h \
			./util/device/device.cu
	$(C_C)	$(KERNEL_DIM) $(OUTPUT) main.c \
			-c \
			-o main.o \
			-O3

./kernel/kernel_gpu_cuda_wrapper.o:	./kernel/kernel_gpu_cuda_wrapper.h \
									./kernel/kernel_gpu_cuda_wrapper.cu
	$(CUD_C) $(KERNEL_DIM)						./kernel/kernel_gpu_cuda_wrapper.cu \
	-I$(CUPTI_INCLUDE) \
	-L$(CUPTI_LIB_DIR) -lcupti \
									-c \
									-o ./kernel/kernel_gpu_cuda_wrapper.o \
									-O3 \
									$(CUDA_FLAG)

./util/num/num.o:	./util/num/num.h \
					./util/num/num.c
	$(C_C)			./util/num/num.c \
					-c \
					-o ./util/num/num.o \
					-O3

./util/timer/timer.o:	./util/timer/timer.h \
						./util/timer/timer.c
	$(C_C)				./util/timer/timer.c \
						-c \
						-o ./util/timer/timer.o \
						-O3

./util/device/device.o:	./util/device/device.h \
						./util/device/device.cu
	$(CUD_C)			./util/device/device.cu \
						-c \
						-o ./util/device/device.o \
						-O3

# delete all object and executable files
clean:
	rm	*.o \
		./kernel/*.o \
		./util/num/*.o \
		./util/timer/*.o \
		./util/device/*.o \
		lavaMD
