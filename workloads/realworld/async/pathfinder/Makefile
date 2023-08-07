include ../../../common/make.config

CC := $(CUDA_DIR)/bin/nvcc
INCLUDE := $(CUDA_DIR)/include

SRC = pathfinder.cu $(CUPTI_ADD_COMMON)/cupti_add.cpp $(CUPTI_ADD_COMMON)/cpu_timestamps.cpp

EXE = pathfinder

release:
	$(CC) $(SRC) -o $(EXE) -I$(INCLUDE) -I$(CUPTI_INCLUDE) -L$(CUDA_LIB_DIR) -L$(CUPTI_LIB_DIR) -lcupti

clean:
	rm -f pathfinder
