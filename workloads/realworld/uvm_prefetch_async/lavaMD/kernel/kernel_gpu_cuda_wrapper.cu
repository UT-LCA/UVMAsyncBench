//========================================================================================================================================================================================================200
//	DEFINE/INCLUDE
//========================================================================================================================================================================================================200

//======================================================================================================================================================150
//	MAIN FUNCTION HEADER
//======================================================================================================================================================150

#include "./../main.h"								// (in the main program folder)	needed to recognized input parameters

//======================================================================================================================================================150
//	UTILITIES
//======================================================================================================================================================150

#include "./../util/device/device.h"				// (in library path specified to compiler)	needed by for device functions
#include "./../util/timer/timer.h"					// (in library path specified to compiler)	needed by timer

//======================================================================================================================================================150
//	KERNEL_GPU_CUDA_WRAPPER FUNCTION HEADER
//======================================================================================================================================================150

#include "./kernel_gpu_cuda_wrapper.h"				// (in the current directory)

//======================================================================================================================================================150
//	KERNEL
//======================================================================================================================================================150

#include "./kernel_gpu_cuda.cu"						// (in the current directory)	GPU kernel, cannot include with header file because of complications with passing of constant memory variables

//========================================================================================================================================================================================================200
//	KERNEL_GPU_CUDA_WRAPPER FUNCTION
//========================================================================================================================================================================================================200
#include <cupti.h>
#include "../../../../common/cupti_add.h"
#include "../../../../common/cpu_timestamps.h"

#define GPU_DEVICE 6

void GPU_argv_init()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
	printf("setting device %d with name %s\n", GPU_DEVICE, deviceProp.name);
	cudaSetDevice(GPU_DEVICE);
}

void 
kernel_gpu_cuda_wrapper(par_str par_cpu,
						dim_str dim_cpu,
						box_str* box_cpu,
						FOUR_VECTOR* rv_cpu,
						fp* qv_cpu,
						FOUR_VECTOR* fv_cpu,
						int nblocks)
{

	//======================================================================================================================================================150
	//	CPU VARIABLES
	//======================================================================================================================================================150

	// timer
	long long time0;
	long long time1;
	long long time2;
	long long time3;
	long long time4;
	long long time5;
	long long time6;

	time0 = get_time();

	//======================================================================================================================================================150
	//	GPU SETUP
	//======================================================================================================================================================150

	//====================================================================================================100
	//	INITIAL DRIVER OVERHEAD
	//====================================================================================================100
	GPU_argv_init();

	initTrace();
	startCPU();

	cudaThreadSynchronize();

	//====================================================================================================100
	//	VARIABLES
	//====================================================================================================100

	box_str* d_box_gpu;
	FOUR_VECTOR* d_rv_gpu;
	fp* d_qv_gpu;
	FOUR_VECTOR* d_fv_gpu;

	dim3 threads;
	dim3 blocks;

	//====================================================================================================100
	//	EXECUTION PARAMETERS
	//====================================================================================================100

	// blocks.x = dim_cpu.number_boxes;
	blocks.x = nblocks * nblocks * nblocks;
	blocks.y = 1;
	threads.x = NUMBER_THREADS;											// define the number of threads in the block
	threads.y = 1;

	int boxes_per_block = 1;
	if (dim_cpu.number_boxes >= blocks.x)
	{
		boxes_per_block = (dim_cpu.number_boxes + blocks.x - 1) / blocks.x;
	}

	time1 = get_time();

	//======================================================================================================================================================150
	//	GPU MEMORY				(MALLOC)
	//======================================================================================================================================================150

	//====================================================================================================100
	//	GPU MEMORY				(MALLOC) COPY IN
	//====================================================================================================100

	//==================================================50
	//	boxes
	//==================================================50

	cudaMallocManaged(	(void **)&d_box_gpu, 
				dim_cpu.box_mem);

	//==================================================50
	//	rv
	//==================================================50

	cudaMallocManaged((void **)&d_rv_gpu,
					  dim_cpu.space_mem);

	//==================================================50
	//	qv
	//==================================================50

	cudaMallocManaged((void **)&d_qv_gpu,
					  dim_cpu.space_mem2);

	//====================================================================================================100
	//	GPU MEMORY				(MALLOC) COPY
	//====================================================================================================100

	//==================================================50
	//	fv
	//==================================================50

	cudaMallocManaged((void **)&d_fv_gpu,
					  dim_cpu.space_mem);

	time2 = get_time();

	//======================================================================================================================================================150
	//	GPU MEMORY			COPY
	//======================================================================================================================================================150

	//====================================================================================================100
	//	GPU MEMORY				(MALLOC) COPY IN
	//====================================================================================================100

	//==================================================50
	//	boxes
	//==================================================50

	memcpy(d_box_gpu,
			   box_cpu,
			   dim_cpu.box_mem);

	//==================================================50
	//	rv
	//==================================================50

	memcpy(d_rv_gpu,
		   rv_cpu,
		   dim_cpu.space_mem);

	//==================================================50
	//	qv
	//==================================================50

	memcpy(d_qv_gpu,
		   qv_cpu,
		   dim_cpu.space_mem2);

	//====================================================================================================100
	//	GPU MEMORY				(MALLOC) COPY
	//====================================================================================================100

	//==================================================50
	//	fv
	//==================================================50

	memcpy(d_fv_gpu,
		   fv_cpu,
		   dim_cpu.space_mem);

	time3 = get_time();

	cudaStream_t stream1;
	cudaStream_t stream2;
	cudaStream_t stream3;
	cudaStream_t stream4;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);
	cudaStreamCreate(&stream3);
	cudaStreamCreate(&stream4);

	cudaMemPrefetchAsync(d_box_gpu, dim_cpu.box_mem, GPU_DEVICE, stream1);
	cudaStreamSynchronize(stream1);
	cudaMemPrefetchAsync(d_rv_gpu, dim_cpu.space_mem, GPU_DEVICE, stream2);
	cudaStreamSynchronize(stream2);
	cudaMemPrefetchAsync(d_qv_gpu, dim_cpu.space_mem2, GPU_DEVICE, stream3);
	cudaStreamSynchronize(stream3);
	cudaMemPrefetchAsync(d_fv_gpu, dim_cpu.space_mem, GPU_DEVICE, stream4);
	cudaStreamSynchronize(stream4);

	//======================================================================================================================================================150
	//	KERNEL
	//======================================================================================================================================================150
	// launch kernel - all boxes
	kernel_gpu_cuda<<<blocks, threads, 0, stream4>>>(par_cpu,
										 dim_cpu,
										 d_box_gpu,
										 d_rv_gpu,
										 d_qv_gpu,
										 d_fv_gpu,
										 boxes_per_block);

	checkCUDAError("Start");
	cudaDeviceSynchronize();

	time4 = get_time();

	//======================================================================================================================================================150
	//	GPU MEMORY			COPY (CONTD.)
	//======================================================================================================================================================150

	memcpy(fv_cpu,
			   d_fv_gpu,
			   dim_cpu.space_mem);

	time5 = get_time();

	//======================================================================================================================================================150
	//	GPU MEMORY DEALLOCATION
	//======================================================================================================================================================150

	cudaFree(d_rv_gpu);
	cudaFree(d_qv_gpu);
	cudaFree(d_fv_gpu);
	cudaFree(d_box_gpu);

	endCPU();
	finiTrace();

	time6 = get_time();

	//======================================================================================================================================================150
	//	DISPLAY TIMING
	//======================================================================================================================================================150

	printf("Time spent in different stages of GPU_CUDA KERNEL:\n");

	printf("%15.12f s, %15.12f % : GPU: SET DEVICE / DRIVER INIT\n", (float)(time1 - time0) / 1000000, (float)(time1 - time0) / (float)(time6 - time0) * 100);
	printf("%15.12f s, %15.12f % : GPU MEM: ALO\n", (float)(time2 - time1) / 1000000, (float)(time2 - time1) / (float)(time6 - time0) * 100);
	printf("%15.12f s, %15.12f % : GPU MEM: COPY IN\n", (float)(time3 - time2) / 1000000, (float)(time3 - time2) / (float)(time6 - time0) * 100);

	printf("%15.12f s, %15.12f % : GPU: KERNEL\n", (float)(time4 - time3) / 1000000, (float)(time4 - time3) / (float)(time6 - time0) * 100);

	printf("%15.12f s, %15.12f % : GPU MEM: COPY OUT\n", (float)(time5 - time4) / 1000000, (float)(time5 - time4) / (float)(time6 - time0) * 100);
	printf("%15.12f s, %15.12f % : GPU MEM: FRE\n", (float)(time6 - time5) / 1000000, (float)(time6 - time5) / (float)(time6 - time0) * 100);

	printf("Total time:\n");
	printf("%.12f s\n", (float)(time6 - time0) / 1000000);
}
