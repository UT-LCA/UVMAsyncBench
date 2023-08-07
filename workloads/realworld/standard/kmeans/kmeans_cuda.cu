#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <omp.h>

#include <cuda.h>

#define THREADS_PER_DIM 16
#define BLOCKS_PER_DIM 64
#define THREADS_PER_BLOCK THREADS_PER_DIM*THREADS_PER_DIM
#include <cupti.h>
#include "kmeans_cuda_kernel.cu"
#include "../../../common/cupti_add.h"
#include "../../../common/cpu_timestamps.h"


//#define BLOCK_DELTA_REDUCE
//#define BLOCK_CENTER_REDUCE

#define CPU_DELTA_REDUCE
#define CPU_CENTER_REDUCE

extern "C"
int setup(int argc, char** argv);									/* function prototype */

// GLOBAL!!!!!
unsigned int num_threads_perdim = THREADS_PER_DIM;					/* sqrt(256) -- see references for this choice */
unsigned int num_blocks_perdim = BLOCKS_PER_DIM;					/* temporary */
unsigned int num_threads = num_threads_perdim*num_threads_perdim;	/* number of threads */
unsigned int num_blocks = num_blocks_perdim*num_blocks_perdim;		/* number of blocks */

/* _d denotes it resides on the device */
int    *membership_new;												/* newly assignment membership */
float  *feature_d;													/* inverted data array */
float  *feature_flipped_d;											/* original (not inverted) data array */
int    *membership_d;												/* membership on the device */
float  *block_new_centers;											/* sum of points in a cluster (per block) */
float  *clusters_d;													/* cluster centers on the device */
float  *block_clusters_d;											/* per block calculation of cluster centers */
int    *block_deltas_d;												/* per block calculation of deltas */


/* -------------- allocateMemory() ------------------- */
/* allocate device memory, calculate number of blocks and threads, and invert the data array */
extern "C"
void allocateMemory(int npoints, int nfeatures, int nclusters, float **features)
{	
	// printf("npoints is %d, num_threads is %d\n", npoints, num_threads);
	// num_blocks = npoints / num_threads;
	// if (npoints % num_threads > 0)		/* defeat truncation */
	// 	num_blocks++;

	// num_blocks_perdim = sqrt((double) num_blocks);
	// while (num_blocks_perdim * num_blocks_perdim < num_blocks)	// defeat truncation (should run once)
	// 	num_blocks_perdim++;

	num_blocks = num_blocks_perdim*num_blocks_perdim;

	/* allocate memory for memory_new[] and initialize to -1 (host) */
	membership_new = (int*) malloc(npoints * sizeof(int));
	for(int i=0;i<npoints;i++) {
		membership_new[i] = -1;
	}

	/* allocate memory for block_new_centers[] (host) */
	block_new_centers = (float *) malloc(nclusters*nfeatures*sizeof(float));
	
	startCPU();
	/* allocate memory for feature_flipped_d[][], feature_d[][] (device) */
	cudaMalloc((void**) &feature_flipped_d, npoints*nfeatures*sizeof(float));
	cudaMemcpy(feature_flipped_d, features[0], npoints*nfeatures*sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc((void**) &feature_d, npoints*nfeatures*sizeof(float));
		
	/* invert the data array (kernel execution) */	
	invert_mapping<<<num_blocks, num_threads>>>(feature_flipped_d, feature_d, npoints, (num_blocks_perdim * num_blocks_perdim * num_threads_perdim * num_threads_perdim), nfeatures);
		
	/* allocate memory for membership_d[] and clusters_d[][] (device) */
	cudaMalloc((void**) &membership_d, npoints*sizeof(int));
	cudaMalloc((void**) &clusters_d, nclusters*nfeatures*sizeof(float));
}
/* -------------- allocateMemory() end ------------------- */

/* -------------- deallocateMemory() ------------------- */
/* free host and device memory */
extern "C"
void deallocateMemory()
{
	free(membership_new);
	free(block_new_centers);
	cudaFree(feature_d);
	cudaFree(feature_flipped_d);
	cudaFree(membership_d);

	cudaFree(clusters_d);

	endCPU();
}
/* -------------- deallocateMemory() end ------------------- */

extern inline __attribute__((always_inline)) unsigned long rdtsc()
{
	   unsigned long a, d;

	      __asm__ volatile("rdtsc" : "=a" (a), "=d" (d));

	         return (a | (d << 32));
}

extern inline __attribute__((always_inline)) unsigned long rdtsp() {
		struct timespec tms;
		    if (clock_gettime(CLOCK_REALTIME, &tms)) {
			            return -1;
				        }
		        unsigned long ns = tms.tv_sec * 1000000000;
			    ns += tms.tv_nsec;
			        return ns;
}

#define GPU_DEVICE 6

void GPU_argv_init()
{
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
  printf("setting device %d with name %s\n", GPU_DEVICE, deviceProp.name);
  cudaSetDevice(GPU_DEVICE);
}
////////////////////////////////////////////////////////////////////////////////
// Program main																  //

int
main( int argc, char** argv) 
{
	uint64_t start_tsc = rdtsc();
	uint64_t start_tsp = rdtsp();
	printf("start_tsc %llu start_tsp %llu\n", start_tsc, start_tsp);
	// make sure we're running on the big card
	GPU_argv_init();
	// as done in the CUDA start/help document provided
	initTrace();
	setup(argc, argv);    
	finiTrace();
}

//																			  //
////////////////////////////////////////////////////////////////////////////////

/* ------------------- kmeansCuda() ------------------------ */    
extern "C"
int	// delta -- had problems when return value was of float type
kmeansCuda(float  **feature,				/* in: [npoints][nfeatures] */
           int      nfeatures,				/* number of attributes for each point */
           int      npoints,				/* number of data points */
           int      nclusters,				/* number of clusters */
           int     *membership,				/* which cluster the point belongs to */
		   float  **clusters,				/* coordinates of cluster centers */
		   int     *new_centers_len,		/* number of elements in each cluster */
           float  **new_centers				/* sum of elements in each cluster */
		   )
{
	int delta = 0;			/* if point has moved */
	int i,j;				/* counters */

	// cudaSetDevice(1);
	

	/* copy membership (host to device) */
	cudaMemcpy(membership_d, membership_new, npoints*sizeof(int), cudaMemcpyHostToDevice);

	// /* copy clusters (host to device) */
	// cudaMemcpy(clusters_d, clusters[0], nclusters*nfeatures*sizeof(float), cudaMemcpyHostToDevice);

	// /* set up texture */
    // cudaChannelFormatDesc chDesc0 = cudaCreateChannelDesc<float>();
    // t_features.filterMode = cudaFilterModePoint;   
    // t_features.normalized = false;
    // t_features.channelDesc = chDesc0;

	// if(cudaBindTexture(NULL, &t_features, feature_d, &chDesc0, npoints*nfeatures*sizeof(float)) != CUDA_SUCCESS)
    //     printf("Couldn't bind features array to texture!\n");

	// cudaChannelFormatDesc chDesc1 = cudaCreateChannelDesc<float>();
    // t_features_flipped.filterMode = cudaFilterModePoint;   
    // t_features_flipped.normalized = false;
    // t_features_flipped.channelDesc = chDesc1;

	// if(cudaBindTexture(NULL, &t_features_flipped, feature_flipped_d, &chDesc1, npoints*nfeatures*sizeof(float)) != CUDA_SUCCESS)
    //     printf("Couldn't bind features_flipped array to texture!\n");

	// cudaChannelFormatDesc chDesc2 = cudaCreateChannelDesc<float>();
    // t_clusters.filterMode = cudaFilterModePoint;   
    // t_clusters.normalized = false;
    // t_clusters.channelDesc = chDesc2;

	// if(cudaBindTexture(NULL, &t_clusters, clusters_d, &chDesc2, nclusters*nfeatures*sizeof(float)) != CUDA_SUCCESS)
    //     printf("Couldn't bind clusters array to texture!\n");

	// /* copy clusters to constant memory */
	// cudaMemcpyToSymbol("c_clusters",clusters[0],nclusters*nfeatures*sizeof(float),0,cudaMemcpyHostToDevice);

	// cudaMemcpy(feature_d, feature, npoints * nfeatures * sizeof(float), cudaMemcpyHostToDevice);

	/* setup execution parameters.
	   changed to 2d (source code on NVIDIA CUDA Programming Guide) */
    dim3  grid( num_blocks_perdim, num_blocks_perdim );
    dim3  threads( num_threads_perdim*num_threads_perdim );
		/* execute the kernel */
		kmeansPoint<<<grid, threads>>>(feature_d,
									   nfeatures,
									   npoints,
									   (num_blocks_perdim * num_blocks_perdim * num_threads_perdim * num_threads_perdim),
									   nclusters,
									   membership_d);
		cudaDeviceSynchronize();

		/* copy back membership (device to host) */
		cudaMemcpy(membership_new, membership_d, npoints * sizeof(int), cudaMemcpyDeviceToHost);	
    
	/* for each point, sum data points in each cluster
	   and see if membership has changed:
	     if so, increase delta and change old membership, and update new_centers;
	     otherwise, update new_centers */
	delta = 0;
	for (i = 0; i < npoints; i++)
	{		
		int cluster_id = membership_new[i];
		new_centers_len[cluster_id]++;
		if (membership_new[i] != membership[i])
		{
#ifdef CPU_DELTA_REDUCE
			delta++;
#endif
			membership[i] = membership_new[i];
		}
#ifdef CPU_CENTER_REDUCE
		for (j = 0; j < nfeatures; j++)
		{			
			new_centers[cluster_id][j] += feature[i][j];
		}
#endif
	}
	return delta;
	
}
/* ------------------- kmeansCuda() end ------------------------ */    

