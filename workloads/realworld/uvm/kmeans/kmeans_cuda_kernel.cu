#ifndef _KMEANS_CUDA_KERNEL_H_
#define _KMEANS_CUDA_KERNEL_H_

#include <stdio.h>
#include <cuda.h>

#include "kmeans.h"

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

using namespace nvcuda::experimental;

#define PREFETCH_COUNT 2

// FIXME: Make this a runtime selectable variable!
#define ASSUMED_NR_CLUSTERS 32

#define SDATA(index) CUT_BANK_CHECKER(sdata, index)

// t_features has the layout dim0[points 0-m-1]dim1[ points 0-m-1]...
texture<float, 1, cudaReadModeElementType> t_features;
// t_features_flipped has the layout point0[dim 0-n-1]point1[dim 0-n-1]
texture<float, 1, cudaReadModeElementType> t_features_flipped;
texture<float, 1, cudaReadModeElementType> t_clusters;

__constant__ float c_clusters[ASSUMED_NR_CLUSTERS * 34]; /* constant memory for cluster centers */

/* ----------------- invert_mapping() --------------------- */
/* inverts data array from row-major to column-major.

   [p0,dim0][p0,dim1][p0,dim2] ...
   [p1,dim0][p1,dim1][p1,dim2] ...
   [p2,dim0][p2,dim1][p2,dim2] ...
										to
   [dim0,p0][dim0,p1][dim0,p2] ...
   [dim1,p0][dim1,p1][dim1,p2] ...
   [dim2,p0][dim2,p1][dim2,p2] ...
*/
__global__ void invert_mapping(float *input,  /* original */
							   float *output, /* inverted */
							   int npoints,	  /* npoints */
							    int batch_size,
							   int nfeatures) /* nfeatures */
{
	int point_id = threadIdx.x + blockDim.x * blockIdx.x; /* id of thread */

	int batches = npoints / batch_size;

	for (int b = 0; b < batches; b++) 
	{
		for (int i = 0; i < nfeatures; i++)
		{
			output[b * batch_size + point_id + npoints * i] = input[(b * batch_size + point_id) * nfeatures + i];
		}
	}



	return;
}
/* ----------------- invert_mapping() end --------------------- */

/* to turn on the GPU delta and center reduction */
// #define GPU_DELTA_REDUCTION
// #define GPU_NEW_CENTER_REDUCTION

/* ----------------- kmeansPoint() --------------------- */
/* find the index of nearest cluster centers and change membership*/
__global__ void
kmeansPoint(float *features, /* in: [npoints*nfeatures] */
			int nfeatures,
			int npoints,
			int batch_size,
			int nclusters,
			int *membership)
{
	cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
	// block ID
	const unsigned int block_id = gridDim.x * blockIdx.y + blockIdx.x;
	// point/thread ID
	const unsigned int point_id = block_id * blockDim.x * blockDim.y + threadIdx.x;

	__shared__ float tmp_features[THREADS_PER_DIM][THREADS_PER_DIM][16];

	int batches = npoints / batch_size;
	int tile = 0;
	int end_tile = tile + batches;

	for (; tile < end_tile; tile += 1) 
	{
		for (int i = 0; i < 16; i++)
		{
			int addr = tile * batch_size + point_id + i * npoints;
			tmp_features[threadIdx.y][threadIdx.x][i] = features[addr];
		}
		block.sync();

		int index = -1;

		float min_dist = FLT_MAX;
		float dist; /* distance square between a point to cluster center */

		/* find the cluster center id with min distance to pt */
		for (int i = 0; i < nclusters; i++)
		{
			int cluster_base_index = i * nfeatures; /* base index of cluster centers for inverted array */
			float ans = 0.0;						/* Euclidean distance sqaure */

			for (int j = 0; j < nfeatures; j++)
			{
				// int addr = point_id + j * npoints; /* appropriate index of data point */
				// float diff = (tex1Dfetch(t_features,addr) - c_clusters[cluster_base_index + j]);	/* distance between a data point to cluster centers */

				// int addr = point_id + j * npoints; /* appropriate index of data point */
				// float diff = features[addr] - c_clusters[cluster_base_index + j]; /* distance between a data point to cluster centers */
				float diff = tmp_features[threadIdx.y][threadIdx.x][j] - c_clusters[cluster_base_index + j]; /* distance between a data point to cluster centers */
				ans += diff * diff;																			 /* sum of squares */
			}
			dist = ans;
			block.sync();

			/* see if distance is smaller than previous ones:
			if so, change minimum distance and save index of cluster center */
			if (dist < min_dist)
			{
				min_dist = dist;
				index = i;
			}
		}
		membership[tile * batch_size + point_id] = index;
		block.sync();
	}
	
}
#endif // #ifndef _KMEANS_CUDA_KERNEL_H_
