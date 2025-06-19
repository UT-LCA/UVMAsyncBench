//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------200
//	plasmaKernel_gpu_2
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------200

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

using namespace nvcuda::experimental;

#define PREFETCH_COUNT 2

__global__ void kernel_gpu_cuda(par_str d_par_gpu,
								dim_str d_dim_gpu,
								box_str *d_box_gpu,
								FOUR_VECTOR *d_rv_gpu,
								fp *d_qv_gpu,
								FOUR_VECTOR *d_fv_gpu,
								int boxes_per_block)
{
	cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
	pipeline pipe;
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------180
	//	THREAD PARAMETERS
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------180
	int bx = blockIdx.x;  // get current horizontal block index (0-n)
	int tx = threadIdx.x; // get current horizontal thread index (0-n)
	int wtx = tx;

	//------------------------------------------------------------------------------------------------------------------------------------------------------160
	//	Extract input parameters
	//------------------------------------------------------------------------------------------------------------------------------------------------------160

	// parameters
	fp a2 = 2.0 * d_par_gpu.alpha * d_par_gpu.alpha;

	// home box
	int first_i;
	FOUR_VECTOR *rA;
	FOUR_VECTOR *fA;
	__shared__ FOUR_VECTOR rA_shared[100];

	// nei box
	int pointer;
	int k = 0;
	int first_j;
	FOUR_VECTOR *rB;
	fp *qB;
	int j = 0;
	__shared__ FOUR_VECTOR rB_shared[NUMBER_PAR_PER_BOX * PREFETCH_COUNT];
	__shared__ double qB_shared[NUMBER_PAR_PER_BOX * PREFETCH_COUNT];

	// common
	fp r2;
	fp u2;
	fp vij;
	fp fs;
	fp fxij;
	fp fyij;
	fp fzij;
	THREE_VECTOR d;

	int box = bx * boxes_per_block;
	int end_box = box + boxes_per_block;

	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------180
	//	DO FOR THE NUMBER OF BOXES
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------180
	for (; box < end_box; box++)
	{
		//------------------------------------------------------------------------------------------------------------------------------------------------------160
		//	Home box
		//------------------------------------------------------------------------------------------------------------------------------------------------------160

		//----------------------------------------------------------------------------------------------------------------------------------140
		//	Setup parameters
		//----------------------------------------------------------------------------------------------------------------------------------140

		// home box - box parameters
		first_i = d_box_gpu[box].offset;

		// home box - distance, force, charge and type parameters
		rA = &d_rv_gpu[first_i];
		fA = &d_fv_gpu[first_i];

		//----------------------------------------------------------------------------------------------------------------------------------140
		//	Copy to shared memory
		//----------------------------------------------------------------------------------------------------------------------------------140

		// home box - shared memory
		while (wtx < NUMBER_PAR_PER_BOX)
		{
			rA_shared[wtx] = rA[wtx];
			wtx = wtx + NUMBER_THREADS;
		}
		wtx = tx;

		// synchronize threads  - not needed, but just to be safe
		block.sync();

		//------------------------------------------------------------------------------------------------------------------------------------------------------160
		//	nei box loop
		//------------------------------------------------------------------------------------------------------------------------------------------------------160

		// if (wtx == 0)
		// 	printf("d_box_gpu[%d].nn is %d\n", bx, d_box_gpu[bx].nn);

		int fetch = 0;
		int end_tile = 1 + d_box_gpu[box].nn;

		// loop over neiing boxes of home box
		for (int compute = fetch; compute < end_tile; compute++)
		{
			for (; fetch < end_tile && fetch < compute + PREFETCH_COUNT; fetch++)
			{
				//----------------------------------------50
				//	nei box - get pointer to the right box
				//----------------------------------------50

				if (fetch == 0)
				{
					pointer = box; // set first box to be processed to home box
				}
				else
				{
					pointer = d_box_gpu[box].nei[fetch - 1].number; // remaining boxes are nei boxes
				}

				//----------------------------------------------------------------------------------------------------------------------------------140
				//	Setup parameters
				//----------------------------------------------------------------------------------------------------------------------------------140

				// nei box - box parameters
				first_j = d_box_gpu[pointer].offset;

				// nei box - distance, (force), charge and (type) parameters
				rB = &d_rv_gpu[first_j];
				qB = &d_qv_gpu[first_j];

				//----------------------------------------------------------------------------------------------------------------------------------140
				//	Setup parameters
				//----------------------------------------------------------------------------------------------------------------------------------140

				// nei box - shared memory
				while (wtx < NUMBER_PAR_PER_BOX)
				{
					memcpy_async(rB_shared[(fetch % PREFETCH_COUNT) * NUMBER_PAR_PER_BOX + wtx], rB[wtx], pipe);
					memcpy_async(qB_shared[(fetch % PREFETCH_COUNT) * NUMBER_PAR_PER_BOX + wtx], qB[wtx], pipe);
					wtx = wtx + NUMBER_THREADS;
				}
				wtx = tx;

				// synchronize threads because in next section each thread accesses data brought in by different threads here
				pipe.commit();
			}
			if (fetch == end_tile)
			{
				for (int i = 0; i < PREFETCH_COUNT - 1; ++i)
				{
					pipe.commit();
				}
				++fetch;
			}
			pipe.wait_prior<PREFETCH_COUNT - 1>();
			block.sync();

			//----------------------------------------------------------------------------------------------------------------------------------140
			//	Calculation
			//----------------------------------------------------------------------------------------------------------------------------------140

			// loop for the number of particles in the home box
			// for (int i=0; i<nTotal_i; i++){
			while (wtx < NUMBER_PAR_PER_BOX)
			{
				// loop for the number of particles in the current nei box
				for (j = 0; j < NUMBER_PAR_PER_BOX; j++)
				{
					r2 = (fp)rA_shared[wtx].v + (fp)rB_shared[(compute % PREFETCH_COUNT) * NUMBER_PAR_PER_BOX + j].v - DOT((fp)rA_shared[wtx], (fp)rB_shared[(compute % PREFETCH_COUNT) * NUMBER_PAR_PER_BOX + j]);
					u2 = a2 * r2;
					vij = exp(-u2);
					fs = 2 * vij;

					d.x = (fp)rA_shared[wtx].x - (fp)rB_shared[(compute % PREFETCH_COUNT) * NUMBER_PAR_PER_BOX + j].x;
					fxij = fs * d.x;
					d.y = (fp)rA_shared[wtx].y - (fp)rB_shared[(compute % PREFETCH_COUNT) * NUMBER_PAR_PER_BOX + j].y;
					fyij = fs * d.y;
					d.z = (fp)rA_shared[wtx].z - (fp)rB_shared[(compute % PREFETCH_COUNT) * NUMBER_PAR_PER_BOX + j].z;
					fzij = fs * d.z;

					fA[wtx].v += (double)((fp)qB_shared[(compute % PREFETCH_COUNT) * NUMBER_PAR_PER_BOX + j] * vij);
					fA[wtx].x += (double)((fp)qB_shared[(compute % PREFETCH_COUNT) * NUMBER_PAR_PER_BOX + j] * fxij);
					fA[wtx].y += (double)((fp)qB_shared[(compute % PREFETCH_COUNT) * NUMBER_PAR_PER_BOX + j] * fyij);
					fA[wtx].z += (double)((fp)qB_shared[(compute % PREFETCH_COUNT) * NUMBER_PAR_PER_BOX + j] * fzij);
				}

				// increment work thread index
				wtx = wtx + NUMBER_THREADS;
			}

			// reset work index
			wtx = tx;

			// synchronize after finishing force contributions from current nei box not to cause conflicts when starting next box
			block.sync();

			//----------------------------------------------------------------------------------------------------------------------------------140
			//	Calculation END
			//----------------------------------------------------------------------------------------------------------------------------------140
		}
	}
}
