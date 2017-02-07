#ifndef STREAMCOMPACTION3PHASES_CUH
#define STREAMCOMPACTION3PHASES_CUH

#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <thrust/remove.h>

#include "Utilities.cuh"

//#define		DEBUG

#define		warp_size	(32)

/********************/
/* PREDICATE STRUCT */
/********************/
struct int_predicate
{
	__host__ __device__ __forceinline__ bool operator()(const int x) { return x >= 5; }
};

/**********/
/* STEP 1 */
/**********/
template <typename T, typename Predicate>
__global__ void computePredicateTruePerBlock(const T * __restrict__ d_input, const int N, int * __restrict__ d_BlockCounts, Predicate predicate) {

	int tid = threadIdx.x + blockIdx.x*blockDim.x;

	if (tid < N) {

		int pred = predicate(d_input[tid]);
		int BC = __syncthreads_count(pred);

		if (threadIdx.x == 0) { d_BlockCounts[blockIdx.x] = BC; }

	}
}

/**********/
/* STEP 3 */
/**********/
// --- unsigned int __ballot(int predicate);
//     Evaluates predicate for all threads of the warp and returns an integer whose n-th bit is set if and only if predicate evaluates 
//     to non - zero for the Nth thread of the warp.		
// --- __device__ ​ int __popc (unsigned int  x)
//     Count the number of bits that are set to 1 in a 32 bit integer.
template <typename T, typename Predicate>
__global__ void compactK(const T * __restrict__ d_input, const int length, T * __restrict__ d_output,
	int * __restrict__ d_BlocksOffset, Predicate predicate) {

	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	extern __shared__ int sh_numThreadsWritingperWarpinBlock[];

	if (tid < length) {

		int pred = predicate(d_input[tid]);

		// --- Warp index within the block
		unsigned int warpIndexwithinBlock = threadIdx.x >> 5;

		// --- Thread index within the warp
		unsigned int threadIndexwithinWarp; asm("mov.u32 %0, %%laneid;":"=r"(threadIndexwithinWarp));

		// --- 32-bit mask with bits set to 1 in all positions less than the thread's 
		//     lane number in the warp.
		unsigned int threadMask; asm("mov.u32 %0, %%lanemask_lt;":"=r"(threadMask));

		//printf("%i %i\n", threadMask, INT_MAX >> (warp_size - threadIndexwithinWarp - 1));

		// --- Evaluates predicate for all threads of the warp and returns an integer 
		//     whose n-th bit is set if and only if predicate evaluates to non-zero for 
		//     the n-th thread of the warp.
		unsigned int ballotResult = __ballot(pred);
		// --- By masking the result of __ballot() such that bits representing threads 
		//     after the current thread are set to zero, and counting the set-bits, we 
		//     get the number of threads before the current thread (in the warp) that 
		//     will write an output.
		// --- __popc(int v) intrinsic returns the number of bits set in the binary 
		//     representation of integer v
		unsigned int numThreadsinWarpWritingBeforeThread = __popc(ballotResult & threadMask);

		// --- The total number of writes within warp can be found by the number of 
		//     threads writing before the last thread within warp plus 1, depending
		//     on the value of the predicate corresponding to the last thread within
		//     warp. The total number of writes per warp is written by the last thread
		//     within the warp.
		if (threadIndexwithinWarp == warp_size - 1) {
			sh_numThreadsWritingperWarpinBlock[warpIndexwithinBlock] = numThreadsinWarpWritingBeforeThread + pred;
		}

		__syncthreads();

		// --- The next stage is to find the total number of writes before the current warp which 
		//     serves as the output offset for the warp. This is performed by a scan operation on all the number 
		//     of threads writing per warp (sh_numThreadsWritingperWarp).
		// --- It is assumed that the total number of warps is 32. This is reasonable since, with
		//     32 * 32 threads, a total number of 1024 threads is launched.
		//     Note that, under the condition (threadIdx.x < totalNumWarps), it is implicitly assumed
		//     that the block size is larger than 32.
		if ((warpIndexwithinBlock == 0) && (threadIndexwithinWarp < blockDim.x / warp_size)) {
			int numWarpsWritingBeforeWarp = 0;
			int temp;
			temp = __ballot(sh_numThreadsWritingperWarpinBlock[threadIndexwithinWarp] & (1 << 0));
			numWarpsWritingBeforeWarp += (__popc(temp & threadMask)) << 0;
			temp = __ballot(sh_numThreadsWritingperWarpinBlock[threadIndexwithinWarp] & (1 << 1));
			numWarpsWritingBeforeWarp += (__popc(temp & threadMask)) << 1;
			temp = __ballot(sh_numThreadsWritingperWarpinBlock[threadIndexwithinWarp] & (1 << 2));
			numWarpsWritingBeforeWarp += (__popc(temp & threadMask)) << 2;
			temp = __ballot(sh_numThreadsWritingperWarpinBlock[threadIndexwithinWarp] & (1 << 3));
			numWarpsWritingBeforeWarp += (__popc(temp & threadMask)) << 3;
			temp = __ballot(sh_numThreadsWritingperWarpinBlock[threadIndexwithinWarp] & (1 << 4));
			numWarpsWritingBeforeWarp += (__popc(temp & threadMask)) << 4;
			temp = __ballot(sh_numThreadsWritingperWarpinBlock[threadIndexwithinWarp] & (1 << 5));
			numWarpsWritingBeforeWarp += (__popc(temp & threadMask)) << 5;

			sh_numThreadsWritingperWarpinBlock[threadIndexwithinWarp] = numWarpsWritingBeforeWarp;
		}

		__syncthreads();

		if (pred) {
			d_output[numThreadsinWarpWritingBeforeThread + sh_numThreadsWritingperWarpinBlock[warpIndexwithinBlock] + d_BlocksOffset[blockIdx.x]] = d_input[tid];
		}

	}
}

/*************************/
/* COORDINATING FUNCTION */
/*************************/
template <typename T, typename Predicate>
void compact(const T * __restrict__ d_input, T * __restrict__ d_output, const int length, Predicate predicate, int * __restrict__ d_BlocksCount, int * __restrict__ d_BlocksOffset, const int blockSize) {

	int numBlocks = iDivUp(length, blockSize);

	thrust::device_ptr<int> d_BlocksCountPtr(d_BlocksCount);
	thrust::device_ptr<int> d_BlocksOffsetPtr(d_BlocksOffset);

	// --- Phase 1
	computePredicateTruePerBlock << <numBlocks, blockSize >> >(d_input, length, d_BlocksCount, predicate);
#ifdef DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

	// --- Phase 2
	thrust::exclusive_scan(d_BlocksCountPtr, d_BlocksCountPtr + numBlocks, d_BlocksOffsetPtr);

	// --- Phase 3
	compactK << <numBlocks, blockSize, sizeof(int) * (blockSize / warp_size) >> >(d_input, length, d_output, d_BlocksOffset, predicate);
#ifdef DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

}

#endif
