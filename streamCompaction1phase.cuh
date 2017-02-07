#ifndef STREAMCOMPACTION1PHASE_CUH
#define STREAMCOMPACTION1PHASE_CUH

//
// An in-kernel compaction method. Only useful if there is sufficient other work to hide the computation
// for example, raytracing is typically unbalanced workload and a good example for in-kernel compation. 
// See bottom for use. 
//

// --- Warp-vote functions enable a single thread to find information about the other 
//     threads in the same warp. 
// --- The __ballot() function has been introduced in the Fermi architecture, so CC >= 2.0
//     is required.
// --- unsigned int __ballot(int predicate);
//     Evaluates predicate for all threads of the warp and returns an integer whose n-th bit is set if and only if predicate evaluates 
//     to non - zero for the Nth thread of the warp.		

// --- Stream compaction it typically defined as an operation preserving the ordering of elements in the final output.
//     However, strictly abiding the ordering constraint is not necessary for some applications.
//     Indeed, the ordering of elements in the input or output arrays is contextual and spending time preserving the 
//     ordering may be wasteful especially if the resulting output is to be sorted. For example, a multi-kernel 
//     ray-casting pipeline might only require valid ray-indices as input and in theory it doesn’t matter in which 
//     order, only that they are valid. In practice, rays that traverse the scene close to one another should be 
//     grouped together as much as possible, however, overall the ordering of the groups is not important.
//     The algorithm is a compromise between ensuring local ordering within a block of threads, but not global 
//     ordering of kernel blocks. Within a block, the output location of threadN is guaranteed to be after threadN−x 
//     whereas the output location of blockN is not guaranteed to be after that of blockN−x. However,
//     the order of blocks will be generally close to each other for natural cache-hits to occur. This approach 
//     allows the algorithm not to rely on other prior blocks to complete their output first.

typedef unsigned __int32 uint32;

#define sectionSize		(32)

#define blockSize_x		(512)

uint32	*_CompactedCount;

// --- Total number of launched warps within block
#define totalNumWarps	(blockSize_x >> 5)

/*************************/
/* RETURN THE SECTION ID */
/*************************/
__device__ __forceinline uint32 sectionID()
{
	return blockIdx.x / sectionSize;
}

/****************************************/
/* WRITE RESULTS TO INTERMEDIATE BUFFER */
/****************************************/
// --- All valid thread outputs are stored in the intermediate buffer, offset by the blockId multiplied by the 
//     number of threads-per-block, plus the computed thread and warp offsets.
template<class T>
__device__ __forceinline void writeResultIntermediateBuffer(uint32 offset, T * __restrict__ intermediateBuffer, T V)
{
	intermediateBuffer[blockIdx.x * blockDim.x + offset] = V;
}

/**************************************************************/
/* RETURN THE NUMBER OF THREADS WRITING BEFORE CURRENT THREAD */
/**************************************************************/
//static const uint32 totalNumWarps = KThreads >> 5;

__device__ uint32 getNumThreadsBeforeCurrentThread(bool P, uint32 * __restrict__ numThreadsWritingperBlock)
{

	// --- sh_numThreadsWritingperWarp reports, for each warp in the block, the number of threads
	//     that need to write.
	__shared__ uint32 sh_numThreadsWritingperWarp[totalNumWarps];
	__shared__ uint32 sh_numWarpsWritingBeforeWarp[totalNumWarps];

	// --- Thread index within the warp
	unsigned int threadIdwithinWarp; asm("mov.u32 %0, %%laneid;":"=r"(threadIdwithinWarp));
	// --- 32-bit mask with bits set to 1 in all positions less than the thread's 
	//     lane number in the warp.
	unsigned int threadMask; asm("mov.u32 %0, %%lanemask_lt;":"=r"(threadMask));

	//threadIdwithinWarp = threadIdx.x & (31);
	unsigned int warpIndexwithinBlock;	warpIndexwithinBlock = threadIdx.x >> 5;
	unsigned int isLastThreadinWarp = threadIdwithinWarp == 31;

	// --- Evaluates predicate for all threads of the warp and returns an integer 
	//     whose n-th bit is set if and only if predicate evaluates to non-zero for 
	//     the n-th thread of the warp.
	unsigned int ballotResult = __ballot(P);
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
	//     warp.
	if (isLastThreadinWarp){
		// --- Each warp stores its total count within a shared memory array.
		//     The writing is performed by the last thread in the warp only.
		sh_numThreadsWritingperWarp[warpIndexwithinBlock] = numThreadsinWarpWritingBeforeThread + (unsigned int)P;
	}

	// --- __syncthreads_count(P) is identical to __syncthreads() with the additional feature 
	//     that it evaluates predicate for all threads of the block and returns the number 
	//     of threads for which predicate evaluates to non-zero. 
	//     The synchronization is necessary to ensure that the writing to sh_numThreadsWritingperWarp is 
	//     visible to the other threads within the block.
	//     Furthermore, by calling _syncthreads_count() and passing the predicate of
	//     each thread, we can also gather the total number of writes within the block, which 
	//     will be used at a later stage.
	unsigned int numThreadsWritingperBlockTemp = __syncthreads_count(P);
	if (threadIdx.x == 0) {
		numThreadsWritingperBlock[blockIdx.x] = numThreadsWritingperBlockTemp;
		//printf("tid = %i; tid in warp = %i; numThreadsinWarpWritingBeforeThread = %i; numThreadsWritingperBlock = %i\n", threadIdx.x + blockIdx.x * blockDim.x, threadIdwithinWarp, numThreadsinWarpWritingBeforeThread, numThreadsWritingperBlock[blockIdx.x]);
	}

	// --- The next stage is to find the total number of writes before the current warp which 
	//     serves as the output offset for the warp. This is performed by a scan operation on all the number 
	//     of threads writing per warp (sh_numThreadsWritingperWarp).
	// --- It is assumed that the total number of warps is 32. This is reasonable since, with
	//     32 * 32 threads, a total number of 1024 threads is launched.
	//     Note that, under the condition (threadIdx.x < totalNumWarps), it is implicitly assumed
	//     that the block size is larger than 32.
	if (threadIdx.x < totalNumWarps)
	{
		unsigned int numThreadsWritingperWarp = sh_numThreadsWritingperWarp[threadIdx.x];
		unsigned int numWarpsWritingBeforeWarp;

		numWarpsWritingBeforeWarp = __popc(__ballot(numThreadsWritingperWarp & (1))  & threadMask);
		numWarpsWritingBeforeWarp += __popc(__ballot(numThreadsWritingperWarp & (2))  & threadMask) << 1;
		numWarpsWritingBeforeWarp += __popc(__ballot(numThreadsWritingperWarp & (4))  & threadMask) << 2;
		numWarpsWritingBeforeWarp += __popc(__ballot(numThreadsWritingperWarp & (8))  & threadMask) << 3;
		numWarpsWritingBeforeWarp += __popc(__ballot(numThreadsWritingperWarp & (16)) & threadMask) << 4;
		numWarpsWritingBeforeWarp += __popc(__ballot(numThreadsWritingperWarp & (32)) & threadMask) << 5;

		sh_numWarpsWritingBeforeWarp[threadIdx.x] = numWarpsWritingBeforeWarp;
	}

	// --- To allow other threads to read the values correctly a memory fence and a 
	//     synchronization is required.
	__threadfence();
	__syncthreads();

	return sh_numWarpsWritingBeforeWarp[warpIndexwithinBlock] + numThreadsinWarpWritingBeforeThread;
}

/*****************************************/
/* COUNT THE NUMBER OF BLOCKS IN SECTION */
/*****************************************/
__device__ __forceinline int countNumBlocksinSection(uint32 * __restrict__ _varsB)
{
	__shared__ int completeId; // ??? completeId or _completeId ???

	//completeId = 0;

	// --- 0xFFFFFFFF = -1
	// --- atomicInc ((unsigned int *)&count[0], n); does the following:
	//     ((count[0] >= n) ? 0 : (count[0] + 1)) and store the result back in count[0]
	if (threadIdx.x == 0) {
		completeId = atomicInc(&_varsB[sectionID()], 0xFFFFFFFF);
		//atomicInc(&_varsB[sectionID()], 0xFFFFFFFF);
	}

	__syncthreads();
	//if (threadIdx.x == 0) printf("%i %i %i\n", completeId, _varsB[sectionID()], 0xFFFFFFFF);

	return completeId;
	//return _varsB[sectionID()];
}

/****************************************************/
/* RETURN THE NUMBER OF SECTIONS IN THE THREAD GRID */
/****************************************************/
__device__ __forceinline uint32 returnNumSections()
{
	return ceil((float)gridDim.x / (float)sectionSize);
}

/******************************************************/
/* RETURN THE NUMBER OF BLOCKS IN THE GENERIC SECTION */
/******************************************************/
__device__ __forceinline uint32 returnNumBlocksInSection(uint32 &sectID)
{
	// --- If section is the last section, then the number of blocks is gridDim.x - (sectID * sectionSize),
	//     otherwise it is sectionSize.
	if (sectID == returnNumSections() - 1)
		return gridDim.x - (sectID * sectionSize);
	else
		return sectionSize;
}

/*****************************/
/* SELECT SECTION CONTROLLER */
/*****************************/
// --- The section controller is designated as being the block within the section that increments
//     the section-counter S to the number of blocks within a section S(N).
__device__ __forceinline bool selectSectionController(uint32 &sectID, uint32 numBlocksinSection)
{
	return ((++numBlocksinSection) == returnNumBlocksInSection(sectID));
}

__device__ __forceinline uint32 __SectionOffsetGet(uint32 * __restrict__ _varsC, uint32 * __restrict__ _CompactedCount)
{
	if (sectionID() > 0)
		return _varsC[sectionID() - 1];
	else
		return _CompactedCount[0];
}

__device__ __forceinline void __SectionOffsetSet(uint32 * __restrict__ _varsC, uint32 off)
{
	_varsC[sectionID()] = off;
}

__device__ __forceinline void __SignalFlagEx(uint32 * __restrict__ _varsC)
{
	_varsC[sectionID()] = 1;
}

// --- The block-counts within the section
//	   are loaded into shared memory, by the section-controller, .The
//	number of scans we must perform is log2(w(s)w(n)) and
//	while we have implemented this in a single warp, there is
//		no reason why this work could not be spread among the
//		other warps.Through the scan operation we also compute
//		the total number of output elements t .	// Finalise Segment.
template<class T>
__device__ void finalizeCompaction(uint32 &sectID,
	const T * __restrict__ intermediateBuffer,
	unsigned int * __restrict__ _varsB,
	uint32 * __restrict__ _varsC,
	uint32 * __restrict__ _CompactedCount,
	uint32 * __restrict__ numThreadsWritingperBlock,
	T * __restrict__ _Outputs)
{
	__shared__ unsigned int blockOffsetinSection[sectionSize];
	__shared__ unsigned int _sh_sc;
	__shared__ volatile unsigned int _sh_gsp;

	uint32 numBlocksInSection = returnNumBlocksInSection(sectID);

	// --- Assumes sectionSize <= 32, namely, we limit the number of blocks within a section to the warp size.
	//     In the following, and a scan operation is performed to find the offset of each block. The scan 
	//     operation is performed by a single warp of the section controller using bit-decomposition, as for
	//     getNumThreadsBeforeCurrentThread.
	if (threadIdx.x < numBlocksInSection)
	{
		// --- Thread index within the warp
		unsigned int threadIdwithinWarp; asm("mov.u32 %0, %%laneid;":"=r"(threadIdwithinWarp));
		// --- 32-bit mask with bits set to 1 in all positions less than the thread's 
		//     lane number in the warp.
		unsigned int threadMask; asm("mov.u32 %0, %%lanemask_lt;":"=r"(threadMask));

		// --- Wait for all blocks in section to complete
		// --- The last block in section Si waits for the flag S(f)i−1 in section Si−1 to go true. The first 
		//     section S0 progresses immediately to the next stage.
		// --- ???
		while (true)
			if (((volatile uint32 *)_varsB)[sectID] == numBlocksInSection)
				break;

		// --- Find the offset of each block within the Section by a single warp using bit-decomposition.
		unsigned int numThreadsWritingperBlock_ = numThreadsWritingperBlock[sectID * sectionSize + threadIdx.x];
		unsigned int _sum = 0;
#pragma unroll
		for (int i = 0; i <= 10; i++)
			_sum += __popc(__ballot(numThreadsWritingperBlock_ & (1 << i)) & threadMask) << i;

		blockOffsetinSection[threadIdx.x] = _sum;

		if (threadIdx.x == numBlocksInSection - 1)
		{
			_sh_sc = _sum + numThreadsWritingperBlock_;
		}

		//
		// Wait for prior Section to complete
		//
		if (threadIdx.x == 0)
		{
			if (sectID > 0)
				while (true)
					if (((volatile uint32*)_varsC)[sectID - 1] > 0)
						break;

			//Grab the global offset for this section
			uint32 _pgsp = __SectionOffsetGet(_varsC, _CompactedCount);
			_sh_gsp = _pgsp;

			//Write the global offset for the next section
			__SectionOffsetSet(_varsC, _pgsp + _sh_sc);

			if (sectID == returnNumSections() - 1)
			{
				_CompactedCount[0] = _pgsp + _sh_sc;
			}

			//Don't Signal complete until our memory is written. 
			__threadfence();

			// Signal section complete flag
			__SignalFlagEx(_varsC);
		}
	}

	__syncthreads();

	int s = sectID * sectionSize;
	int e = s + numBlocksInSection;
	int offset = 0;

	for (int i = s; i< e; i++)
	{
		uint32 count = numThreadsWritingperBlock[i];

		if (threadIdx.x < count)
		{
			_Outputs[_sh_gsp + offset + threadIdx.x] = intermediateBuffer[blockDim.x * i + threadIdx.x];
		}

		offset += count;
	}
}

/***************/
/* TEST KERNEL */
/***************/
template<class T>
__global__ void testKernel(const int N, uint32 * __restrict__ numThreadsWritingperBlock, T * intermediateBuffer,
	uint32 * __restrict__ _varsB, uint32 * __restrict__ _varsC,
	uint32 * __restrict__ _CompactedCount, const T * __restrict__ d_in, T * __restrict__ d_out)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid >= N) return;

	T element = d_in[tid];							// --- Load element from global memory
	bool predicate = element > 5;					// --- Evaluate predicate

	// --- Section number
	uint32 sectID = sectionID();

	// --- Calculate the number of threads writing before current thread in the corresponding thread block.
	uint32 numThreadsBeforeCurrentThread = getNumThreadsBeforeCurrentThread(predicate, numThreadsWritingperBlock);

	// --- Following the above __device__ function, we have calculated the number of threads writing before
	//     the corresponding thread in the relevant thread block. So, we can offset the writing of the relevant
	//     thread in the relevant block. Also, we know the number of threads writing per block. It remains now 
	//     to calculate the offset within the whole thread grid.

	// --- Before computing the grid offset, the output is written to an intermediate buffer.
	if (predicate) writeResultIntermediateBuffer(numThreadsBeforeCurrentThread, intermediateBuffer, element);

	// --- To achieve an order-preserving compaction, first, as a pre-process, a kernel is divided into S 
	//     sections, where each section is made by 32 blocks. 
	// --- Signal Complete
	int numBlocksInSection = countNumBlocksinSection(_varsB);

	//printf("sectionId %i numBlocksInSection %i\n", sectID, numBlocksInSection);
	// --- Only One Block is a mover, the rest exit early
	// --- A single thread in the block atomically increments the section counter by one. All blocks within 
	//     the section, except for the section controller block, are allowed to exit the compaction function.
	//     This behaviour allows all but one of the blocks in the section to finish and free
	//     up resources for other blocks to start commencing without delay. It also allows blocks to do 
	//     additional work and/or additional compactions. 
	//printf("Section controller? %i\n", selectSectionController(sectID, numBlocksInSection));
	if (!selectSectionController(sectID, numBlocksInSection))
	{
		// --- Let Block exit
		return;
	}

	// --- Finish off compaction. This is performed by the section controller only.
	finalizeCompaction(sectID, intermediateBuffer, _varsB, _varsC, _CompactedCount, numThreadsWritingperBlock, d_out);
}


#endif