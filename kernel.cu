#include "Utilities.cuh"
#include "TimingGPU.cuh"

using namespace std;

#include "streamCompaction3phases.cuh"

#include "streamCompaction1phase.cuh"

/********/
/* MAIN */
/********/
int main() {

	srand(time(0));

	TimingGPU timerGPU;

	//int *d_input, *d_output;

	unsigned int N = 2 * 1048576;
	unsigned int blockSize = 1024;
	//unsigned int N			= 64;
	//unsigned int blockSize	= 32;

	// --- Set up data on the CPU and allocate space for results
	int *h_input = (int *)malloc(N * sizeof(int));
	int *h_output = (int *)malloc(N * sizeof(int));
	int *h_outputThrust = (int *)malloc(N * sizeof(int));
	for (int k = 0; k < N; k++) {
		h_input[k] = rand() % 10 + 1;
		//printf("h_input[%i] = %i\n", k, h_input[k]);
	}

	// --- Move the data to GPU and allocate space for results
	int *d_input;			gpuErrchk(cudaMalloc(&d_input, N * sizeof(int)));
	int *d_output;			gpuErrchk(cudaMalloc(&d_output, N * sizeof(int)));
	int *d_outputThrust;	gpuErrchk(cudaMalloc(&d_outputThrust, N * sizeof(int)));
	gpuErrchk(cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice));

	/*************************/
	/* THREE-PHASES APPROACH */
	/*************************/
	int *d_BlocksCount;		gpuErrchk(cudaMalloc(&d_BlocksCount, sizeof(int) * iDivUp(N, blockSize)));
	int *d_BlocksOffset;	gpuErrchk(cudaMalloc(&d_BlocksOffset, sizeof(int) * iDivUp(N, blockSize)));

	// --- Perform stream compaction
	timerGPU.StartCounter();
	compact<int>(d_input, d_output, N, int_predicate(), d_BlocksCount, d_BlocksOffset, blockSize);
	printf("Timing alternative approach = %f\n", timerGPU.GetCounter());

	thrust::device_ptr<int> d_outputThrustPtr(d_outputThrust);
	thrust::device_ptr<int> d_dataPtr(d_input);
	timerGPU.StartCounter();
	//thrust::remove_if(d_dataPtr, d_dataPtr + N, d_outputThrustPtr, int_not_predicate());
	thrust::copy_if(d_dataPtr, d_dataPtr + N, d_dataPtr, d_outputThrustPtr, int_predicate());
	printf("Timing Thrust = %f\n", timerGPU.GetCounter());

	// --- Move the results to CPU
	gpuErrchk(cudaMemcpy(h_output, d_output, N * sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_outputThrust, d_outputThrust, N * sizeof(int), cudaMemcpyDeviceToHost));
	for (int k = 0; k < N; k++) {
		if (h_output[k] != h_outputThrust[k])
			printf("h_output[%i] = %i; h_outputThrust[%i] = %i\n", k, h_output[k], k, h_outputThrust[k]);
	}

	timerGPU.StartCounter();
	// --- Pointer to a temporary array of size equal to the number of blocks containing the number of threads writing per each block
	const int numBlocks = iDivUp(N, blockSize_x);
	uint32		*numThreadsWritingperBlock;			gpuErrchk(cudaMalloc(&numThreadsWritingperBlock, sizeof(uint32) * numBlocks));
	int *intermediateBuffer;	gpuErrchk(cudaMalloc(&intermediateBuffer, sizeof(int) * N));
	// --- Maximum number of sections
	static const uint32 maxNumSections = numBlocks / sectionSize + (uint32)((numBlocks % sectionSize) > 0);
	// --- Pointer to a temporary array of size equal to the number of sections containing the number of blocks in each section
	uint32		*_varsB;			gpuErrchk(cudaMalloc(&_varsB, sizeof(uint32) * maxNumSections));
	gpuErrchk(cudaMemset((void*)_varsB, 0, sizeof(uint32) * maxNumSections));
	uint32      *_varsC;			gpuErrchk(cudaMalloc(&_varsC, sizeof(uint32) * maxNumSections));
	gpuErrchk(cudaMemset((void*)_varsC, 0, sizeof(uint32) * maxNumSections));
	uint32		*_CompactedCount;	gpuErrchk(cudaMalloc(&_CompactedCount, sizeof(uint32)));
	//timerGPU.StartCounter();
	testKernel << <numBlocks, blockSize_x >> >(N, numThreadsWritingperBlock, intermediateBuffer, _varsB, _varsC, _CompactedCount, d_input, d_output);
	printf("Timing alternative approach = %f\n", timerGPU.GetCounter());

	int *h_intermediateBuffer = (int *)malloc(N * sizeof(int));
	//gpuErrchk(cudaMemcpy(h_intermediateBuffer, intermediateBuffer, N * sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_intermediateBuffer, d_output, N * sizeof(int), cudaMemcpyDeviceToHost));

	//for (int k = 0; k < N; k++) printf("%i - h_data = %i; h_compact = %i\n", k, h_input[k], h_intermediateBuffer[k]);

	return 0;
}
