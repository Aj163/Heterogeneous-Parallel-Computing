#include "wb.h"
#include <bits/stdc++.h>
using namespace std;

#define NUM_BINS 4096
#define BIN_CAP 127
#define CEIL(a, b) ((a-1)/b +1)

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
	
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {

	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),
		file, line);
		if (abort)
			exit(code);
	}
}

__global__ void histogram(uint *input, uint *bins, int N) {

	int id = threadIdx.x + blockDim.x * blockIdx.x;
	if(id >= N)
		return;

	atomicAdd(&bins[input[id]], 1);
	atomicMin(&bins[input[id]], BIN_CAP);
}

int main(int argc, char *argv[]) {

	int inputLength;
	uint *hostInput;
	uint *hostBins;
	uint *deviceInput;
	uint *deviceBins;

	/* Read input arguments here */
	wbArg_t args = wbArg_read(argc, argv);

	wbTime_start(Generic, "Importing data and creating memory on host");
	hostInput = (uint *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
	hostBins = (uint *)calloc(NUM_BINS, sizeof(uint));
	wbTime_stop(Generic, "Importing data and creating memory on host");

	wbLog(TRACE, "The input length is ", inputLength);
	wbLog(TRACE, "The number of bins is ", NUM_BINS);

	wbTime_start(GPU, "Allocating GPU memory.");
	// Allocating GPU memory
	cudaMalloc((void **)&deviceInput, inputLength * sizeof(uint));
	cudaMalloc((void **)&deviceBins, NUM_BINS * sizeof(uint));
	CUDA_CHECK(cudaDeviceSynchronize());
	wbTime_stop(GPU, "Allocating GPU memory.");

	wbTime_start(GPU, "Copying input memory to the GPU.");
	// Copying memory to the GPU
	CUDA_CHECK(cudaDeviceSynchronize());
	cudaMemcpy(deviceInput, hostInput, inputLength * sizeof(uint), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceBins, hostBins, NUM_BINS * sizeof(uint), cudaMemcpyHostToDevice);
	wbTime_stop(GPU, "Copying input memory to the GPU.");

	// Launch kernel
	dim3 grid(CEIL(inputLength, 1024), 1, 1);
	dim3 block(1024, 1, 1);
	wbLog(TRACE, "Launching kernel");
	wbTime_start(Compute, "Performing CUDA computation");
	// Kernel computation
	histogram <<<grid, block>>> (deviceInput, deviceBins, inputLength);
	wbTime_stop(Compute, "Performing CUDA computation");

	wbTime_start(Copy, "Copying output memory to the CPU");
	// Copying the GPU memory back to the CPU
	CUDA_CHECK(cudaDeviceSynchronize());
	cudaMemcpy(hostBins, deviceBins, NUM_BINS * sizeof(uint), cudaMemcpyDeviceToHost);
	wbTime_stop(Copy, "Copying output memory to the CPU");

	wbTime_start(GPU, "Freeing GPU Memory");
	// Freeing the GPU memory
	cudaFree(deviceBins);
	cudaFree(deviceInput);
	wbTime_stop(GPU, "Freeing GPU Memory");

	// Verify correctness
	// -----------------------------------------------------
	wbSolution(args, hostBins, NUM_BINS);

	free(hostBins);
	free(hostInput);
}
