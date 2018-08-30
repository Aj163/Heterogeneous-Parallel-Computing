#include <bits/stdc++.h>
#include "wb.h"
using namespace std;

#define BLUR_SIZE 5
#define CHANNELS 3
#define CEIL(a, b) ((a-1)/b +1)

__global__ void imageBlur(float *inputImageData, float *outputImageData, int height, int width) {

	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if(x>=height || y>=width)
		return;

	for (int channel=0; channel<CHANNELS; channel++) {

		float pixVal = 0;
		int pixels = 0;

		for (int blurrow = -BLUR_SIZE; blurrow <= BLUR_SIZE; ++blurrow) {
			for (int blurcol = -BLUR_SIZE; blurcol <= BLUR_SIZE; ++blurcol) {	

				int currow = x + blurrow;
				int curcol = y + blurcol;

				if (currow > -1 && currow < height && curcol > -1 && curcol < width) {
					pixVal += inputImageData[CHANNELS*(currow * width + curcol) + channel];
					pixels++;
				}
			}
		}

		outputImageData[CHANNELS*(x * width + y) + channel] = (pixVal / pixels);
	}
}

int main(int argc, char *argv[]) {

	int imageWidth;
	int imageHeight;
	char *inputImageFile;
	wbImage_t inputImage;
	wbImage_t outputImage;
	float *hostInputImageData;
	float *hostOutputImageData;
	float *deviceInputImageData;
	float *deviceOutputImageData;


	/* parse the input arguments */
	wbArg_t args = wbArg_read(argc, argv);
	inputImageFile = wbArg_getInputFile(args, 0);

	inputImage = wbImport(inputImageFile);

	imageWidth  = wbImage_getWidth(inputImage);
	imageHeight = wbImage_getHeight(inputImage);

	outputImage = wbImage_new(imageWidth, imageHeight, CHANNELS);

	hostInputImageData  = wbImage_getData(inputImage);
	hostOutputImageData = wbImage_getData(outputImage);

	wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

	// Allocate data
	wbTime_start(GPU, "Doing GPU memory allocation");
	cudaMalloc((void **)&deviceInputImageData, imageWidth * imageHeight * CHANNELS * sizeof(float));
	cudaMalloc((void **)&deviceOutputImageData, imageWidth * imageHeight * CHANNELS * sizeof(float));
	wbTime_stop(GPU, "Doing GPU memory allocation");

	// Copy data
	wbTime_start(Copy, "Copying data to the GPU");
	cudaMemcpy(deviceInputImageData, hostInputImageData,
		imageWidth * imageHeight * CHANNELS * sizeof(float), cudaMemcpyHostToDevice);
	wbTime_stop(Copy, "Copying data to the GPU");

	wbTime_start(Compute, "Doing the computation on the GPU");

	// Kernel call
	dim3 block(32, 32, 1);
	dim3 grid(CEIL(imageHeight, 32), CEIL(imageWidth, 32), 1);

	imageBlur <<<grid, block>>> (deviceInputImageData,  deviceOutputImageData, 
		imageHeight, imageWidth);

	wbTime_stop(Compute, "Doing the computation on the GPU");

	// Copy data back
	wbTime_start(Copy, "Copying data from the GPU");
	cudaMemcpy(hostOutputImageData, deviceOutputImageData,
		imageWidth * imageHeight * CHANNELS * sizeof(float), cudaMemcpyDeviceToHost);
	wbTime_stop(Copy, "Copying data from the GPU");

	wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

	// Check solution
	wbSolution(args, outputImage);

	cudaFree(deviceInputImageData);
	cudaFree(deviceOutputImageData);

	wbImage_delete(outputImage);
	wbImage_delete(inputImage);
}
