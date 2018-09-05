#include "wb.h"
#include <bits/stdc++.h>
using namespace std;

#define CEIL(a, b) ((a-1)/b +1)
#define Clamp(a, start, end) (max(min(a, end), start))
#define value(arry, i, j, k) (arry[((i)*width + (j)) * depth + (k)])
#define output(i, j, k) value(output, i, j, k)
#define input(i, j, k) value(input, i, j, k)
#define data(i, j, k) data[i*121 + j*11 + k]

#define wbCheck(stmt)                                                           \
    do {                                                                        \
        cudaError_t err = stmt;                                                 \
        if (err != cudaSuccess) {                                               \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
            return -1;                                                          \
        }                                                                       \
    } while (0)

__global__ void stencil(float *output, float *input, int width, int height, int depth) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;

    int bi = blockDim.x * blockIdx.x;
    int bj = blockDim.y * blockIdx.y;
    int bk = blockDim.z * blockIdx.z;

    int x = threadIdx.x;
    int y = threadIdx.y;
    int z = threadIdx.z;

    __shared__ float data[11*11*11];

    if(i < height && j < width && k < depth) {
        data(x+1, y+1, z+1) = input(i, j, k);

        // Z border
        if(bk -1 >= 0)
            data(x, y, 0) = input(i, j, bk -1);
        if(bk + blockDim.z < depth)
            data(x, y, 10) = input(i, j, bk + blockDim.z);

        // Y border
        if(bj -1 >= 0)
            data(x, 0, z) = input(i, bj -1, k);
        if(bj + blockDim.y < width)
            data(x, 10, z) = input(i, bj + blockDim.y, k);

        // X border
        if(bi -1 >= 0)
            data(0, y, z) = input(bi -1, j, k);
        if(bi + blockDim.x < height)
            data(10, y, z) = input(bi + blockDim.x, j, k);

    }

    __syncthreads();


    if(i < 1 || i >= height -1 || j < 1 || j >= width -1 || k < 1 || k >= depth -1) {
        return;
    }

    float res = data(x, y, z + 1) + data(x, y, z - 1) + data(x, y + 1, z) +
        data(x, y - 1, z) + data(x + 1, y, z) + data(x - 1, y, z) - 6 * data(x, y, z);
    res = Clamp(res, 0.0, 1.0);
    output(i, j, k) = res;
}

static void launch_stencil(float *deviceOutputData, float *deviceInputData, 
    int width, int height, int depth) {

    dim3 block(9, 9, 9);
    dim3 grid(CEIL(height, 9), CEIL(width, 9), CEIL(depth, 9));
    stencil <<<grid, block>>> (deviceOutputData, deviceInputData, width, height, depth);
}

int main(int argc, char *argv[]) {

    wbArg_t arg;
    int width;
    int height;
    int depth;
    char *inputFile;
    wbImage_t input;
    wbImage_t output;
    float *hostInputData;
    float *deviceInputData;
    float *deviceOutputData;

    arg = wbArg_read(argc, argv);

    inputFile = wbArg_getInputFile(arg, 0);
    input = wbImport(inputFile);

    width  = wbImage_getWidth(input);
    height = wbImage_getHeight(input);
    depth  = wbImage_getChannels(input);

    output = wbImage_new(width, height, depth);

    hostInputData  = wbImage_getData(input);

    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void **)&deviceInputData, width * height * depth * sizeof(float));
    cudaMalloc((void **)&deviceOutputData, width * height * depth * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");

    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputData, hostInputData, width * height * depth * sizeof(float),
        cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");

    wbTime_start(Compute, "Doing the computation on the GPU");
    launch_stencil(deviceOutputData, deviceInputData, width, height, depth);
    wbTime_stop(Compute, "Doing the computation on the GPU");

    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(output.data, deviceOutputData, width * height * depth * sizeof(float),
        cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");

    wbSolution(arg, output);

    cudaFree(deviceInputData);
    cudaFree(deviceOutputData);

    wbImage_delete(output);
    wbImage_delete(input);
}