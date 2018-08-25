#include <stdio.h> 

int main() {

	const int kb = 1024;
	const int mb = kb * kb;
	const int gb = mb * kb;
	int nDevices;

	cudaGetDeviceCount(&nDevices);
  
	for (int i = 0; i < nDevices; i++) {

		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);

		printf("\nDevice %d - GPU Card name 	: %s\n", i, prop.name);
		printf("Compute Capabilities 			: %d.%d\n", prop.major, prop.minor);
		printf("Maximum Block Dimensions 		: %d x %d x %d\n", 
			prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		printf("Maximum Grid dimensions 		: %d x %d x %d\n",
			prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
		printf("Total global memory 			: %0.2lf GB\n", prop.totalGlobalMem *1.0 / gb);
		printf("Total Constant memory 			: %0.2lf KB\n", prop.totalConstMem *1.0 / kb);
		printf("Shared memory per block 		: %0.2lf KB\n", prop.sharedMemPerBlock *1.0 / kb);
		printf("Warp size 						: %d\n", prop.warpSize);
	}
}