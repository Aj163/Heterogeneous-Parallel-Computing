#include <bits/stdc++.h>
#include <cuda.h>
using namespace std;

#define N ((int)1e7)
#define CEIL(a, b) ((a-1)/b +1)

__global__ void add(int *d_a, int *d_b, int *d_c) {

	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i < N)
		d_c[i] = d_b[i] + d_a[i];
}

int main() {

	int *h_a, *h_b, *h_c;
	int *d_a, *d_b, *d_c;

	// Allocate host memory
	h_a = new int[N];
	h_b = new int[N];
	h_c = new int[N];

	printf("Number of elements in array : %d\n\n", N);

	srand(time(0));
	for(int i=0; i<N; i++) {
		h_a[i] = rand()%N;
		h_b[i] = rand()%N;
	}

	// Allocate memory on device
	cudaMalloc((void**)&d_a, N*sizeof(int));
	cudaMalloc((void**)&d_b, N*sizeof(int));
	cudaMalloc((void**)&d_c, N*sizeof(int));

	//Copy data into device memory
	cudaMemcpy(d_a, h_a, N*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, N*sizeof(int), cudaMemcpyHostToDevice);

	// Kernel call
	add<<<CEIL(N, 1024), 1024>>>(d_a, d_b, d_c);
	cudaThreadSynchronize();

	// Copy data back to host
	cudaMemcpy(h_c, d_c, N*sizeof(int), cudaMemcpyDeviceToHost);

	// Free device memory
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	int errors = 0;
	for(int i=0; i<N; i++)
		if(h_c[i] != h_a[i] + h_b[i]) {
			errors++;
			if(errors <= 10)
				printf("Test failed at index : %9d\n", i);
		}

	if(errors)
		printf("\n%9d Tests failed!\n\n", errors);
	else
		printf("All tests passed !\n\n");	

	// Free host memory
	delete[] h_a, h_b, h_c;
}