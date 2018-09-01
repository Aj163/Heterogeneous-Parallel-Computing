#include <bits/stdc++.h>
#include <cuda.h>
using namespace std;

#define N ((int)1e4)
#define CEIL(a, b) ((a-1)/b +1)

__global__ void add(int *d_a, int *d_b, int *d_c) {

	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x < N && y < N)
		d_c[x*N + y] = d_b[x*N + y] + d_a[x*N + y];
}

template <class T>
void testSolution(T *h_a, T *h_b, T *h_c, float precision=0.0) {

	int errors = 0;
	for(int i=0; i<N; i++)
		if(abs(h_c[i] - h_a[i] - h_b[i]) > precision) {
			errors++;
			if(errors <= 10)
				printf("Test failed at index : %d\n", i);
		}

	if(errors)
		printf("\n%d Tests failed!\n\n", errors);
	else
		printf("All tests passed !\n\n");
}

int main() {

	int *h_a, *h_b, *h_c;
	int *d_a, *d_b, *d_c;

	// Allocate host memory
	h_a = new int[N*N];
	h_b = new int[N*N];
	h_c = new int[N*N];

	printf("\nSize of matrices : %d x %d\n\n", N, N);

	srand(time(0));
	for(int i=0; i<N; i++)
		for(int j=0; j<N; j++) {
			h_a[i*N + j] = rand()%N;
			h_b[i*N + j] = rand()%N;
		}

	// Allocate memory on device
	cudaMalloc((void**)&d_a, N*N*sizeof(int));
	cudaMalloc((void**)&d_b, N*N*sizeof(int));
	cudaMalloc((void**)&d_c, N*N*sizeof(int));

	//Copy data into device memory
	cudaMemcpy(d_a, h_a, N*N*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, N*N*sizeof(int), cudaMemcpyHostToDevice);

	// Kernel call
	dim3 grid(CEIL(N, 32), CEIL(N, 32), 1);
	dim3 block(32, 32, 1);

	add<<<grid, block>>>(d_a, d_b, d_c);
	cudaThreadSynchronize();

	// Copy data back to host
	cudaMemcpy(h_c, d_c, N*N*sizeof(int), cudaMemcpyDeviceToHost);

	// Free device memory
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	testSolution(h_a, h_b, h_c, 1e-6);	

	// Free host memory
	delete[] h_a, h_b, h_c;
}