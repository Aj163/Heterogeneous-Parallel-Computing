#include <bits/stdc++.h>
#include <cuda.h>
using namespace std;

#define N ((int)1e3)
#define CEIL(a, b) ((a-1)/b +1)

__global__ void multiply(float *d_a, float *d_b, float *d_c) {

	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= N || y >= N)
		return;

	float cij=0.0;
	for(int k=0; k<N; k++)
		cij += d_a[x*N + k] * d_b[k*N + y];
	d_c[x*N + y] = cij;
}

template <class T>
void testSolution(T *h_a, T *h_b, T *h_c, float precision=0.0) {

	int errors = 0;
	for(int i=0; i<N; i++)
		for(int j=0; j<N; j++) {

			float exp = 0.0, act;
			for(int k=0; k<N; k++)
				exp += h_a[i*N + k] * h_b[k*N + j];
			act = h_c[i*N + j];

			if(abs(act-exp) / (max(exp, precision)) > precision) {
				
				errors++;
				if(errors <= 10)
					printf("Test failed at index : (%d, %d) [Expected: %10.2f | Got: %10.2f]\n", 
						i, j, exp, act);
			}
		}

	if(errors)
		printf("\n%d Tests failed!\n\n", errors);
	else
		printf("All tests passed !\n\n");
}

int main() {

	float *h_a, *h_b, *h_c;
	float *d_a, *d_b, *d_c;

	// Allocate host memory
	h_a = new float[N*N];
	h_b = new float[N*N];
	h_c = new float[N*N];

	printf("\nSize of matrices : %d x %d\n\n", N, N);

	srand(time(0));
	for(int i=0; i<N; i++)
		for(int j=0; j<N; j++) {
			h_a[i*N + j] = (rand()%N) *1.0/ (rand()%N +1);
			h_b[i*N + j] = (rand()%N) *1.0/ (rand()%N +1);
		}

	// Allocate memory on device
	cudaMalloc((void**)&d_a, N*N*sizeof(float));
	cudaMalloc((void**)&d_b, N*N*sizeof(float));
	cudaMalloc((void**)&d_c, N*N*sizeof(float));

	//Copy data into device memory
	cudaMemcpy(d_a, h_a, N*N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, N*N*sizeof(float), cudaMemcpyHostToDevice);

	// Kernel call
	dim3 grid(CEIL(N, 32), CEIL(N, 32), 1);
	dim3 block(32, 32, 1);

	multiply <<<grid, block>>> (d_a, d_b, d_c);
	cudaThreadSynchronize();

	// Copy data back to host
	cudaMemcpy(h_c, d_c, N*N*sizeof(float), cudaMemcpyDeviceToHost);

	// Free device memory
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	testSolution(h_a, h_b, h_c, 1e-3); //Tolerates 0.1% relative error

	// Free host memory
	delete[] h_a, h_b, h_c;
}