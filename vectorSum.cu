#include <bits/stdc++.h>
#include <cuda.h>
using namespace std;

#define N (1<<30)

__global__ void add(float *d_a, float *d_b, float *d_c) {

	int i = blockIdx.x;
	if(i < N)
		d_c[i] = d_b[i] + d_a[i];
}

int main() {

	float *h_a, *h_b, *h_c;
	float *d_a, *d_b, *d_c;
	clock_t tim;

	// Allocate host memory
	h_a = new float[N];
	h_b = new float[N];
	h_c = new float[N];

	printf("Number of elements in array : %d\n", N);

	srand(time(0));
	for(int i=0; i<2; i++) {
		h_a[i] = rand()%2;
		h_b[i] = rand()%2;
	}

	// Allocate memory on device
	cudaMalloc((void**)&d_a, N*sizeof(float));
	cudaMalloc((void**)&d_b, N*sizeof(float));
	cudaMalloc((void**)&d_c, N*sizeof(float));

	//Copy data into device memory
	cudaMemcpy(d_a, h_a, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, N*sizeof(float), cudaMemcpyHostToDevice);

	
	// Create a timer for device
	cudaEvent_t start, stop;
	float tims;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	// Kernel call
	add<<<N, 1>>>(d_a, d_b, d_c);
	cudaThreadSynchronize();

	// Stop times=r
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&tims, start, stop);
	printf("Device Time: %0.2lf us\n", 1000*tims);

	// Copy data back to host
	cudaMemcpy(h_c, d_c, N*sizeof(float), cudaMemcpyDeviceToHost);

	// Free device memory
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	tim = clock();
	for(int i=0; i<N; i++)
		h_c[i] = h_a[i] + h_b[i];
	tim = clock() - tim;
	printf("Host Time: %0.2lf s\n", tim*1.0/CLOCKS_PER_SEC);

	// Free host memory
	delete[] h_a, h_b, h_c;
}