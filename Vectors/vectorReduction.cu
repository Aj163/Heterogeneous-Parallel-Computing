#include <bits/stdc++.h>
#include <cuda.h>
using namespace std;

#define N ((int)1e7)
#define CEIL(a, b) ((a-1)/b +1)

__global__ void reduce(int *d_a, int *sum) {

	__shared__ int data[1024];
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	// Copy all elements in block to shared memory and wait
	data[threadIdx.x] = d_a[i];
	__syncthreads();

	for(int step=1; step<1024; step*=2) {
		int threadID = 2*step*threadIdx.x;
		if(threadID + step < 1024)
			data[threadID] += data[threadID + step];

		__syncthreads();
	}

	if(threadIdx.x == 0)
		atomicAdd(sum, data[0]);
}

int main() {

	int *h_a, *h_sum;
	int *d_a, *d_sum;
	clock_t tim;

	h_a = new int[N];
	h_sum = new int;
	*h_sum = 0;

	printf("\nValue of N   : %d\n", N);

	srand(time(0));
	for(int i=0; i<N; i++)
		h_a[i] = rand()%2;

	cudaMalloc((void**)&d_a, N*sizeof(int));
	cudaMalloc((void**)&d_sum, sizeof(int));

	cudaMemcpy(d_a, h_a, N*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_sum, h_sum, sizeof(int), cudaMemcpyHostToDevice);

	// Device timer	
	cudaEvent_t start, stop;
	float tims;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	// Kernel call
	reduce<<<CEIL(N, 1024), 1024>>>(d_a, d_sum);
	cudaThreadSynchronize();

	// End timer
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&tims, start, stop);
	printf("\nDevice Time : %0.2lf ms\n", tims);

	cudaMemcpy(h_sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_sum);

	int sum = 0;
	tim = clock();
	for(int i=0; i<N; i++)
		sum += h_a[i];
	tim = clock() - tim;
	printf("Host Time   : %0.2lf ms\n", tim*1.0/CLOCKS_PER_SEC*1000.0);

	printf("\nDevice sum  : %d\nHost sum    : %d\n\n", *h_sum, sum);

	delete[] h_a;
	delete h_sum;
}	