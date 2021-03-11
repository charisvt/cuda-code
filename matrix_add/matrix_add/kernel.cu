#include <stdio.h>
#include <cuda_runtime.h>
#define SIZE 1024

__global__ void VectorAdd(int *a, int *b, int *c, int n) { // __global__ is used to define a function that runs on the gpu
	int i = threadIdx.x;  //threadIdx. is used to get the thread number (can have x,y,z coordinates)
	if (i < n) { // stay within limits
		c[i] = a[i] + b[i]; //just add them, no loop required since they run on parallel threads
	}
}

int main() {
	int *a, *b, *c; 
	cudaMallocManaged(&a, SIZE * sizeof(int)); //allocate unified memory
	cudaMallocManaged(&b, SIZE * sizeof(int));
	cudaMallocManaged(&c, SIZE * sizeof(int));


	for (int i = 0; i < SIZE; i++) {
		a[i] = i;
		b[i] = i*i;
		c[i] = 0;
	}

	VectorAdd <<<1, SIZE>>> (a, b, c, SIZE); //use <<<>>> to call a function that runs on gpu
	cudaDeviceSynchronize(); //this is needed to make the cpu wait for all the gpu threads to finish before we continue

	int sizeToShow = 10;
	for (int i = 0; i < sizeToShow; i++) {
		printf("c[%d] = %d\n", i, c[i]);
	}
	
	cudaFree(a); //free unified memory
	cudaFree(b);
	cudaFree(c);
	return 0;
}