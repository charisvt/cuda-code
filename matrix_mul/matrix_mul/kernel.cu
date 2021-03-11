
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdlib>
#include <iostream>
#include <cassert>
#include <stdio.h>
using namespace std;

__global__ void matrixMul(int *a, int *b, int *c, int N) {
    //calc row and column for each thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    //checks limits, stay within boundaries
    if (row < N && col < N) {
        int temp = 0;
        for (int i = 0; i < N; i++) {
            temp += a[row * N + i] * b[i * N + col];
        }
        c[row * N + col] = temp;
    }
}

void init_matrix(int* m, int N) {
    for (int i = 0; i < N; i++) {
        m[i] = rand() % 100;
    }
}

//verify result on cpu
void verify(int* a, int* b, int* c, int N) {
    int temp;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            temp=0;
            for (int k = 0; k < N; k++) {
                temp += a[i * N + k] * b[k * N + j];
            }

            assert(temp == c[i * N + j]);
        }
    }
}

int main()
{
    //define size
    int N = 1 << 10;
    size_t bytes = N * N * sizeof(int);

    //allocate unified memory
    int* a, * b, * c;
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    //init matrices 
    init_matrix(a, N);
    init_matrix(b, N);

    //set CTA and Grid dimensions
    int threads = 16;
    int blocks = (N + threads - 1) / threads;

    dim3 THREADS(threads, threads);
    dim3 BLOCKS(blocks, blocks);
    
    matrixMul<<<BLOCKS, THREADS>>>(a, b, c, N);
    
    cudaDeviceSynchronize();

    verify(a, b, c, N);
    cout << "Programm success" << endl;
    
    return 0;
}
