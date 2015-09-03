#ifndef __CUDACC__
#define __CUDACC__
#endif
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <curand_kernel.h>

#include <stdio.h>
#include <iostream>
#include <iomanip>

#define N 8
#define BLOCKSIZE 2

cudaError_t multCuda(double *c, double *c1, const double *a, const double *b, float &naive_time, float &tiling_time);

__global__ void naiveKernel(double *c, const double *a, const double *b) {
	double temp = 0;
    
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

	for (int k = 0; k < N; ++k)
		temp += a[i*N+k] * b[k*N+j];

	c[i*N+j] = temp;
}

__global__ void tilingKernel(double *c, const double *a, const double *b) {
	__shared__ double A_tile[BLOCKSIZE][BLOCKSIZE];
    __shared__ double B_tile[BLOCKSIZE][BLOCKSIZE];

    int row = blockIdx.y * BLOCKSIZE + threadIdx.y;
    int col = blockIdx.x * BLOCKSIZE + threadIdx.x;
    double temp = 0;

    for (int i = 0; i < (N - 1) / BLOCKSIZE + 1; ++i) {
       A_tile[threadIdx.y][threadIdx.x] = a[row * N + i * BLOCKSIZE + threadIdx.x];
       B_tile[threadIdx.y][threadIdx.x] = b[(i * BLOCKSIZE + threadIdx.y) * N + col]; // No Shared Mem Bank conflict
	   __syncthreads();

       for (int k = 0; k < BLOCKSIZE; ++k)
          temp += A_tile[threadIdx.y][k] * B_tile[k][threadIdx.x]; // No Shared Mem Bank conflict
       __syncthreads();
    }

    c[row*N+col] = temp;
}

void coutMatrix(int d, double *m) { 
	std::cout << std::endl;
	for	(int i = 0; i < d; ++i) {
		for	(int j = 0; j < d; ++j) 
			std::cout << std::setw(9) << m[i*d+j];
		std::cout << std::endl;
	}
}

int main()
{
    const double a[N*N] = {2.0,1.0,1.0,3.0,2.0,1.0,2.0,2.0,
						1.0,2.0,2.0,1.0,1.0,1.0,2.0,9.0,
						1.0,2.0,9.0,1.0,5.0,3.0,1.0,1.0,
						3.0,1.0,1.0,7.0,1.0,2.0,1.0,5.0,
						2.0,1.0,5.0,1.0,8.0,2.0,1.0,1.0,
						5.0,3.0,1.0,1.0,2.0,1.0,5.0,1.0,
						1.0,1.0,2.0,9.0,1.0,2.0,9.0,1.0,
						8.0,2.0,1.0,1.0,3.0,1.0,1.0,7.0};
    const double b[N*N] = {2.0,1.0,1.0,3.0,2.0,1.0,2.0,2.0,
						1.0,2.0,2.0,1.0,1.0,1.0,2.0,9.0,
						1.0,2.0,9.0,1.0,5.0,3.0,1.0,1.0,
						3.0,1.0,1.0,7.0,1.0,2.0,1.0,5.0,
						2.0,1.0,5.0,1.0,8.0,2.0,1.0,1.0,
						5.0,3.0,1.0,1.0,2.0,1.0,5.0,1.0,
						1.0,1.0,2.0,9.0,1.0,2.0,9.0,1.0,
						8.0,2.0,1.0,1.0,3.0,1.0,1.0,7.0};
    double c[N*N] = { 0 };
	double c1[N*N] = { 0 };
	float naive_time = 0.0f;
	float tiling_time = 0.0f;

    cudaError_t cudaStatus = multCuda(c, c1, a, b, naive_time, tiling_time);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

	std::cout << "Naive GPU Implementation" << std::endl;
    coutMatrix(N,c);
	std::cout << "Execution Time : " << naive_time / 1000 << " seconds" << std::endl;
	std::cout << "Effective Bandwidth : " << (N*N*sizeof(double)*2) / (naive_time / 1000) << " GB/s" << std::endl;
	std::cout << std::endl;

	std::cout << "Tiling GPU Implementation" << std::endl;
    coutMatrix(N,c1);
	std::cout << "Execution Time : " << tiling_time / 1000 << " seconds" << std::endl;
	std::cout << "Effective Bandwidth : " << (N*N*sizeof(double)*2) / (tiling_time / 1000) << " GB/s" << std::endl;
	std::cout << std::endl;

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

cudaError_t multCuda(double *c, double *c1, const double *a, const double *b, float &naive_time, float &tiling_time)
{
    double *dev_a = 0;
    double *dev_b = 0;
    double *dev_c = 0;
	double *dev_c1 = 0;
	float milliseconds = 0;
	float milliseconds1 = 0;
	dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);
    dim3 dimGrid(N/dimBlock.x, N/dimBlock.y);
    cudaError_t cudaStatus;
	cudaEvent_t start, stop;
	cudaEvent_t start1, stop1;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&start1);
	cudaEventCreate(&stop1);

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_c, (N * N) * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	cudaStatus = cudaMalloc((void**)&dev_c1, (N * N) * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, (N * N) * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, (N * N) * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_a, a, (N * N) * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, (N * N) * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	cudaEventRecord(start);
    naiveKernel<<<dimGrid, dimBlock>>>(dev_c, dev_a, dev_b);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	cudaEventRecord(start1);
	tilingKernel<<<dimGrid, dimBlock>>>(dev_c1, dev_a, dev_b);
	cudaEventRecord(stop1);
	cudaEventSynchronize(stop1);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMemcpy(c, dev_c, (N * N) * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	cudaStatus = cudaMemcpy(c1, dev_c1, (N * N) * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	cudaEventElapsedTime(&milliseconds, start, stop);
	cudaEventElapsedTime(&milliseconds1, start1, stop1);
	naive_time = milliseconds;
	tiling_time = milliseconds1;

Error:
    cudaFree(dev_c);
	cudaFree(dev_c1);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
