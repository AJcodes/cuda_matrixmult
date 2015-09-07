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

#define N 16
#define BLOCKSIZE 8

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
    const double a[N*N] = {73.0,7.0,87.0,29.0,68.0,53.0,16.0,6.0,65.0,41.0,63.0,38.0,17.0,31.0,91.0,42.0,
							66.0,100.0,32.0,100.0,22.0,84.0,53.0,4.0,78.0,49.0,63.0,1.0,50.0,98.0,55.0,48.0,
							87.0,19.0,37.0,65.0,37.0,55.0,88.0,9.0,1.0,78.0,99.0,75.0,58.0,51.0,62.0,29.0,
							20.0,22.0,7.0,32.0,47.0,41.0,55.0,50.0,16.0,81.0,76.0,77.0,7.0,15.0,47.0,91.0,
							8.0,97.0,68.0,99.0,5.0,21.0,67.0,21.0,45.0,95.0,19.0,52.0,48.0,39.0,39.0,38.0,
							20.0,88.0,52.0,100.0,45.0,62.0,52.0,94.0,12.0,56.0,11.0,74.0,71.0,53.0,34.0,1.0,
							68.0,16.0,68.0,27.0,85.0,27.0,33.0,30.0,66.0,46.0,17.0,36.0,61.0,24.0,93.0,81.0,
							63.0,47.0,71.0,41.0,2.0,18.0,67.0,4.0,23.0,30.0,35.0,19.0,36.0,59.0,1.0,37.0,
							71.0,42.0,22.0,16.0,95.0,12.0,66.0,32.0,100.0,5.0,66.0,90.0,52.0,20.0,1.0,30.0,
							31.0,51.0,89.0,79.0,52.0,21.0,100.0,96.0,33.0,3.0,49.0,49.0,53.0,45.0,49.0,7.0,
							26.0,8.0,84.0,78.0,91.0,90.0,94.0,88.0,30.0,26.0,25.0,98.0,24.0,74.0,70.0,9.0,
							10.0,58.0,17.0,92.0,24.0,15.0,85.0,7.0,80.0,8.0,67.0,35.0,27.0,50.0,89.0,47.0,
							30.0,85.0,47.0,77.0,86.0,52.0,21.0,15.0,94.0,30.0,87.0,42.0,56.0,57.0,66.0,86.0,
							17.0,1.0,89.0,43.0,67.0,66.0,33.0,10.0,64.0,88.0,69.0,22.0,71.0,62.0,84.0,28.0,
							21.0,68.0,86.0,5.0,100.0,45.0,72.0,96.0,77.0,23.0,30.0,49.0,6.0,63.0,21.0,67.0,
							50.0,63.0,13.0,17.0,89.0,29.0,80.0,57.0,18.0,39.0,6.0,14.0,14.0,57.0,59.0,38.0};
    const double b[N*N] = {49.0,66.0,27.0,23.0,94.0,81.0,98.0,59.0,63.0,54.0,50.0,90.0,29.0,31.0,1.0,57.0,
							63.0,99.0,64.0,44.0,96.0,90.0,56.0,56.0,76.0,96.0,79.0,47.0,69.0,4.0,9.0,76.0,
							25.0,42.0,70.0,67.0,80.0,30.0,12.0,50.0,11.0,87.0,17.0,98.0,54.0,19.0,70.0,45.0,
							77.0,71.0,5.0,96.0,96.0,67.0,68.0,33.0,77.0,69.0,1.0,8.0,74.0,15.0,85.0,57.0,
							10.0,69.0,47.0,33.0,90.0,57.0,82.0,21.0,20.0,38.0,47.0,32.0,15.0,56.0,87.0,61.0,
							41.0,38.0,16.0,94.0,85.0,89.0,87.0,12.0,50.0,89.0,69.0,31.0,15.0,8.0,1.0,27.0,
							22.0,82.0,73.0,27.0,5.0,34.0,58.0,39.0,14.0,54.0,2.0,7.0,15.0,63.0,41.0,38.0,
							72.0,66.0,74.0,75.0,3.0,84.0,69.0,32.0,4.0,67.0,80.0,12.0,60.0,13.0,57.0,48.0,
							29.0,7.0,27.0,72.0,41.0,19.0,47.0,86.0,35.0,60.0,79.0,88.0,10.0,36.0,51.0,40.0,
							6.0,78.0,30.0,21.0,3.0,34.0,38.0,55.0,46.0,12.0,78.0,28.0,26.0,39.0,57.0,17.0,
							11.0,61.0,57.0,62.0,45.0,82.0,94.0,16.0,58.0,67.0,22.0,77.0,81.0,80.0,100.0,97.0,
							98.0,52.0,60.0,97.0,99.0,90.0,87.0,100.0,93.0,47.0,59.0,10.0,13.0,100.0,3.0,36.0,
							5.0,6.0,27.0,67.0,84.0,21.0,58.0,39.0,80.0,97.0,91.0,99.0,98.0,45.0,98.0,30.0,
							73.0,42.0,20.0,63.0,65.0,14.0,39.0,54.0,61.0,51.0,63.0,4.0,12.0,34.0,11.0,13.0,
							14.0,51.0,66.0,94.0,41.0,9.0,3.0,86.0,9.0,49.0,96.0,16.0,41.0,34.0,82.0,4.0,
							48.0,68.0,58.0,12.0,68.0,6.0,52.0,66.0,30.0,20.0,91.0,31.0,93.0,60.0,82.0,73.0};
    double c[N*N] = { 0 };
	double c1[N*N] = { 0 };
	float naive_time = 0.0f;
	float tiling_time = 0.0f;

    cudaError_t cudaStatus = multCuda(c, c1, a, b, naive_time, tiling_time);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "multCuda failed!");
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

	cudaStatus = cudaThreadSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching naiveKernel!\n", cudaStatus);
        goto Error;
    }

	cudaEventRecord(start1);
	tilingKernel<<<dimGrid, dimBlock>>>(dev_c1, dev_a, dev_b);
	cudaEventRecord(stop1);
	cudaEventSynchronize(stop1);

	cudaStatus = cudaThreadSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching tilingKernel!\n", cudaStatus);
        goto Error;
    }

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
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
