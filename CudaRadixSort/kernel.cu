#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define blockSize 128

__global__ void kernelCount(int* G_arr,int *G_Count,int dataSize,int digit) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < dataSize)
        atomicAdd(&(G_Count[(G_arr[idx] / digit) % 10]), 1);
    else
        return;
}

__global__ void getMaxOnGpu(const int* G_arr, int dataSize, int* Out) {
    int tid = threadIdx.x;
    int gthIdx = tid + blockIdx.x * blockSize;
    const int gridSize = blockSize * gridDim.x;
    int sum = 0;
    for (int i = gthIdx; i < dataSize; i += gridSize)
        sum += G_arr[i];
    __shared__ int particalMax[blockSize];
    particalMax[tid] = sum;
    __syncthreads();
    for (int stride = blockSize / 2; stride> 0; stride>>= 1) { 
        if (tid < stride)
            if (particalMax[tid] < particalMax[tid + stride])
                particalMax[tid] = particalMax[tid + stride];
        __syncthreads();
    }
    if (tid == 0)
        Out[blockIdx.x] = particalMax[0];
}

__global__ void kernelOutput(int* G_arr, int* Output, int dataSize) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < dataSize)
        G_arr[tid] = Output[tid];
    else
        return;
}

__host__ void Radix(int *C_arr,int *C_Output,int *Count,int dataSize, int digit) {

    for (int i = 1; i < 10; i++)
        Count[i] += Count[i - 1];

    for (int i = dataSize - 1; i >= 0; i--)
    {
        C_Output[Count[(C_arr[i] / digit) % 10] - 1] = C_arr[i];
        Count[(C_arr[i] / digit) % 10]--;
    }

}

void initializeArr(int* C_arr, int dataSize) {

    time_t t;
    srand((unsigned)time(&t));

    for (int i = 0; i < dataSize; i++)
        C_arr[i] = rand() % dataSize;

    printf("Array was constructed\n");

}

void printArr(int* C_arr, int dataSize) {

    for (int i = 0; i < dataSize; i++)
        printf("%d \n", C_arr[i]);

}

int main(int argc,char *argv[])
{   

    double runTime;
    int *C_arr, *G_arr, *Count,*C_Output, *Output,*G_Count;
    int* max, * GpuMax, reduceSize, dataSize = 1000;

    if (argc == 2) {
        dataSize = atoi(argv[1]);
    }

    clock_t gpuTime;
    size_t dataBytes = dataSize * sizeof(int);
    size_t countBytes = 10 * sizeof(int);

    dim3 threads(128);
    dim3 blocks(128);

    reduceSize = blocks.x;

    C_arr = (int*)malloc(dataBytes);
    C_Output = (int*)malloc(dataBytes);

    Count = (int*)malloc(countBytes);
    max = (int*)malloc(sizeof(int));

    cudaMalloc((int**)&G_arr,dataBytes);
    cudaMalloc((int**)&Output,dataBytes);

    cudaMalloc((int**)&G_Count, countBytes);
    cudaMalloc((int**)&GpuMax, sizeof(int));

    initializeArr(C_arr, dataSize);

    cudaMemcpy(G_arr, C_arr, dataBytes, cudaMemcpyHostToDevice);

    gpuTime = clock();

    cudaDeviceSynchronize();

    getMaxOnGpu << <blocks, threads>> > (G_arr, dataSize, GpuMax);
   
    getMaxOnGpu<< <1, threads >> > (GpuMax, reduceSize, GpuMax);

    cudaMemcpy(max, GpuMax, sizeof(int), cudaMemcpyDeviceToHost);

    for (int digit = 1; *max / digit > 0; digit *= 10)
    {
        cudaMemset(G_Count, 0, countBytes);
        kernelCount << < blocks, threads >> > (G_arr, G_Count, dataSize, digit);
        cudaMemcpy(Count, G_Count, countBytes, cudaMemcpyDeviceToHost);
        Radix(C_arr, C_Output, Count ,dataSize, digit);
        cudaMemcpy(Output, C_Output, dataBytes, cudaMemcpyHostToDevice);
        kernelOutput << <blocks, threads >> > (G_arr, Output, dataSize);
        cudaMemcpy(C_arr, G_arr, dataBytes, cudaMemcpyDeviceToHost);
    }
  
    gpuTime = clock() - gpuTime;
    
    runTime = ((double)gpuTime) / CLOCKS_PER_SEC;
    
    cudaFree(G_arr);
    cudaFree(Output);
    cudaFree(G_Count);
    cudaFree(GpuMax);

    printArr(C_arr, dataSize);

    free(C_arr);
    free(C_Output);
    free(Count);
    free(max);

    printf("%d dimensional array was sorted in %f seconds\n",dataSize, runTime);

    cudaDeviceReset();

    return 0;
}
