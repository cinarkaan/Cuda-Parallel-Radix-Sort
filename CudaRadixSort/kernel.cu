#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <chrono>
#include <iostream>
#include <stdlib.h>
#include <time.h>

#define blockSize 1024

//CountSort algoritmas�nda basamaklar�n tekrar say�s�n� gpu �zerinde paralel hesapla
__global__ void kernelCount(int* G_arr,int *G_Count,int dataSize,int digit) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < dataSize)
        atomicAdd(&(G_Count[(G_arr[idx] / digit) % 10]), 1);
    else
        return;
}

//CountSort en b�y�k say�n�n basamak say�s� kadar tekrar edecek.En b�y�k say�y� gpu �zerinde paralel hesapla.---> O(log2n)
__global__ void getMaxOnGpu(const int* G_arr, int dataSize, int* Out) {
    
    //Global bellekten payla��lan belle�e dizi kopyalan�yor
    int tid = threadIdx.x;
    int gthIdx = tid + blockIdx.x * blockSize;
    const int gridSize = blockSize * gridDim.x;
    int sum = 0;
    for (int i = gthIdx; i < dataSize; i += gridSize)
        sum += G_arr[i];
    __shared__ int particalMax[blockSize];
    particalMax[tid] = sum;
    __syncthreads();

    //Par�a di�er yar�s� ile kar��la�d�r�l�yor
    for (int stride = blockDim.x / 2; stride> 0; stride>>= 1) { 
        if (tid < stride)
            if (particalMax[tid] < particalMax[tid + stride])
                particalMax[tid] = particalMax[tid + stride];
        __syncthreads();
    }
    if (tid == 0) 
        Out[blockIdx.x] = particalMax[0];
}

//Basamaklar� s�ralanan diziyi orjinal diziye gpu �zerinde paralel olarak kopyala
__global__ void kernelOutput(int* G_arr, int* Output, int dataSize) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < dataSize)
        G_arr[tid] = Output[tid];
    else
        return;
}

//Cpu �zerinde basamaklar� s�rala
__host__ void Radix(int *C_arr,int *C_Output,int *Count,int dataSize, int digit) {

    for (int i = 1; i < 10; i++)
        Count[i] += Count[i - 1];

    for (int i = dataSize - 1; i >= 0; i--)
    {
        C_Output[Count[(C_arr[i] / digit) % 10] - 1] = C_arr[i];
        Count[(C_arr[i] / digit) % 10]--;
    }

}

//Diziyi yap�land�r
void initializeArr(int* C_arr, int dataSize) {

    time_t t;
    srand((unsigned)time(&t));

    for (int i = 0; i < dataSize; i++)
        C_arr[i] = rand() % dataSize;

    printf("Array was constructed\n");

}

//S�ral� diziyi ekrana bas
void printArr(int* C_arr, int dataSize) {

    for (int i = 0; i < dataSize; i++)
        printf("%d \n", C_arr[i]);

}

int main(int argc,char *argv[])
{   

    //Ram ve VRam i�inde saklanacak de�i�kenler
    int *C_arr, *G_arr, *Count,*C_Output, *Output,*G_Count;
    int* max, * GpuMax, reduceSize, dataSize = 1000000;

    //Program komut sat�r�ndan �al��t���nda d��ardan al�nacak de�er
    if (argc == 2) {
        dataSize = atoi(argv[1]);
    }

    //De�i�kenlerin byte cinsinden boyutu
    size_t dataBytes = dataSize * sizeof(int);
    size_t countBytes = 10 * sizeof(int);

    //Blok ve thread tan�mlar� yap�l�yor
    dim3 threads(1024);
    dim3 blocks(blockSize);

    reduceSize = blocks.x;

    //Verilen Boyut kadar ramde diziyi olu�tur
    C_arr = (int*)malloc(dataBytes);
    C_Output = (int*)malloc(dataBytes);

    Count = (int*)malloc(countBytes);
    max = (int*)malloc(sizeof(int));

    //Verilen boyut kadar Vramde diziyi olu�tur
    cudaMalloc((int**)&G_arr,dataBytes);
    cudaMalloc((int**)&Output,dataBytes);

    cudaMalloc((int**)&G_Count, countBytes);
    cudaMalloc((int**)&GpuMax, sizeof(int));

    //S�ralanacak diziyi yap�land�r
    initializeArr(C_arr, dataSize);

    //Yap�land�r�lan diziyi ramdan vram i�erisine aktar
    cudaMemcpy(G_arr, C_arr, dataBytes, cudaMemcpyHostToDevice);

    //Y�ksek hassasiyetli say�c�y� �al��t�r
    auto start = std::chrono::high_resolution_clock::now();

    //En b�y�k de�eri bulup getiren kernel fonksiyonlar�
    getMaxOnGpu << <blocks, threads>> > (G_arr, dataSize, GpuMax);
   
    getMaxOnGpu<< <1, threads >> > (GpuMax, reduceSize, GpuMax);

    //Bulunan en b�y�k de�eri ram e getir
    cudaMemcpy(max, GpuMax, sizeof(int), cudaMemcpyDeviceToHost);

    //CountSort algoritmas�n� �al��t�r
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
    
    //Algoritma bitti y�ksek hassasiyetli say�c�y� durdur.
    auto stop = std::chrono::high_resolution_clock::now();

    //Algoritman�n �al��mas� ne kadar zaman ald�
    auto Duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    //Vram temizle
    cudaFree(G_arr);
    cudaFree(Output);
    cudaFree(G_Count);
    cudaFree(GpuMax);

    //printArr(C_arr, dataSize);

    //Ram temizle
    free(C_arr);
    free(C_Output);
    free(Count);
    free(max);

    std::cout << std::endl;

    std::cout << dataSize << " elemented array was sorted in " << Duration.count() << " microseconds " << "on gpu" <<std::endl;

    //Kullan�lan cuda kaynaklar�n� temizle
    cudaDeviceReset();

    return 0;
}
