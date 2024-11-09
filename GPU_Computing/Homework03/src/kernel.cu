/*************************************************************************************************
 *
 *        Computer Engineering Group, Heidelberg University - GPU Computing Exercise 03
 *
 *                           Group : TBD
 *
 *                            File : main.cu
 *
 *                         Purpose : Memory Operations Benchmark
 *
 *************************************************************************************************/

#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <chCommandLine.h>
#include <chTimer.hpp>
#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>

//
// Kernels
//

__global__ void 
globalMemCoalescedKernel(int* out, const int* in, int size_in_bytes)
{
    int num_kernels = blockDim.x * gridDim.x;

    int size = size_in_bytes / sizeof(int);

    int copies_per_kernel = size + num_kernels - 1 / num_kernels;

    for (int i = 0; i < copies_per_kernel; i++){
        int index =  blockIdx.x * blockDim.x + threadIdx.x + i * num_kernels;
        if (index < size){
            //out[index] = 1;
        }
    }
}

void 
globalMemCoalescedKernel_Wrapper(dim3 gridDim, dim3 blockDim, int* out, const int* in, int size_in_bytes) {
	globalMemCoalescedKernel<<< gridDim, blockDim, 0 /*Shared Memory Size*/ >>>( out, in, size_in_bytes );
}

__global__ void 
globalMemStrideKernel(/*TODO Parameters*/)
{
    /*TODO Kernel Code*/
}

void 
globalMemStrideKernel_Wrapper(dim3 gridDim, dim3 blockDim /*TODO Parameters*/) {
	globalMemStrideKernel<<< gridDim, blockDim, 0 /*Shared Memory Size*/ >>>( /*TODO Parameters*/);
}

__global__ void 
globalMemOffsetKernel(/*TODO Parameters*/)
{
    /*TODO Kernel Code*/
}

void 
globalMemOffsetKernel_Wrapper(dim3 gridDim, dim3 blockDim /*TODO Parameters*/) {
	globalMemOffsetKernel<<< gridDim, blockDim, 0 /*Shared Memory Size*/ >>>( /*TODO Parameters*/);
}

