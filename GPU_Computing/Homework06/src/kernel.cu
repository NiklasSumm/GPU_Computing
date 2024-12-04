/**************************************************************************************************
 *
 *       Computer Engineering Group, Heidelberg University - GPU Computing Exercise 06
 *
 *                 Gruppe : TODO
 *
 *                   File : kernel.cu
 *
 *                Purpose : Reduction
 *
 **************************************************************************************************/

#include <thrust/reduce.h>
#include <thrust/device_ptr.h>

//
// Reduction_Kernel
//
__global__ void
reduction_Kernel(int numElements, float* dataIn, float* dataOut)
{
  	extern __shared__ float sh_Data[];

  	int elementId = blockIdx.x * blockDim.x + threadIdx.x;

  	sh_Data[threadIdx.x] = dataIn[elementId];

  	__syncthreads();

  	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
    	if ((threadIdx.x % (2 * s)) == 0) {
      		sh_Data[threadIdx.x] += sh_Data[threadIdx.x + s];
    	}
    	__syncthreads();
  	}

  	if (threadIdx.x == 0) dataOut[blockIdx.x] = sh_Data[0];
}

void reduction_Kernel_Wrapper(dim3 gridSize, dim3 blockSize, int numElements, float* dataIn, float* dataOut) {
	int sharedMemSize = numElements * sizeof(float) / gridSize.x;
	reduction_Kernel<<< gridSize, blockSize, sharedMemSize>>>(numElements, dataIn, dataOut);
}

__global__ void
reduction_Kernel_improved(int numElements, float* dataIn, float* dataOut)
{
	extern __shared__ float sh_Data[];

  	int elementId = 2 * blockIdx.x * blockDim.x + threadIdx.x;

  	sh_Data[threadIdx.x] = dataIn[elementId] + dataIn[elementId + blockDim.x];

  	__syncthreads();

  	for ( unsigned int o = blockDim.x / 2; o > 0; o >>= 1 ) {
		if (threadIdx.x < o ) {
			sh_Data[threadIdx.x] += sh_Data[threadIdx.x + o];
		}
		__syncthreads();
	}

  	if (threadIdx.x == 0) dataOut[blockIdx.x] = sh_Data[0];
}

void reduction_Kernel_improved_Wrapper(dim3 gridSize, dim3 blockSize, int numElements, float* dataIn, float* dataOut) {
	int sharedMemSize = numElements * sizeof(float) / (gridSize.x * 2);
	reduction_Kernel_improved<<< gridSize, blockSize, sharedMemSize>>>(numElements, dataIn, dataOut);
}

//
// Reduction Kernel using CUDA Thrust
//

void thrust_reduction_Wrapper(int numElements, float* dataIn, float* dataOut) {
	thrust::device_ptr<float> in_ptr = thrust::device_pointer_cast(dataIn);
	thrust::device_ptr<float> out_ptr = thrust::device_pointer_cast(dataOut);
	
	*out_ptr = thrust::reduce(in_ptr, in_ptr + numElements, (float) 0., thrust::plus<float>());	
}
