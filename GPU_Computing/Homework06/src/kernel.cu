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
reduction_Kernel(int numElements, int numElementsShared, float* dataIn, float* dataOut)
{
  	extern __shared__ float sh_Data[];

	int totalThreads = blockDim.x * gridDim.x;

	int copiesPerThread = numElements + totalThreads + 1 / totalThreads;

	for (int i = 0; i < copiesPerThread; i++){
		int elementId = copiesPerThread * blockIdx.x * blockDim.x + threadIdx.x + i * blockDim.x;
		int sharedId = threadIdx.x + i * blockDim.x;
		if (sharedId < numElementsShared){
			if (elementId < numElements){
				sh_Data[sharedId] = dataIn[elementId];
			}
			else{
				sh_Data[sharedId] = 0;
			}
		}
	}

  	__syncthreads();

  	for (unsigned int s = 1; s < blockDim.x * copiesPerThread; s *= 2) {
		for (int i = 0; i < copiesPerThread; i++){
			int id = threadIdx.x + i * blockDim.x;
    		if ((id % (2 * s)) == 0) {
				if (id + s < numElementsShared){
					sh_Data[id] += sh_Data[id + s];
				}
    		}
    		__syncthreads();
  		}
	}

  	if (threadIdx.x == 0) dataOut[blockIdx.x] = sh_Data[0];
}

void reduction_Kernel_Wrapper(dim3 gridSize, dim3 blockSize, int numElements, float* dataIn, float* dataOut) {
	int sharedMemSize = numElements * sizeof(float) / gridSize.x;
	reduction_Kernel<<< gridSize, blockSize, sharedMemSize>>>(numElements, sharedMemSize / sizeof(float), dataIn, dataOut);
}

__global__ void
reduction_Kernel_improved(int numElements, int numElementsShared, float* dataIn, float* dataOut)
{
	extern __shared__ float sh_Data[];

  	int tid = 2 * blockIdx.x * blockDim.x + threadIdx.x;

  	sh_Data[threadIdx.x] = dataIn[tid] + dataIn[tid + blockDim.x];

  	__syncthreads();

  	for ( unsigned int o = blockDim.x / 2; o > 0; o >>= 1 ) {
		if (threadIdx.x < o ) {
			if (tid < numElements){
				sh_Data[threadIdx.x] += sh_Data[threadIdx.x + o];
			}
		}
		__syncthreads();
	}

  	if (threadIdx.x == 0) dataOut[blockIdx.x] = sh_Data[0];
}

void reduction_Kernel_improved_Wrapper(dim3 gridSize, dim3 blockSize, int numElements, float* dataIn, float* dataOut) {
	int sharedMemSize = numElements * sizeof(float) / (gridSize.x * 2);
	reduction_Kernel_improved<<< gridSize, blockSize, sharedMemSize>>>(numElements, sharedMemSize / sizeof(float), dataIn, dataOut);
}

//
// Reduction Kernel using CUDA Thrust
//

void thrust_reduction_Wrapper(int numElements, float* dataIn, float* dataOut) {
	thrust::device_ptr<float> in_ptr = thrust::device_pointer_cast(dataIn);
	thrust::device_ptr<float> out_ptr = thrust::device_pointer_cast(dataOut);
	
	*out_ptr = thrust::reduce(in_ptr, in_ptr + numElements, (float) 0., thrust::plus<float>());	
}
