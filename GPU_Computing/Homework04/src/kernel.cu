/******************************************************************************
 *
 *Computer Engineering Group, Heidelberg University - GPU Computing Exercise 04
 *
 *                  Group : TBD
 *
 *                   File : kernel.cu
 *
 *                Purpose : Memory Operations Benchmark
 *
 ******************************************************************************/


//
// Test Kernel
//

__global__ void 
globalMem2SharedMem(const float* src, float* out_float, size_t size)
{
    extern __shared__ float sharedData[];

    int threadId = threadIdx.x;
    int globalId = blockIdx.x * blockDim.x + threadId;

	int num_threads = blockDim.x * gridDim.x;

	int num_copies_per_thread = (size + num_threads - 1) / num_threads;

	for (int i = 0; i < num_copies_per_thread; i++){
		int globalIndex = globalId + i * gridDim.x * blockDim.x;
		int sharedIndex = threadId + i * blockDim.x;

		if (globalIndex < size) {
            sharedData[sharedIndex] = src[globalIndex];
        }
	}

    if (threadId == 0) *out_float = static_cast<float>(size);
}

void globalMem2SharedMem_Wrapper(dim3 gridSize, dim3 blockSize, int shmSize, const float* src, float* out_float, size_t size) {
	globalMem2SharedMem<<< gridSize, blockSize, shmSize >>>(src, out_float, size);
}

__global__ void 
SharedMem2globalMem(float* dest, size_t size )
{
	extern __shared__ float sharedData[];

    int threadId = threadIdx.x;
    int globalId = blockIdx.x * blockDim.x + threadId;

	int num_threads = blockDim.x * gridDim.x;

	int num_copies_per_thread = (size + num_threads - 1) / num_threads;

	for (int i = 0; i < num_copies_per_thread; i++){
		int globalIndex = globalId + i * gridDim.x * blockDim.x;
		int sharedIndex = threadId + i * blockDim.x;

		if (globalIndex < size) {
			dest[globalIndex] = sharedData[sharedIndex];
        }
	}
}
void SharedMem2globalMem_Wrapper(dim3 gridSize, dim3 blockSize, int shmSize, float* dest, size_t size) {
	SharedMem2globalMem<<< gridSize, blockSize, shmSize >>>(dest, size);
}

__global__ void 
SharedMem2Registers(size_t size)
{
	extern __shared__ float sharedData[];

	int threadId = threadIdx.x;

	if (threadId < size){
		float registerValue = sharedData[threadId];
	}
}
void SharedMem2Registers_Wrapper(dim3 gridSize, dim3 blockSize, int shmSize, size_t size) {
	SharedMem2Registers<<< gridSize, blockSize, shmSize >>>(size);
}

__global__ void 
Registers2SharedMem(size_t size)
{
	extern __shared__ float sharedData[];

	int threadId = threadIdx.x;

	float registerValue = 3.0f;

	if (threadId < size){
		sharedData[threadId] = registerValue;
	}
}
void Registers2SharedMem_Wrapper(dim3 gridSize, dim3 blockSize, int shmSize, size_t size) {
	Registers2SharedMem<<< gridSize, blockSize, shmSize >>>(size);
}

__global__ void 
bankConflictsRead
(size_t size, int stride, int iterations)
{
	extern __shared__ float sharedData[];

	int index = threadIdx.x * stride;

	if (index < size){
		float registerValue = sharedData[index];
	}
}

void bankConflictsRead_Wrapper(dim3 gridSize, dim3 blockSize, int shmSize, size_t size, int stride) {
	bankConflictsRead<<< gridSize, blockSize, shmSize >>>(size, stride, iterations);
}
