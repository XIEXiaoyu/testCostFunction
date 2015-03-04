#include <stdio.h> 
#include <stdlib.h> 
#include <cuda.h> 
#include <cuda_runtime.h>  // for cudaEvent
#include <curand_kernel.h> // for curandState 

_global_ void setup_randSeed(curandState *globalState, unsigned long seed)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(seed, idx, 0, &globalState[idx]);
}

int main(void) {

	int *device_array = 0;
	int *host_array = 0;
	unsigned long seed = 12; // randomly chosen

	// malloc host memory
	host_array = (curandState *)malloc(sizeof(curandState));

	// cudamalloc device memory
	cudaMalloc((void**)&device_array, curandState);

	// if either memory allocation failed, report an error message
    if(host_array == 0 || device_array == 0)
    {
    	printf("couldn't allocate memory\n");
    	return 1;
    }

    // give input values to the pointer host_array(i.e. struct curandStateXORWOW).
	host_array->d = 12;
	host_array->v[] = {0,1,2,3,4};
	host_array->boxmuller_flag = 2;
	host_array->boxmuller_extra = 0.1;
	host_array->boxmuller_extra_double = 0.2;

	// copy the value from host to device
	cudaMemcpy(host_array, device_array, sizeof(curandState), cudaMemcpyHostToDevice); 

	// set up time test
	float gpuElapsedTime ; 
	cudaEvent_t gpuStart, gpuStop; 
	cudaEventCreate(&gpuStart); 
	cudaEventCreate(&gpuStop); 
	cudaEventRecord(gpuStart,0);

	setup_randSeed<<<1,1>>>(device_array, seed); // need to give two input

	// get time test result
	cudaEventRecord(gpuStop,0);
	cudaEventSynchronize(gpuStop);
	cudaEventElapsedTime(&gpuElapsedTime, gpuStart, gpuStop) ; // time in milliseconds
	cudaEventDestroy(gpuStart); 
	cudaEventDestroy(gpuStop);

	//print the time
	printf("GPU Time elapsed: %f seconds\n", gpuElapsedTime/1000.0);

	return 0;
}