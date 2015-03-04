// This file is to time GPU compuatition time of cost_function, for 10, 100, 1000, etc blocks
// XieJun

#include <stdio.h>
#include <stdlib.h> // for malloc()
#include <assert.h> // for assert() debugging
#define NETS 10
#define SIZE NETS * NETS

#include <cuda.h>
#include <cuda_runtime.h>
//#define DEBUG_WHOLE
//#define DEBUG

__device__ int max(int i, int j) {
	int maximum;
	if (i < j) {
		maximum = j;
	}
	else maximum = i;

	return maximum;
}

__device__ int min(int i , int j) {
	int minimum;
	if (i < j) {
		minimum = i;
	}
	else minimum = j;

	return minimum;
}

__global__ void cost_function(int i, int* netOffset, int* blockID, int *X, int* Y, int SIZE_X, int SIZE_Y) {
	float elapsedTime;

	#ifdef DEBUG_WHOLE
	clock_t start = clock();
	#endif

	//for easy timing: section 1
	#ifdef DEBUG
	clock_t start = clock();
	#endif

	float cross_count[50] = {  /* [0..49] */
		1.0, 1.0, 1.0, 1.0828, 1.1536, 1.2206, 1.2823, 1.3385, 1.3991, 1.4493,
		1.4974, 1.5455, 1.5937, 1.6418, 1.6899, 1.7304, 1.7709, 1.8114, 1.8519,
		1.8924,
		1.9288, 1.9652, 2.0015, 2.0379, 2.0743, 2.1061, 2.1379, 2.1698, 2.2016,
		2.2334,
		2.2646, 2.2958, 2.3271, 2.3583, 2.3895, 2.4187, 2.4479, 2.4772, 2.5064,
		2.5356,
		2.5610, 2.5864, 2.6117, 2.6371, 2.6625, 2.6887, 2.7148, 2.7410, 2.7671,
		2.7933
	};

	#ifdef DEBUG
	elapsedTime = ((double)clock() - start) / CLOCKS_PER_SEC;
	printf("t1 is: %f seconds \n", elapsedTime);
	#endif
	// for easy timing: end of section 1

	// for easy timing: section 2
	#ifdef DEBUG
	start = clock();
	#endif
	int block_id=blockID[netOffset[i]];
	int srcx=X[block_id];
	int srcy=Y[block_id];

	int max_x=srcx, min_x=srcx;
	int max_y=srcy, min_y=srcy;

	#ifdef DEBUG
	elapsedTime = ((double)clock() - start) / CLOCKS_PER_SEC;
	printf("t2 is: %f seconds \n", elapsedTime);
	#endif
	// for easy timing: section 2

	// for easy timing: section 3
	#ifdef DEBUG
	start = clock();
	#endif
	// loop over all nets and compute bounding box
	int sinks=netOffset[i+1]-netOffset[i];
	//	assert(sinks>=0);

	for(int j=1;j<sinks;++j) {
		block_id=blockID[netOffset[i]+j];

		max_x=max(max_x,X[block_id]);
		max_y=max(max_y,Y[block_id]);

		min_x=min(min_x,X[block_id]);
		min_y=min(min_y,Y[block_id]);
	}

	#ifdef DEBUG
	elapsedTime = ((double)clock() - start) / CLOCKS_PER_SEC;
	printf("t3 is: %f seconds \n", elapsedTime);
	#endif
	// for easy timing: end of section 3

	// for easy timing: section 4
	#ifdef DEBUG
	start = clock();
	#endif

	max_x=min(max_x,SIZE_X);
	max_y=min(max_y,SIZE_Y);

	float crossing;
	if(sinks>50) {
		crossing = 2.7933 + 0.02616 * (sinks-50);

	} else {
		crossing = cross_count[sinks-1];
	}

	float chanx_place_cost_fac = 0.01; // printffed from VPR

	float cost_DEV = (max_x-min_x+1)*crossing*chanx_place_cost_fac+
		(max_y-min_y+1)*crossing*chanx_place_cost_fac; 
		// needs channel width related scaling factors 

	#ifdef DEBUG
	elapsedTime = ((double)clock() - start) / CLOCKS_PER_SEC;
	printf("t4 is: %f seconds \n", elapsedTime);
	#endif

	#ifdef DEBUG_WHOLE
	elapsedTime = ((double)clock() - start) / CLOCKS_PER_SEC;
	printf("Whole cost_function running time is: %f seconds \n", elapsedTime);
	#endif

	return cost_DEV; 
	// for easy timing: end of section 4
}

int main(void) {
	
	int i = 0, j = 0; 
	float cost = 0;

	// pointers to host & device
	int * netOffset = 0;
	int * netOffset_Dev = 0;

	// set number of nets and how many blocks in a net
	// in our program, each net have NETS number of blocks
	i = 0; int num_bytes = (NETS + 1) * sizeof(int);
	netOffset = (int*)malloc(num_bytes * sizeof(int));
	for(i = 1; i < NETS + 1; i++) {
		netOffset[i] = i * NETS;
	}

	cudaMalloc((int**)&netOffset_Dev, num_bytes);
	cudaMemcpy(netOffset, netOffset_Dev, num_bytes, cudaMemcpyHostToDevice);

	// set which blocks are in which net
	// for example, if NET = 4 , block = 4 * 4 = 16
	// the structure is as below:
	/*
	+--+--+--+--+
	|0 |4 |8 |12| block 0, 4... is in net 0; block 8 is at location(0, 2);
	+--+--+--+--+
	|1 |5 |9 |13| block 1, 5... is in net 1; block 5 is at location(1, 1);
	+--+--+--+--+
	|2 |6 |10|14| ...
	+--+--+--+--+
	|3 |7 |11|15| ...
	+--+--+--+--+
	*/
	// this structure is used so that the blocks in each net are scattered, 
	// so the X[bloc_id] and Y[block_id] in the X array and Y array can not be bring into the cache effectively,
	// this mimics the real situlation
	// pointers to host & device
	int * blockID = 0; 
	int * blockID_Dev = 0;
	int row = 0, col = 0;
	int * X = 0, * Y = 0;
	int * X_Dev = 0, *Y_Dev = 0;

	num_bytes = SIZE * sizeof(int);
	

	X = (int *)malloc(mum_bytes);
	assert(X != NULL);
	Y = (int *)malloc(mum_bytes); 
	assert(Y != NULL);

	cudaMalloc((int**)X_Dev, num_bytes);
	cudaMalloc((int**)Y_Dev, num_bytes);

	blockID = (int *)malloc(sizeof(int) * SIZE);
	for (row = 0; row < NETS; row++) {
		for (col = 0; col < NETS; col++) {
			int pos = row * NETS + col;
			int block_index = col * NETS + row;
			blockID[pos] = block_index;
			X[block_index] = row;
			Y[block_index] = col;
			// printf("%3d ", blockID[pos]);  // for debugging
		}
		// printf("\n");  // for debugging
	}

	cudaMalloc((int**)&blockID_Dev, num_bytes);
	cudaMemcpy(blockID, blockID_Dev, num_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(X, X_Dev, num_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(Y, Y_Dev, num_bytes, cudaMemcpyHostToDevice);

	// for debugging
	// printf("netOfset[2] is : %d\n", netOffset[2]);
	// printf("netOfset[3] is : %d\n", netOffset[3]);
	// for (j = 0; j < NETS; j++) {
	// 	printf("netOfset[2] + %d is: %d,\n", j, netOffset[2] + j);
	// 	int kk = netOffset[2] + j;
	// 	printf("kk is : %d,\n", kk);
	// 	printf("blockID[%d] is : %d,\n", kk,blockID[kk]);	
	// 	int ll = blockID[kk];
	// 	printf("ll is : %d,\n", ll);
	// 	printf("X[%d] is : %d,\n",ll, X[ll]);
	// 	printf("Y[%d] is : %d,\n",ll, Y[ll]);
	// }

	//set SIZE_X, SIZE_Y
	int SIZE_X = NETS;
	int SIZE_Y = NETS;

	// //for debugging
	// printf("SIZE_X is : %d\n", SIZE_X);

    i = 2;
	cost_function<<<1, 1>>>(i, netOffset_Dev, blockID_Dev, X_Dev, Y_Dev, SIZE_X, SIZE_Y);



	free(netOffset);
	free(blockID);
	free(X);
	free(Y);

	cudaFree(netOffset_Dev);
	cudaFree(blockID_Dev);
	cudaFree(X_Dev);
	cudaFree(Y_Dev);

	return 0; 
}