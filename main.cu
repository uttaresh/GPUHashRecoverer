/******************************************************************************
 *cr
 *cr         (C) Copyright 2010-2013 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>

#include "support.h"
#include "kernel.cu"

int main(int argc, char* argv[])
{
    Timer timer;

    // Initialize host variables ----------------------------------------------

    #if TEST_MODE
    printf("\n***Running in test mode***\n"); fflush(stdout);
    #endif

    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);

	float *in_h, *out_h;
	float *in_d, *out_d;
	unsigned num_elements;
	cudaError_t cuda_ret;

	/* Allocate and initialize input vector */
    if(argc == 1) {
        num_elements = 1000000;
    } else if(argc == 2) {
        num_elements = atoi(argv[1]);
    } else {
        printf("\n    Invalid input parameters!"
           "\n    Usage: ./prefix-scan        # Input of size 1,000,000 is used"
           "\n    Usage: ./prefix-scan <m>    # Input of size m is used"
           "\n");
        exit(0);
    }
    initVector(&in_h, num_elements);

	/* Allocate and initialize output vector */
	out_h = (float*)calloc(num_elements, sizeof(float));
	if(out_h == NULL) FATAL("Unable to allocate host");

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("    Input size = %u\n", num_elements);

    // Allocate device variables ----------------------------------------------

    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);

	cuda_ret = cudaMalloc((void**)&in_d, num_elements*sizeof(float));
	if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
	cuda_ret = cudaMalloc((void**)&out_d, num_elements*sizeof(float));
	if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy host variables to device ------------------------------------------

    printf("Copying data from host to device..."); fflush(stdout);
    startTime(&timer);

    cuda_ret = cudaMemcpy(in_d, in_h, num_elements*sizeof(float),
        cudaMemcpyHostToDevice);
	if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");

	cuda_ret = cudaMemset(out_d, 0, num_elements*sizeof(float));
	if(cuda_ret != cudaSuccess) FATAL("Unable to set device memory");

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Launch kernel ----------------------------------------------------------
    printf("Launching kernel..."); fflush(stdout);
    startTime(&timer);

    preScan(out_d, in_d, num_elements);

	cuda_ret = cudaDeviceSynchronize();
	if(cuda_ret != cudaSuccess) FATAL("Unable to launch/execute kernel");

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy device variables from host ----------------------------------------

    printf("Copying data from device to host..."); fflush(stdout);
    startTime(&timer);

    cuda_ret = cudaMemcpy(out_h, out_d, num_elements*sizeof(float),
        cudaMemcpyDeviceToHost);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to host");

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

	printf("Total sum: %.2f\n", out_h[0]);

    #if TEST_MODE
    printf("\nResult:\n");
    for(int i = 0; i < num_elements; ++i) {
        printf("%.2f ", out_h[i]);
    }
    printf("\n");
    #endif

    // Verify correctness -----------------------------------------------------

    printf("Verifying results..."); fflush(stdout);

    verify(in_h, out_h, num_elements);

    // Free memory ------------------------------------------------------------

	cudaFree(in_d); cudaFree(out_d);
	free(in_h); free(out_h);

	return 0;
}

