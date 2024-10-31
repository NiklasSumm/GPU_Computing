/*
 *
 * nullKernelAsync.cu
 *
 * Microbenchmark for throughput of asynchronous kernel launch.
 *
 * Build with: nvcc -I ../chLib <options> nullKernelAsync.cu
 * Requires: No minimum SM requirement.
 *
 * Copyright (c) 2011-2012, Archaea Software, LLC.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions 
 * are met: 
 *
 * 1. Redistributions of source code must retain the above copyright 
 *    notice, this list of conditions and the following disclaimer. 
 * 2. Redistributions in binary form must reproduce the above copyright 
 *    notice, this list of conditions and the following disclaimer in 
 *    the documentation and/or other materials provided with the 
 *    distribution. 
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS 
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE 
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, 
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT 
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN 
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include <stdio.h>
#include <time.h>

#include "chTimer.h"

__global__
void
NullKernel()
{
    clock_t start = clock();

    int sum = 0;
    for (int i = 0; i < 10000; i++){
        sum += i;
    }

    clock_t end = clock();
    printf("Busy wait took %.2f cycles", (double) end - start);
}

int
main()
{
    cudaError_t err = cudaSuccess;

    const int cIterations = 1000000;
    printf( "Measuring asynchronous launch time... " ); fflush( stdout );

    chTimerTimestamp start, stop;

    chTimerGetTime( &start );
    for ( int i = 0; i < cIterations; i++ ) {
        NullKernel<<<1,1>>>();
    }
    cudaDeviceSynchronize();
    chTimerGetTime( &stop );

    {
        double microseconds = 1e6*chTimerElapsedTime( &start, &stop );
        double usPerLaunch = microseconds / (float) cIterations;

        printf( "%.2f us\n", usPerLaunch );
    }


    printf( "Measuring synchronous launch time... " ); fflush( stdout );

    //CUDA_LAUNCH_BLOCKING = 1;

    chTimerGetTime( &start );
    for ( int i = 0; i < cIterations; i++ ) {
        NullKernel<<<1,1>>>();
        cudaDeviceSynchronize();
    }
    chTimerGetTime( &stop );

    {
        double microseconds = 1e6*chTimerElapsedTime( &start, &stop );
        double usPerLaunch = microseconds / (float) cIterations;

        printf( "%.2f us\n", usPerLaunch );
    }


    int blockNums[10] = {1, 2, 4, 8, 16, 64, 256, 1024, 4096, 16384};
    int threadNums[11] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};

    for ( int b = 0; b <  sizeof(blockNums) / sizeof(blockNums[0]); b++ ){
        for ( int t = 0; t < sizeof(threadNums) / sizeof(threadNums[0]); t++ ){
            //CUDA_LAUNCH_BLOCKING = 0;

            const int cIterations = 1000000;
            printf( "Measuring asynchronous launch time... " ); fflush( stdout );

            chTimerTimestamp start, stop;

            chTimerGetTime( &start );
            for ( int i = 0; i < cIterations; i++ ) {
                NullKernel<<<1,1>>>();
            }
            cudaDeviceSynchronize();
            chTimerGetTime( &stop );

            {
                double microseconds = 1e6*chTimerElapsedTime( &start, &stop );
                double usPerLaunch = microseconds / (float) cIterations;

                printf( "%.2f us\n", usPerLaunch );
            }


            printf( "Measuring synchronous launch time... " ); fflush( stdout );

            //CUDA_LAUNCH_BLOCKING = 1;

            chTimerGetTime( &start );
            for ( int i = 0; i < cIterations; i++ ) {
                NullKernel<<<1,1>>>();
                cudaDeviceSynchronize();
            }
            chTimerGetTime( &stop );

            {
                double microseconds = 1e6*chTimerElapsedTime( &start, &stop );
                double usPerLaunch = microseconds / (float) cIterations;

                printf( "%.2f us\n", usPerLaunch );
            }
        }
    }


    int numElements = 50000;
    size_t size = numElements * sizeof(int);

    int *h_Data = (int *)malloc(size);

    for (int i = 0; i < numElements; ++i) {
        h_Data[i] = rand();
    }

    int *d_Data = NULL;
    err = cudaMalloc((void **)&d_Data, size);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n",
            cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_Data, h_Data, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess) {
        fprintf(stderr,
            "Failed to copy vector A from host to device (error code %s)!\n",
            cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(h_Data, d_Data, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess) {
        fprintf(stderr,
            "Failed to copy vector C from device to host (error code %s)!\n",
            cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    return 0;
}
