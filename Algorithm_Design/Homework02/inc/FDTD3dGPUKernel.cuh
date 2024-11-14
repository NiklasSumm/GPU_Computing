/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "FDTD3dGPU.h"
#include <cooperative_groups.h>

/// Helper macros for stringification
//#define TO_STRING_HELPER(X)   #X
//#define TO_STRING(X)          TO_STRING_HELPER(X)
//
//// Define loop unrolling depending on the compiler
//#if defined(__ICC) || defined(__ICL)
//  #define UNROLL_LOOP(n)      _Pragma(TO_STRING(unroll (n)))
//#elif defined(__clang__)
//  #define UNROLL_LOOP(n)      _Pragma(TO_STRING(unroll (n)))
//#elif defined(__GNUC__) && !defined(__clang__)
//  #define UNROLL_LOOP(n)      _Pragma(TO_STRING(GCC unroll (16)))
//#elif defined(_MSC_BUILD)
//  #pragma message ("Microsoft Visual C++ (MSVC) detected: Loop unrolling not supported!")
//  #define UNROLL_LOOP(n)
//#else
//  #warning "Unknown compiler: Loop unrolling not supported!"
//  #define UNROLL_LOOP(n)
//#endif

template<int Begin, int End, int Step = 1>
//lambda unroller
struct UnrollerL {
    template<typename Lambda>
    static void step(Lambda& func) {
        func(Begin);
        UnrollerL<Begin+Step, End, Step>::step(func);
    }
};
//end of lambda unroller
template<int End, int Step>
struct UnrollerL<End, End, Step> {
    template<typename Lambda>
    static void step(Lambda& func) {
    }
};

namespace cg = cooperative_groups;

// Note: If you change the RADIUS, you should also change the unrolling below
//#define RADIUS 4

__constant__ float stencil[10 + 1]; //Size is adjusted to maximum radius, so that is always large enough

template<int Radius>
__global__ void FiniteDifferencesKernel(float *output, const float *input,
                                        const int dimx, const int dimy,
                                        const int dimz) {
  bool validr = true;
  bool validw = true;
  const int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
  const int gtidy = blockIdx.y * blockDim.y + threadIdx.y;
  const int ltidx = threadIdx.x;
  const int ltidy = threadIdx.y;
  const int workx = blockDim.x;
  const int worky = blockDim.y;
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  __shared__ float tile[k_blockDimMaxY + 2 * Radius][k_blockDimX + 2 * Radius];

  const int stride_y = dimx + 2 * Radius;
  const int stride_z = stride_y * (dimy + 2 * Radius);

  int inputIndex = 0;
  int outputIndex = 0;

  // Advance inputIndex to start of inner volume
  inputIndex += Radius * stride_y + Radius;

  // Advance inputIndex to target element
  inputIndex += gtidy * stride_y + gtidx;

  float infront[Radius];
  float behind[Radius];
  float current;

  const int tx = ltidx + Radius;
  const int ty = ltidy + Radius;

  // Check in bounds
  if ((gtidx >= dimx + Radius) || (gtidy >= dimy + Radius)) validr = false;

  if ((gtidx >= dimx) || (gtidy >= dimy)) validw = false;

  // Preload the "infront" and "behind" data
  for (int i = Radius - 2; i >= 0; i--) {
    if (validr) behind[i] = input[inputIndex];

    inputIndex += stride_z;
  }

  if (validr) current = input[inputIndex];

  outputIndex = inputIndex;
  inputIndex += stride_z;

  for (int i = 0; i < Radius; i++) {
    if (validr) infront[i] = input[inputIndex];

    inputIndex += stride_z;
  }

// Step through the xy-planes
//#pragma unroll 9

  for (int iz = 0; iz < dimz; iz++) {
    // Advance the slice (move the thread-front)
    for (int i = Radius - 1; i > 0; i--) behind[i] = behind[i - 1];

    behind[0] = current;
    current = infront[0];

    UnrollerL<0, Radius>::step( [&] (int i){
      infront[i] = infront[i + 1];
    }
    //for (int i = 0; i < Radius - 1; i++) infront[i] = infront[i + 1];

    if (validr) infront[Radius - 1] = input[inputIndex];

    inputIndex += stride_z;
    outputIndex += stride_z;
    cg::sync(cta);

    // Note that for the work items on the boundary of the problem, the
    // supplied index when reading the halo (below) may wrap to the
    // previous/next row or even the previous/next xy-plane. This is
    // acceptable since a) we disable the output write for these work
    // items and b) there is at least one xy-plane before/after the
    // current plane, so the access will be within bounds.

    // Update the data slice in the local tile
    // Halo above & below
    if (ltidy < Radius) {
      tile[ltidy][tx] = input[outputIndex - Radius * stride_y];
      tile[ltidy + worky + Radius][tx] = input[outputIndex + worky * stride_y];
    }

    // Halo left & right
    if (ltidx < Radius) {
      tile[ty][ltidx] = input[outputIndex - Radius];
      tile[ty][ltidx + workx + Radius] = input[outputIndex + workx];
    }

    tile[ty][tx] = current;
    cg::sync(cta);

    // Compute the output value
    float value = stencil[0] * current;

    UnrollerL<0, Radius>::step( [&] (int i)){
      value +=
          stencil[i] * (infront[i - 1] + behind[i - 1] + tile[ty - i][tx] +
                        tile[ty + i][tx] + tile[ty][tx - i] + tile[ty][tx + i]);
    }
    //for (int i = 1; i <= Radius; i++) {
    //  value +=
    //      stencil[i] * (infront[i - 1] + behind[i - 1] + tile[ty - i][tx] +
    //                    tile[ty + i][tx] + tile[ty][tx - i] + tile[ty][tx + i]);
    //}

    // Store the output value
    if (validw) output[outputIndex] = value;
  }
}
