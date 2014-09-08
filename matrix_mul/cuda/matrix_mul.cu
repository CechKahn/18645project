/*
    Copyright (C) 2011  Abhinav Jauhri (abhinav.jauhri@gmail.com), Carnegie Mellon University - Silicon Valley 

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
/* Modified by Cheng Zhang and Zhe Qian
 * Last Modification: Feb 25th
 * Modified in: matrix_mul_kernel function and
 */
#include <cuda.h>
#include <cuda_runtime.h>
#include "matrix_mul.h"
#define TILE_WIDTH 2
#define BLOCK_SIZE 32
namespace cuda
{
  /********************* Modified by adding shared memory ****************/
  __global__
  void 
  matrix_mul_kernel(float *sq_matrix_1, float *sq_matrix_2, float *sq_matrix_result, int sq_dimension)
  {
    /* two-dimensional way to get row and col in matrix */
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    /* assgin two 32*32 array in shared memory */
     __shared__ float A[32][32];
     __shared__ float B[32][32];

    float sum = 0.0f;
    /* divide into 32 * 32 parts and do matrix multiplication in each part */
    for (int k = 0; k < (sq_dimension + BLOCK_SIZE - 1) / BLOCK_SIZE; k ++){ 
        /* copy sq_matrix_1 to shared memory A */
        if (k * BLOCK_SIZE + threadIdx.x < sq_dimension && row < sq_dimension)
            A[threadIdx.y][threadIdx.x] = sq_matrix_1[row * sq_dimension + (k * BLOCK_SIZE + threadIdx.x)];
        else
            A[threadIdx.y][threadIdx.x] = 0.0;
        /* copy sq_matrix_1 to shared memory B */
         if (k * BLOCK_SIZE + threadIdx.y < sq_dimension && col < sq_dimension)
            B[threadIdx.y][threadIdx.x] = sq_matrix_2[(k * BLOCK_SIZE + threadIdx.y) * sq_dimension + col];
         else                                                   
            B[threadIdx.y][threadIdx.x] = 0.0;

         __syncthreads();
        /* Do martix multiplication */
         for (int n = 0; n < BLOCK_SIZE; n++)
            sum += A[threadIdx.y][n] * B[n][threadIdx.x];

         __syncthreads();
    
    }
    if (row < sq_dimension && col < sq_dimension)
        sq_matrix_result[row * sq_dimension + col] = sum;
    
  }
/****************************************************************/
  void 
  matrix_multiplication(float *sq_matrix_1, float *sq_matrix_2, float *sq_matrix_result, unsigned int sq_dimension)
  {
    int size = sq_dimension * sq_dimension * sizeof(float);
    float *sq_matrix_1_d, *sq_matrix_2_d, *sq_matrix_result_d;
    
    /***************************************************
  1st Part: Allocation of memory on device memory  
    ****************************************************/
    
    /* copy sq_matrix_1 and sq_matrix_2 to device memory */
    cudaMalloc((void**) &sq_matrix_1_d, size);
    cudaMemcpy(sq_matrix_1_d, sq_matrix_1, size, cudaMemcpyHostToDevice);
    cudaMalloc((void**) &sq_matrix_2_d, size);
    cudaMemcpy(sq_matrix_2_d, sq_matrix_2, size, cudaMemcpyHostToDevice);
    
    /*allocate sq_matrix_result on host */
    cudaMalloc((void**) &sq_matrix_result_d, size);
    
    /***************************************************
   2nd Part: Inovke kernel 
    ****************************************************/

/********* Modified ***********************************************/
/* Using two dimensional grids and blocks instead of only 1 grid */
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((sq_dimension + dimBlock.x - 1) / dimBlock.x, (sq_dimension + dimBlock.y - 1) / dimBlock.y);
    matrix_mul_kernel<<<dimGrid, dimBlock>>>(sq_matrix_1_d, sq_matrix_2_d, sq_matrix_result_d, sq_dimension);
/******************************************************************/
    
    /***************************************************
   3rd Part: Transfer result from device to host 
    ****************************************************/
    cudaMemcpy(sq_matrix_result, sq_matrix_result_d, size, cudaMemcpyDeviceToHost);
    cudaFree(sq_matrix_1_d);
    cudaFree(sq_matrix_2_d);
    cudaFree(sq_matrix_result_d);
  }  
} // namespace cuda
