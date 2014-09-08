/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*   File:         cuda_kmeans.cu  (CUDA version)                            */
/*   Description:  Implementation of simple k-means clustering algorithm     */
/*                 This program takes an array of N data objects, each with  */
/*                 M coordinates and performs a k-means clustering given a   */
/*                 user-provided value of the number of clusters (K). The    */
/*                 clustering results are saved in 2 arrays:                 */
/*                 1. a returned array of size [K][N] indicating the center  */
/*                    coordinates of K clusters                              */
/*                 2. membership[N] stores the cluster center ids, each      */
/*                    corresponding to the cluster a data object is assigned */
/*                                                                           */
/*   Author:  Wei-keng Liao                                                  */
/*            ECE Department, Northwestern University                        */
/*            email: wkliao@ece.northwestern.edu                             */
/*   Copyright, 2005, Wei-keng Liao                                          */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

// Copyright (c) 2005 Wei-keng Liao
// Copyright (c) 2011 Serban Giuroiu
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// -----------------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include "kmeans.h"

static inline int nextPowerOfTwo(int n) {
    n--;

    n = n >>  1 | n;
    n = n >>  2 | n;
    n = n >>  4 | n;
    n = n >>  8 | n;
    n = n >> 16 | n;
//  n = n >> 32 | n;    //  For 64-bit ints

    return ++n;
}

/*----< euclid_dist_2() >----------------------------------------------------*/
/* square of Euclid distance between two multi-dimensional points            */
__host__ __device__ inline static
float euclid_dist_2(int    numCoords,
                    int    numObjs,
                    int    numClusters,
                    float *objects,     // [numCoords][numObjs]
                    float *clusters,    // [numCoords][numClusters]
                    int    objectId,
                    int    clusterId)
{
    int i;
    float ans=0.0;

    for (i = 0; i < numCoords; i++) {
        ans += (objects[numObjs * i + objectId] - clusters[numClusters * i + clusterId]) *
               (objects[numObjs * i + objectId] - clusters[numClusters * i + clusterId]);
    }

    return(ans);
}

/* reduce function to calculate newClusterSize */
__global__ static
void update_new_cluster_size(int *sizeIntermediates,     //[numClusters][numBlocks]
                        int *deviceMembership,
                        int numObjs, 
                        int numClusters,
                        int numberBlocks)
{
    __shared__ int a[3][1024];
    int a0 = 0;
    int a1 = 0;
    int a2 = 0;

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    if (gid < numObjs){
        if (deviceMembership[gid] == 0)
            a0 ++;
        if (deviceMembership[gid] == 1)
            a1 ++;
        if(deviceMembership[gid] == 2)
            a2 ++;
        a[0][tid] = a0;
        a[1][tid] = a1;
        a[2][tid] = a2;
    }else{
        a[0][tid] = 0;
        a[1][tid] = 0;
        a[2][tid] = 0;
    }
     __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            a[0][tid] += a[0][tid + s];
            a[1][tid] += a[1][tid + s];
            a[2][tid] += a[2][tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        for (int i = 0; i < numClusters; i++)
            sizeIntermediates[i * numberBlocks + blockIdx.x] = a[i][0];
    }

}

// just map algorithm
__global__ static
void update_dim(float* deviceNewClusters,
                int* deviceNewClusterSize,
                float* deviceClusters,
                int numClusters,
                int numCoords)
{
    
    deviceClusters[threadIdx.x * numClusters + blockIdx.x] = deviceNewClusters[threadIdx.x * numClusters + blockIdx.x] / deviceNewClusterSize[blockIdx.x]; 
    deviceNewClusters[threadIdx.x * numClusters + blockIdx.x] = 0.0; 
    deviceNewClusterSize[blockIdx.x] = 0;
    
}

/* reduce fuction to update cluster's coords */
__global__ static
void update_cluster(float *intermediates,     //[numCoords][numClusters][numBlocks]
                        int *deviceMembership,
                        float *objects,
                        int numObjs, 
                        int numClusters,
                        int numCoords,
                        int numberBlocks)
{
    __shared__ float a[3][1024];

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    int i, s, j;
    float a0, a1, a2;


    for (i = 0; i < numCoords; i ++){

        if (gid < numObjs){
            if (deviceMembership[gid] == 0)
                a0 = objects[i * numObjs + gid];
            if (deviceMembership[gid] == 1)
                a1 = objects[i * numObjs + gid];
            if(deviceMembership[gid] == 2)
                a2 = objects[i * numObjs + gid];
            a[0][tid] = a0;
            a[1][tid] = a1;
            a[2][tid] = a2;

        }else{

            a[0][tid] = 0;
            a[1][tid] = 0;
            a[2][tid] = 0;
        }
        __syncthreads();

        for (s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                a[0][tid] += a[0][tid + s];
                a[1][tid] += a[1][tid + s];
                a[2][tid] += a[2][tid + s];
            }
            __syncthreads();
        }
    
        if (tid == 0) {
            for (j = 0; j < numClusters; j++)
                intermediates[i * numberBlocks * numClusters + j * numberBlocks + blockIdx.x] = a[j][0];
        }
    }
}

/*----< find_nearest_cluster() >---------------------------------------------*/
__global__ static
void find_nearest_cluster(int numCoords,
                          int numObjs,
                          int numClusters,
                          float *objects,           //  [numCoords][numObjs]
                          float *deviceClusters,    //  [numCoords][numClusters]
                          int *membership,          //  [numObjs]
                          int *intermediates)
{
    extern __shared__ char sharedMemory[];

    //  The type chosen for membershipChanged must be large enough to support
    //  reductions! There are blockDim.x elements, one for each thread in the
    //  block.
    unsigned char *membershipChanged = (unsigned char *)sharedMemory;
    float *clusters = (float *)(sharedMemory + blockDim.x);

    membershipChanged[threadIdx.x] = 0;

    //  BEWARE: We can overrun our shared memory here if there are too many
    //  clusters or too many coordinates!
    for (int i = threadIdx.x; i < numClusters; i += blockDim.x) {
        for (int j = 0; j < numCoords; j++) {
            clusters[numClusters * j + i] = deviceClusters[numClusters * j + i];
        }
    }
    __syncthreads();

    int objectId = blockDim.x * blockIdx.x + threadIdx.x;

    if (objectId < numObjs) {
        int   index, i;
        float dist, min_dist;

        /* find the cluster id that has min distance to object */
        index    = 0;
        min_dist = euclid_dist_2(numCoords, numObjs, numClusters,
                                 objects, clusters, objectId, 0);

        for (i=1; i<numClusters; i++) {
            dist = euclid_dist_2(numCoords, numObjs, numClusters,
                                 objects, clusters, objectId, i);
            /* no need square root */
            if (dist < min_dist) { /* find the min and its array index */
                min_dist = dist;
                index    = i;
            }
        }

        if (membership[objectId] != index) {
            membershipChanged[threadIdx.x] = 1;
        }

        /* assign the membership to object objectId */
        membership[objectId] = index;

        __syncthreads();    //  For membershipChanged[]

        //  blockDim.x *must* be a power of two!
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                membershipChanged[threadIdx.x] +=
                    membershipChanged[threadIdx.x + s];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            intermediates[blockIdx.x] = membershipChanged[0];
        }
    }
}

__global__ static
void compute_delta(int *deviceIntermediates,
                   int numIntermediates,    //  The actual number of intermediates
                   int numIntermediates2)   //  The next power of two
{
    //  The number of elements in this array should be equal to
    //  numIntermediates2, the number of threads launched. It *must* be a power
    //  of two!
    extern __shared__ unsigned int intermediates[];
    //  Copy global intermediate values into shared memory.
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
     intermediates[tid] = (gid < numIntermediates) ? deviceIntermediates[gid] : 0;

    __syncthreads();

    //  numIntermediates2 *must* be a power of two!
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            intermediates[tid] += intermediates[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32){
        volatile unsigned int * temp = intermediates;
        if (blockDim.x >= 64)
            temp[tid] += temp[tid + 32];
        if (blockDim.x >= 32)
            temp[tid] += temp[tid + 16];
        if (blockDim.x >= 16)
            temp[tid] += temp[tid + 8];
        if (blockDim.x >= 8)
            temp[tid] += temp[tid + 4];
        if (blockDim.x >= 4)
            temp[tid] += temp[tid + 2];
        if (blockDim.x >= 2)
            temp[tid] += temp[tid + 1];
    }
    if (tid == 0) {
        deviceIntermediates[blockIdx.x] = intermediates[0];
    }
}
/*----< cuda_kmeans() >-------------------------------------------------------*/
//
//  ----------------------------------------
//  DATA LAYOUT
//
//  objects         [numObjs][numCoords]
//  clusters        [numClusters][numCoords]
//  dimObjects      [numCoords][numObjs]
//  dimClusters     [numCoords][numClusters]
//  newClusters     [numCoords][numClusters]
//  deviceObjects   [numCoords][numObjs]
//  deviceClusters  [numCoords][numClusters]
//  ----------------------------------------
//
/* return an array of cluster centers of size [numClusters][numCoords]       */
float** cuda_kmeans(float **objects,      /* in: [numObjs][numCoords] */
                   int     numCoords,    /* no. features */
                   int     numObjs,      /* no. objects */
                   int     numClusters,  /* no. clusters */
                   float   threshold,    /* % objects change membership */
                   int    *membership,   /* out: [numObjs] */
                   int    *loop_iterations)
{
    int      i, j, loop=0;
    int     *newClusterSize; /* [numClusters]: no. objects assigned in each
                                new cluster */
    float    delta;          /* % of objects change their clusters */
    float  **dimObjects;
    float  **clusters;       /* out: [numClusters][numCoords] */
    float  **dimClusters;
    float  **newClusters;    /* [numCoords][numClusters] */

    float *deviceObjects;
    float *deviceClusters;
    int *deviceMembership;
    int *deviceIntermediates;

    //  Copy objects given in [numObjs][numCoords] layout to new
    //  [numCoords][numObjs] layout
    malloc2D(dimObjects, numCoords, numObjs, float);
    for (i = 0; i < numCoords; i++) {
        for (j = 0; j < numObjs; j++) {
            dimObjects[i][j] = objects[j][i];
        }
    }

    /* pick first numClusters elements of objects[] as initial cluster centers*/
    malloc2D(dimClusters, numCoords, numClusters, float);
    for (i = 0; i < numCoords; i++) {
        for (j = 0; j < numClusters; j++) {
            dimClusters[i][j] = dimObjects[i][j];
        }
    }

    /* initialize membership[] */
    for (i=0; i<numObjs; i++) membership[i] = -1;

    /* need to initialize newClusterSize and newClusters[0] to all 0 */
    newClusterSize = (int*) calloc(numClusters, sizeof(int));
    assert(newClusterSize != NULL);

    malloc2D(newClusters, numCoords, numClusters, float);
    memset(newClusters[0], 0, numCoords * numClusters * sizeof(float));

    //  To support reduction, numThreadsPerClusterBlock *must* be a power of
    //  two, and it *must* be no larger than the number of bits that will
    //  fit into an unsigned char, the type used to keep track of membership
    //  changes in the kernel.
    const unsigned int numThreadsPerClusterBlock = 128;
    const unsigned int numClusterBlocks =
        (numObjs + numThreadsPerClusterBlock - 1) / numThreadsPerClusterBlock;
    const unsigned int clusterBlockSharedDataSize =
        numThreadsPerClusterBlock * sizeof(unsigned char) +
        numClusters * numCoords * sizeof(float);

    /************** Change adjust the block size and add numGrid in case numClusterBlocks > 1 **************/
    const unsigned int numReductionThreads =
        nextPowerOfTwo(numClusterBlocks) > 1024 ? 1024 : nextPowerOfTwo(numClusterBlocks);
    const unsigned int reductionBlockSharedDataSize =
        numReductionThreads * sizeof(unsigned int);
    const unsigned int numGrid =  nextPowerOfTwo(numClusterBlocks) / numReductionThreads;
    /*****************************************************************************************************/
    checkCuda(cudaMalloc(&deviceObjects, numObjs*numCoords*sizeof(float)));
    checkCuda(cudaMalloc(&deviceClusters, numClusters*numCoords*sizeof(float)));
    checkCuda(cudaMalloc(&deviceMembership, numObjs*sizeof(int)));
    checkCuda(cudaMalloc(&deviceIntermediates,  nextPowerOfTwo(numClusterBlocks)*sizeof(unsigned int)));  //changed

    checkCuda(cudaMemcpy(deviceObjects, dimObjects[0],
              numObjs*numCoords*sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(deviceMembership, membership,
              numObjs*sizeof(int), cudaMemcpyHostToDevice));
    /***************Varibale for using CUDA to update new clusters ****************************************/
      
      const int numberBlocks = (numObjs + 1023) / 1024;
        // ------------------------- Array for updae cluster size -----------------------
        int * sizeIntermediates;
        checkCuda(cudaMalloc(&sizeIntermediates, numClusters * numberBlocks *sizeof(int)));
        int temp[numClusters * numberBlocks];
        // ------------------------- Array for update new clusters-----------------------
        float *deviceCluster;
        checkCuda(cudaMalloc(&deviceCluster, numCoords * numClusters * numberBlocks *sizeof(float)));
        float temp2[numCoords * numClusters * numberBlocks];
        // ------------------------- device array for update cluster dim -----------------------
        float *deviceNewClusters;
        int *deviceNewClusterSize;
        // ------------ malloc deviceNewClusters deviceNewClusterSize & deviceDimClusters ---------
        checkCuda(cudaMalloc(&deviceNewClusters, numCoords * numClusters * sizeof(float)));
        checkCuda(cudaMalloc(&deviceNewClusterSize, numClusters * sizeof(int)));
    /*****************************************************************************************************/
    do {
        checkCuda(cudaMemcpy(deviceClusters, dimClusters[0],
                  numClusters*numCoords*sizeof(float), cudaMemcpyHostToDevice));

        find_nearest_cluster
            <<< numClusterBlocks, numThreadsPerClusterBlock, clusterBlockSharedDataSize >>>
            (numCoords, numObjs, numClusters,
             deviceObjects, deviceClusters, deviceMembership, deviceIntermediates);

        cudaThreadSynchronize(); checkLastCudaError();
        
        /**************** Modify the parameter for compute_delta to make it runnable **************/
        compute_delta <<< numGrid, numReductionThreads, reductionBlockSharedDataSize >>>
            (deviceIntermediates, numClusterBlocks, numReductionThreads);
        
        cudaThreadSynchronize(); checkLastCudaError();
        /***************Use loop to calculate delta from deviceIntermeidates(the number of blocks is numGrid) *************/
        int d;
        delta = 0;
        for (i = 0; i < numGrid; i ++){
            checkCuda(cudaMemcpy(&d, deviceIntermediates + i,
                  sizeof(int), cudaMemcpyDeviceToHost));
             delta += (float)d;
        }
        /*****************************************************************************************************/
        checkCuda(cudaMemcpy(membership, deviceMembership,
                  numObjs*sizeof(int), cudaMemcpyDeviceToHost));
        
        /*************** Update new cluster size using CUDA *************************************************************************/

        update_new_cluster_size<<< numberBlocks, 1024>>>(sizeIntermediates, deviceMembership, numObjs, numClusters, numberBlocks);
         cudaThreadSynchronize(); checkLastCudaError();

         checkCuda(cudaMemcpy(temp, sizeIntermediates, numClusters * numberBlocks * sizeof(int), cudaMemcpyDeviceToHost));
         for (i = 0; i < numberBlocks; i ++)
            for (j = 0; j < numClusters; j ++)
                newClusterSize[j] += temp[j * numberBlocks + i];

        /*************** Update new cluster using CUDA *************************************************************************/
        update_cluster<<< numberBlocks, 1024>>>(deviceCluster, deviceMembership,deviceObjects,numObjs,numClusters, numCoords, numberBlocks);
        cudaThreadSynchronize(); checkLastCudaError();
        checkCuda(cudaMemcpy(temp2, deviceCluster, numCoords * numClusters * numberBlocks * sizeof(float), cudaMemcpyDeviceToHost));
        for (i = 0; i < numberBlocks; i ++)
            for (j = 0; j < numClusters; j ++)
                for (int k = 0; k < numCoords; k ++)
                    newClusters[k][j] += temp2[k * numberBlocks * numClusters + j * numberBlocks + i];

        //  TODO: Flip the nesting order
        //  TODO: Change layout of newClusters to [numClusters][numCoords]
        /* average the sum and replace old cluster centers with newClusters */
        /*************** Update new cluster using CUDA *************************************************************************/
        checkCuda(cudaMemcpy(deviceNewClusters, newClusters[0], numCoords * numClusters * sizeof(float), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(deviceNewClusterSize, newClusterSize, numClusters * sizeof(int), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(deviceClusters, dimClusters[0], numCoords * numClusters * sizeof(float), cudaMemcpyHostToDevice));

        update_dim <<< numClusters, 64 >>> (deviceNewClusters, deviceNewClusterSize, deviceClusters, numClusters, numCoords);
        checkCuda(cudaMemcpy(newClusters[0], deviceNewClusters, numCoords * numClusters * sizeof(float), cudaMemcpyDeviceToHost));
        checkCuda(cudaMemcpy(newClusterSize, deviceNewClusterSize, numClusters * sizeof(int), cudaMemcpyDeviceToHost));
        checkCuda(cudaMemcpy(dimClusters[0], deviceClusters, numCoords * numClusters * sizeof(float), cudaMemcpyDeviceToHost));

    } while (delta > threshold * numObjs && loop++ < 500);

    *loop_iterations = loop + 1;

    /* allocate a 2D space for returning variable clusters[] (coordinates
       of cluster centers) */
    malloc2D(clusters, numClusters, numCoords, float);
    for (i = 0; i < numClusters; i++) {
        for (j = 0; j < numCoords; j++) {
            clusters[i][j] = dimClusters[j][i];
        }
    }

    checkCuda(cudaFree(deviceObjects));
    checkCuda(cudaFree(deviceClusters));
    checkCuda(cudaFree(deviceMembership));
    checkCuda(cudaFree(deviceIntermediates));

    free(dimObjects[0]);
    free(dimObjects);
    free(dimClusters[0]);
    free(dimClusters);
    free(newClusters[0]);
    free(newClusters);
    free(newClusterSize);

    return clusters;
}

