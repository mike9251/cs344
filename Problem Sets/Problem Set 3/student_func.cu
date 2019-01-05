/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"

#define MAX(a, b) (((a) > (b)) ? (a): (b))
#define MIN(a, b) (((a) < (b)) ? (a): (b))

__global__
void min_reduce(const float* const inputChannel,
                   float* const block_min_value,
                   int numRows, int numCols)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint idy = blockIdx.y * blockDim.y + threadIdx.y;

    uint pos_1d = idx + idy * numCols;

    uint pos_1d_block = threadIdx.x + threadIdx.y * blockDim.x;

    extern __shared__ float sdata[];

    sdata[pos_1d_block] = inputChannel[pos_1d];
    __syncthreads();

    for(uint i = (blockDim.x * blockDim.y)/2; i > 0; i>>=1)
    {
        if (pos_1d_block < i)
        {
            sdata[pos_1d_block] = MIN(sdata[pos_1d_block], sdata[pos_1d_block + i]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        block_min_value[blockIdx.x + blockIdx.y * (1 + numCols / blockDim.x)] = sdata[0];
    }

}

__global__
void max_reduce(const float* const inputChannel,
                   float* const block_max_value,
                   int numRows, int numCols)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint idy = blockIdx.y * blockDim.y + threadIdx.y;

    uint pos_1d = idx + idy * numCols;

    uint pos_1d_block = threadIdx.x + threadIdx.y * blockDim.x;

    extern __shared__ float sdata[];

    sdata[pos_1d_block] = inputChannel[pos_1d];
    __syncthreads();
        

    for(uint i = (blockDim.x * blockDim.y)/2; i > 0; i>>=1)
    {
        if (pos_1d_block < i)
        {
            sdata[threadIdx.y * blockDim.x + threadIdx.x] = MAX(sdata[pos_1d_block], sdata[pos_1d_block + i]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        block_max_value[blockIdx.x + blockIdx.y * (1 + numCols / blockDim.x)] = sdata[0];
    }

}


__global__
void hist(const float* const d_logLuminance,
                   unsigned int * d_hist,
                   int numBins, float lumMin, float lumRange,
                   int numRows, int numCols)
{

    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint idy = blockIdx.y * blockDim.y + threadIdx.y;

    uint pos_1d = idy * numCols + idx;

    if (idx >= numCols || idy >= numRows)
      return;

    uint bin = MIN(numBins - 1, static_cast<int>((d_logLuminance[pos_1d] - lumMin) / lumRange * numBins));

    atomicAdd(&d_hist[bin], 1);
}

__global__
void prefix_sum(unsigned int* const d_hist, unsigned int* const d_cdf, const size_t numBins)
{
    uint tid = threadIdx.x;

    uint ai = tid;
    uint bi = tid + (numBins / 2);

    __shared__ unsigned int temp[1024];

    temp[ai] = d_hist[ai];
    temp[bi] = d_hist[bi];
    __syncthreads();

    // Reduce
    int offset = 1;

    for(int d = numBins >> 1; d > 0; d >>= 1)
    {
        __syncthreads();

        if (tid < d)
        {
            int ai = offset*(2*tid+1)-1;
            int bi = offset*(2*tid+2)-1;
            temp[bi] += temp[ai];
        }
        offset *= 2; 
    }

    //Down-sweep
    if (tid == 0) { temp[numBins - 1] = 0; } // clear the last element

    for (int d = 1; d < numBins; d *= 2) // traverse down tree & build scan
    {
        offset >>= 1;
        __syncthreads();
        if (tid < d)
        {
            int ai = offset*(2*tid+1)-1;
            int bi = offset*(2*tid+2)-1;

            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }

    __syncthreads();

    d_cdf[ai] = temp[ai]; // write results to device memory
    d_cdf[bi] = temp[bi];    
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */
  // 1), 2)
  const dim3 blockSize(32, 32, 1);
  const dim3 gridSize( (numCols + blockSize.x - 1) / blockSize.x, 
                       (numRows + blockSize.y - 1) / blockSize.y, 1);
  
  float *d_block_min_value;
  checkCudaErrors(cudaMalloc(&d_block_min_value, gridSize.x * gridSize.y * sizeof(float)));

  float *d_block_max_value;
  checkCudaErrors(cudaMalloc(&d_block_max_value, gridSize.x * gridSize.y * sizeof(float)));

  min_reduce<<<gridSize, blockSize, blockSize.x * blockSize.y * sizeof(float)>>>(d_logLuminance, d_block_min_value, numRows, numCols);
  max_reduce<<<gridSize, blockSize, blockSize.x * blockSize.y * sizeof(float)>>>(d_logLuminance, d_block_max_value, numRows, numCols);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  float *d_min_value;
  checkCudaErrors(cudaMalloc(&d_min_value, sizeof(float)));
  float *d_max_value;
  checkCudaErrors(cudaMalloc(&d_max_value, sizeof(float)));
  // Launch 1 block with gridSize.x * gridSize.y * gridSize.z threads from the previous step
  min_reduce<<<1, gridSize, gridSize.x * gridSize.y * sizeof(float)>>>(d_block_min_value, d_min_value, gridSize.y, gridSize.x);
  max_reduce<<<1, gridSize, gridSize.x * gridSize.y * sizeof(float)>>>(d_block_max_value, d_max_value, gridSize.y, gridSize.x);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  checkCudaErrors(cudaMemcpy(&min_logLum,   d_min_value,   sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(&max_logLum,   d_max_value,   sizeof(float), cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // 3) Allocate mem for a histogram on the device and initialize it with 0.
  unsigned int *d_hist;
  unsigned int *d_hist_init = new unsigned int[numBins];
  for (int i = 0; i < numBins; i++)
  {
      d_hist_init[i] = 0;
  }
  checkCudaErrors(cudaMalloc(&d_hist, numBins * sizeof(unsigned int)));
  checkCudaErrors(cudaMemcpy(d_hist,   d_hist_init,   numBins * sizeof(unsigned int), cudaMemcpyHostToDevice));
  // Prepare and launch the hist kernel
  const dim3 blockSize_hist(32, 32, 1);
  const dim3 gridSize_hist( (numCols + blockSize_hist.x - 1) / blockSize_hist.x, 
                       (numRows + blockSize_hist.y - 1) / blockSize_hist.y, 1);

  hist<<<gridSize_hist, blockSize_hist>>>(d_logLuminance, d_hist, numBins, min_logLum, (max_logLum - min_logLum), numRows, numCols);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  
  //4) Perform exclusive scan. Launch numBins / 2 threads as each thread reads 2 elements
  const dim3 blockSize_prefix_sum(512, 1, 1);
  const dim3 gridSize_prefix_sum(1, 1, 1);
  prefix_sum<<<gridSize_prefix_sum, blockSize_prefix_sum>>>(d_hist, d_cdf, numBins);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());


  checkCudaErrors(cudaFree(d_hist));
  checkCudaErrors(cudaFree(d_block_min_value));
  checkCudaErrors(cudaFree(d_block_max_value));
  checkCudaErrors(cudaFree(d_min_value));
  checkCudaErrors(cudaFree(d_max_value));
}