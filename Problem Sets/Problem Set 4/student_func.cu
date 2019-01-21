//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */


__global__
void histogram(unsigned int* const d_input, unsigned int* const d_hist, 
                const size_t numBins, const size_t digit_pos, const size_t length)
{
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= length)
        return;

    extern __shared__ unsigned int s_hist[];

    if (threadIdx.x < numBins)
        s_hist[threadIdx.x] = 0;
    __syncthreads();

    unsigned int bin = (d_input[tid] & (1 << digit_pos)) == (1 << digit_pos) ? 1 : 0;

    atomicAdd(&(s_hist[bin]), 1);
    __syncthreads();

    if (threadIdx.x < numBins)
        atomicAdd(&(d_hist[threadIdx.x]), s_hist[threadIdx.x]);
}

__global__
void prefix_sum(unsigned int const * d_input, unsigned int* const d_cdf, unsigned int * const d_offsets,
                const size_t digit_pos, const size_t length)
{
    uint tid = blockIdx.x * 1 * blockDim.x + threadIdx.x;
    uint sm_tid = threadIdx.x;

    if (tid >= length)
        return;


    uint ai = sm_tid;
    //uint bi = sm_tid + blockDim.x;

    extern __shared__ unsigned int predicate[];

    unsigned int value_1 = ((d_input[tid] & (1 << digit_pos)) == (1 << digit_pos)) ? 1 : 0;
    //unsigned int value_2 = ((d_input[tid + blockDim.x] & (1 << digit_pos)) == (1 << digit_pos)) ? 1 : 0;

    predicate[ai] = value_1;
    //predicate[bi] = value_2;
    __syncthreads();

    // Reduce
    int offset = 1;    

    for(int d = (blockDim.x * 1) >> 1; d > 0; d >>= 1)
    {
        __syncthreads();

        if (sm_tid < d)
        {
            int ai = offset*(2*sm_tid+1)-1;
            int bi = offset*(2*sm_tid+2)-1;
            predicate[bi] += predicate[ai];
        }
        offset *= 2; 
    }

    //Down-sweep
    if (sm_tid == 0) { predicate[blockDim.x*1 - 1] = 0; } // clear the last element

    for (int d = 1; d < blockDim.x*1; d *= 2) // traverse down tree & build scan
    {
        offset >>= 1;
        __syncthreads();
        if (sm_tid < d)
        {
            int ai = offset*(2*sm_tid+1)-1;
            int bi = offset*(2*sm_tid+2)-1;

            float t = predicate[ai];
            predicate[ai] = predicate[bi];
            predicate[bi] += t;
        }
    }

    __syncthreads();

    d_cdf[tid] = predicate[ai];
    //d_cdf[tid + blockDim.x] = predicate[bi];// + add;
    
    if (threadIdx.x == blockDim.x - 1)
    {
        unsigned int add = predicate[ai] + value_1;
        d_offsets[blockIdx.x] = add;
    } 
}

//Use constant memory for broadcast acces
__constant__ unsigned int d_offsets_const[216];

__global__
void combine_prefix(unsigned int const * d_offsets, unsigned int* const d_cdf, const size_t length)
{
    uint tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid >= length)
        return;

    if (blockIdx.x > 0)
    {
        unsigned int add = 0;
        for (int i = 0; i < blockIdx.x; i++)
            add += d_offsets_const[i];
        d_cdf[tid] += add;
    }
}

__global__
void move_data(unsigned int const * d_inputVals, unsigned int* const d_inputPos,
               unsigned int* const d_outputVals, unsigned int* const d_outputPos,
               unsigned int * const d_prefix, unsigned int shift_1,
               const size_t digit, const size_t length)
{
    uint tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid >= length)
        return;
    
    uint out_ind = 0;
    if ((d_inputVals[tid] & (1 << digit)) == (1 << digit))
    {
        out_ind = d_prefix[tid] + shift_1;
    }
    else
    {
        out_ind = tid - d_prefix[tid];
    }

    d_outputVals[out_ind] = d_inputVals[tid];
    d_outputPos[out_ind] = d_inputPos[tid];
}

__global__
void calc_predicate(unsigned int const * d_input, unsigned int * const d_predicate, const size_t digit_pos, const size_t length)
{
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= length)
        return;
    
    unsigned int value = ((d_input[tid] & (1 << digit_pos)) == (1 << digit_pos)) ? 1 : 0;

    d_predicate[tid] = value;
}



void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{ 
  //TODO
  //PUT YOUR SORT HERE

  unsigned int my_hist[2] = {0, 0};

  unsigned int *d_hist;
  checkCudaErrors(cudaMalloc(&d_hist, 2 * sizeof(unsigned int)));
  
  unsigned int *d_cdf;
  checkCudaErrors(cudaMalloc(&d_cdf, numElems * sizeof(unsigned int)));

  const dim3 blockSize_hist(256, 1, 1);
  const dim3 gridSize_hist( (numElems + blockSize_hist.x - 1) / blockSize_hist.x, 1, 1);
  
  const dim3 blockSize_prefix_sum(1024, 1, 1);
  const dim3 gridSize_prefix_sum(ceil(numElems / (blockSize_prefix_sum.x * 1)) + 1, 1, 1);

  const dim3 blockSize_move(512, 1, 1);
  const dim3 gridSize_move( (numElems + blockSize_move.x - 1) / blockSize_move.x, 1, 1);

  unsigned int * d_offsets;
  checkCudaErrors(cudaMalloc(&d_offsets, gridSize_prefix_sum.x * sizeof(unsigned int)));
  checkCudaErrors(cudaMemset(d_offsets, 0, gridSize_prefix_sum.x * sizeof(unsigned int)));

  thrust::device_vector<unsigned int> d_predicate(numElems);

  for(int digit = 0; digit < 8 * sizeof(unsigned int); digit++)
  {
      checkCudaErrors(cudaMemset(d_hist, 0, 2 * sizeof(unsigned int)));

      histogram<<<gridSize_hist, blockSize_hist, 2 * sizeof(unsigned int)>>>(d_inputVals, d_hist, 2, digit, numElems);
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

      calc_predicate<<<gridSize_prefix_sum, blockSize_prefix_sum>>>(d_inputVals, thrust::raw_pointer_cast(&d_predicate[0]),
                                                                   digit, numElems);
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

      thrust::exclusive_scan(d_predicate.begin(), d_predicate.end(), d_predicate.begin());
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

      /*prefix_sum<<<gridSize_prefix_sum, blockSize_prefix_sum, 1 * blockSize_prefix_sum.x * sizeof(unsigned int)>>>(d_inputVals, d_cdf, d_offsets,
                                                                                              digit, numElems);
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

      cudaMemcpyToSymbol(d_offsets_const, d_offsets, gridSize_prefix_sum.x * sizeof(unsigned int), 0, cudaMemcpyDeviceToDevice);
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

      combine_prefix<<<gridSize_prefix_sum.x, blockSize_prefix_sum.x>>>(d_offsets, d_cdf, numElems);
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());*/

      checkCudaErrors(cudaMemcpy(my_hist, d_hist, 2 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
      move_data<<<gridSize_move, blockSize_move>>>(d_inputVals, d_inputPos, d_outputVals, d_outputPos,
                                                  thrust::raw_pointer_cast(&d_predicate[0])/*d_cdf*/, my_hist[0], digit, numElems);
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

      checkCudaErrors(cudaMemcpy(d_inputVals, d_outputVals, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
      checkCudaErrors(cudaMemcpy(d_inputPos, d_outputPos, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
      cudaDeviceSynchronize();
  }  

  
  checkCudaErrors(cudaFree(d_hist));
  checkCudaErrors(cudaFree(d_cdf));
  checkCudaErrors(cudaFree(d_offsets));
}
