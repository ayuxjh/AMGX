/* Copyright (c) 2013-2017, NVIDIA CORPORATION. All rights reserved.
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

#include <string.h>
#include <cutil.h>
#include <miscmath.h>
#include <amgx_cusparse.h>
#include <thrust/copy.h>
#include <solvers/multicolor_ilu_solver.h>
#include <solvers/block_common_solver.h>
#include <csr_multiply.h>
#include <gaussian_elimination.h>
#include <basic_types.h>
#include <util.h>
#include <texture.h>
#include <ld_functions.h>
#include <logger.h>
#include <matrix_io.h>
#include <permute.h>
#include <thrust/logical.h>
#include <profile.h>
#include <sm_utils.inl>
#include <algorithm>

// TODO: Have 2 groups of 16 threads collaborate
// TODO: Add support for outside diagonal
// TODO: Add support for unsorted rows

#define EXPERIMENTAL_LU_FACTORS
#define EXPERIMENTAL_LU_FORWARD
#define EXPERIMENTAL_LU_BACKWARD

using namespace std;
namespace amgx
{

namespace multicolor_ilu_solver
{

// -----------
// Kernels
// -----------

#ifdef EXPERIMENTAL_LU_FORWARD

template<typename IndexType, typename ValueTypeA, typename ValueTypeB, int CtaSize, int bsize, bool ROW_MAJOR, bool hasDiag>
__global__
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
__launch_bounds__( CtaSize, 16 )
#elif defined(__CUDA_ARCH__)
__launch_bounds__( CtaSize, 8 )
#endif
void LU_forward_4x4_kernel_warp( const IndexType *LU_row_offsets,
                                 const IndexType *LU_smaller_color_offsets,
                                 const IndexType *LU_column_indices,
                                 const ValueTypeA *LU_nonzero_values,
                                 const IndexType *A_row_offsets,
                                 const IndexType *A_column_indices,
                                 const ValueTypeA *A_nonzero_values,
                                 const IndexType *A_dia_indices,
                                 const ValueTypeB *x,
                                 const ValueTypeB *b,
                                 ValueTypeB *delta,
                                 const int *sorted_rows_by_color,
                                 const int num_rows_per_color,
                                 const int current_color,
                                 bool xIsZero )
{
    const int nHalfWarps = CtaSize / 16; // Number of half warps per Cta
    const int warpId = utils::warp_id();
    const int laneId = utils::lane_id();
    const int halfWarpId = threadIdx.x / 16;
    const int halfLaneId = threadIdx.x % 16;
    const int halfLaneId_div_4 = halfLaneId / 4;
    const int halfLaneId_mod_4 = halfLaneId % 4;
#if __CUDA_ARCH__ < 300
    // Shared memory to broadcast column IDs.
    __shared__ volatile int s_aColIds[CtaSize];
    // Each thread keeps its own pointer.
    volatile int *my_s_aColIds = &s_aColIds[16 * halfWarpId];
#else
    const int upperHalf = 16 * (laneId / 16);
#endif
    // Shared memory needed to exchange X and delta.
    __shared__ volatile ValueTypeB s_mem[CtaSize];
    // Each thread keeps its own pointer to shared memory to avoid some extra computations.
    volatile ValueTypeB *my_s_mem = &s_mem[16 * halfWarpId];

    // Iterate over the rows of the matrix. One warp per row.
    for ( int aRowIt = blockIdx.x * nHalfWarps + halfWarpId ; aRowIt < num_rows_per_color ; aRowIt += gridDim.x * nHalfWarps )
    {
        int aRowId = sorted_rows_by_color[aRowIt];
        // Load one block of B.
        ValueTypeB my_bmAx(0);

        unsigned int active_mask = utils::activemask();

        if ( ROW_MAJOR )
        {
            if ( halfLaneId_mod_4 == 0 )
            {
                my_bmAx = __cachingLoad(&b[4 * aRowId + halfLaneId_div_4]);
            }
        }
        else
        {
            if ( halfLaneId_div_4 == 0 )
            {
                my_bmAx = __cachingLoad(&b[4 * aRowId + halfLaneId_mod_4]);
            }
        }

        // Don't do anything if X is zero.
        if ( !xIsZero )
        {
            int aColBegin = A_row_offsets[aRowId  ];
            int aColEnd   = A_row_offsets[aRowId + 1];
            int aColMax = aColEnd;

            if ( hasDiag )
            {
                ++aColMax;
            }

            // Each warp load column indices of 32 nonzero blocks
            for ( ; utils::any( aColBegin < aColMax, active_mask ) ; aColBegin += 16 )
            {
                int aColIt = aColBegin + halfLaneId;
                // Get the ID of the column.
                int aColId = -1;

                if ( aColIt < aColEnd )
                {
                    aColId = A_column_indices[aColIt];
                }

                if ( hasDiag && aColIt == aColEnd )
                {
                    aColId = aRowId;
                }

#if __CUDA_ARCH__ < 300
                my_s_aColIds[halfLaneId] = aColId;
#endif
                // Count the number of active columns.
                int vote =  utils::ballot(aColId != -1, active_mask);
                // The number of iterations.
                int nCols = max( __popc( vote & 0x0000ffff ), __popc( vote & 0xffff0000 ) );

                // Loop over columns. We compute 8 columns per iteration.
                for ( int k = 0 ; k < nCols ; k += 4 )
                {
                    int my_k = k + halfLaneId_div_4;
                    // Load 8 blocks of X.
#if __CUDA_ARCH__ < 300
                    int waColId = my_s_aColIds[my_k];
#else
                    int waColId = utils::shfl( aColId, upperHalf + my_k, warpSize, active_mask );
#endif
                    ValueTypeB my_x(0);

                    if ( waColId != -1 )
                    {
                        my_x = __cachingLoad(&x[4 * waColId + halfLaneId_mod_4]);
                    }

                    my_s_mem[halfLaneId] = my_x;
                    // Load 8 blocks of A.
#pragma unroll
                    for ( int i = 0 ; i < 4 ; ++i )
                    {
                        int w_aColTmp = aColBegin + k + i, w_aColIt = -1;

                        if ( w_aColTmp < aColEnd )
                        {
                            w_aColIt = w_aColTmp;
                        }

                        if ( hasDiag && w_aColTmp == aColEnd )
                        {
                            w_aColIt = A_dia_indices[aRowId];
                        }

                        ValueTypeA my_val(0);

                        if ( w_aColIt != -1 )
                        {
                            my_val = A_nonzero_values[16 * w_aColIt + halfLaneId];
                        }

                        if ( ROW_MAJOR )
                        {
                            my_bmAx -= my_val * my_s_mem[4 * i + halfLaneId_mod_4];
                        }
                        else
                        {
                            my_bmAx -= my_val * my_s_mem[4 * i + halfLaneId_div_4];
                        }
                    }
                } // Loop over k
            } // Loop over aColIt
        } // if xIsZero

        // Contribution from each nonzero column that has color less than yours
        if ( current_color != 0 )
        {
            // TODO: Use constant or texture here
            int aColBegin = LU_row_offsets[aRowId];
            int aColEnd   = LU_smaller_color_offsets[aRowId];

            // Each warp load column indices of 32 nonzero blocks
            for ( ; utils::any( aColBegin < aColEnd, active_mask ) ; aColBegin += 16 )
            {
                int aColIt = aColBegin + halfLaneId;
                int aColId = -1;

                if ( aColIt < aColEnd )
                {
                    aColId = LU_column_indices[aColIt];
                }

#if __CUDA_ARCH__ < 300
                my_s_aColIds[halfLaneId] = aColId;
#endif

                // Count the number of active columns.
                int vote =  utils::ballot(aColId != -1, active_mask);
                // The number of iterations.
                int nCols = max( __popc( vote & 0x0000ffff ), __popc( vote & 0xffff0000 ) );

                for ( int k = 0 ; k < nCols ; k += 4 )
                {
                    int my_k = k + halfLaneId_div_4;
                    // Load 8 blocks of X.
#if __CUDA_ARCH__ < 300
                    int waColId = my_s_aColIds[my_k];
#else
                    int waColId = utils::shfl( aColId, upperHalf + my_k, warpSize, active_mask );
#endif
                    ValueTypeB my_delta(0);

                    if ( waColId != -1 )
                    {
                        my_delta = delta[4 * waColId + halfLaneId_mod_4];
                    }

                    my_s_mem[halfLaneId] = my_delta;
                    utils::syncwarp(); // making sure smem write propagated
                    // Update b-Ax.
#pragma unroll
                    for ( int i = 0 ; i < 4 ; ++i )
                    {
                        int w_aColTmp = aColBegin + k + i, w_aColIt = -1;

                        if ( w_aColTmp < aColEnd )
                        {
                            w_aColIt = w_aColTmp;
                        }

                        ValueTypeA my_val(0);

                        if ( w_aColIt != -1 )
                        {
                            my_val = LU_nonzero_values[16 * w_aColIt + halfLaneId];
                        }

                        if ( ROW_MAJOR )
                        {
                            my_bmAx -= my_val * my_s_mem[4 * i + halfLaneId_mod_4];
                        }
                        else
                        {
                            my_bmAx -= my_val * my_s_mem[4 * i + halfLaneId_div_4];
                        }
                    }
                } // Loop over k
            } // Loop over aColIt
        } // If current_color != 0

        // Reduce bmAx terms.
#if __CUDA_ARCH__ >= 300

        if ( ROW_MAJOR )
        {
            my_bmAx += utils::shfl_xor( my_bmAx, 1, warpSize, active_mask );
            my_bmAx += utils::shfl_xor( my_bmAx, 2, warpSize, active_mask );
        }
        else
        {
            my_bmAx += utils::shfl_xor( my_bmAx, 4, warpSize, active_mask );
            my_bmAx += utils::shfl_xor( my_bmAx, 8, warpSize, active_mask );
        }

#else
        s_mem[threadIdx.x] = my_bmAx;

        if ( ROW_MAJOR )
        {
            if ( laneId < 31 ) { s_mem[threadIdx.x] = my_bmAx += s_mem[threadIdx.x + 1]; }

            if ( laneId < 30 ) { s_mem[threadIdx.x] = my_bmAx += s_mem[threadIdx.x + 2]; }
        }
        else
        {
            if ( laneId < 28 ) { s_mem[threadIdx.x] = my_bmAx += s_mem[threadIdx.x + 4]; }

            if ( laneId < 24 ) { s_mem[threadIdx.x] = my_bmAx += s_mem[threadIdx.x + 8]; }
        }

#endif

        // Store the results.
        if ( ROW_MAJOR )
        {
            if ( halfLaneId_mod_4 == 0 )
            {
                delta[4 * aRowId + halfLaneId_div_4] = my_bmAx;
            }
        }
        else
        {
            if ( halfLaneId_div_4 == 0 )
            {
                delta[4 * aRowId + halfLaneId_mod_4] = my_bmAx;
            }
        }
    }
}

#else
template<typename IndexType, typename ValueTypeA, typename ValueTypeB, int blockrows_per_cta, int blockrows_per_warp, int bsize, bool ROW_MAJOR>
__global__
void LU_forward_4x4_kernel(const IndexType *LU_row_offsets, const IndexType *LU_smaller_color_offsets, const IndexType *LU_column_indices, const ValueTypeA *LU_nonzero_values,  const IndexType *A_row_offsets, const IndexType *A_column_indices, const ValueTypeA *A_nonzero_values,
                           const ValueTypeB *x, const ValueTypeB *b,  ValueTypeB *delta, const int *sorted_rows_by_color,
                           const int num_rows_per_color, const int current_color, bool xIsZero)


{
    int warp_id = threadIdx.x / 32;
    int warp_thread_id = threadIdx.x & 31;

    // padding row blocks to fit in a single warp
    if ( warp_thread_id >= blockrows_per_warp * bsize ) { return; }

    // new thread id with padding
    int tid = warp_id * blockrows_per_warp * bsize + warp_thread_id;
    // Here we use one thread per row (not block row)
    int cta_blockrow_id = (tid) / bsize;
    int blockrow_id = blockIdx.x * blockrows_per_cta + cta_blockrow_id;
    const int vec_entry_index = tid - cta_blockrow_id * bsize;
    volatile __shared__ ValueTypeB s_delta_temp[ bsize * blockrows_per_cta];
    int offset, s_offset, i;
    ValueTypeB bmAx, temp[bsize];

    while (blockrow_id < num_rows_per_color &&  cta_blockrow_id < blockrows_per_cta)
    {
        i = sorted_rows_by_color[blockrow_id];
        // Load RHS and x
        offset = i * bsize + vec_entry_index;
        bmAx = b[offset];

        if (!xIsZero)
        {
            int jmin = A_row_offsets[i];
            int jmax = A_row_offsets[i + 1];

            //TODO: Assumes inside diagonal
            for (int jind = jmin; jind < jmax; jind++)
            {
                IndexType jcol = A_column_indices[jind];
                offset = jcol * bsize + vec_entry_index;
                s_delta_temp[tid] = x[offset];

                // Load nonzero_values
                if (ROW_MAJOR)
                {
                    offset = jind * bsize * bsize + vec_entry_index * bsize;
                    loadAsVector<bsize>(A_nonzero_values + offset, temp);
                }
                else
                {
                    offset = jind * bsize * bsize + vec_entry_index;
#pragma unroll
                    for (int m = 0; m < bsize; m++)
                    {
                        temp[m] = A_nonzero_values[offset + bsize * m];
                    }
                }

                // Do matrix multiply
                s_offset = cta_blockrow_id * bsize;
#pragma unroll
                for (int m = 0; m < bsize; m++)
                {
                    bmAx -= temp[m] * s_delta_temp[s_offset++];
                }
            }
        }

        // Contribution from each nonzero column that has color less than yours
        if (current_color != 0)
        {
            int jmin = LU_row_offsets[i];
            int jmax = LU_smaller_color_offsets[i];

            for (int jind = jmin; jind < jmax; jind++)
            {
                IndexType jcol = LU_column_indices[jind];
                offset = jcol * bsize + vec_entry_index;
                s_delta_temp[tid] = ld_cg(delta + offset);

                // Load nonzero_values
                if (ROW_MAJOR)
                {
                    offset = jind * bsize * bsize + vec_entry_index * bsize;
                    loadAsVector<bsize>(LU_nonzero_values + offset, temp);
                }
                else
                {
                    offset = jind * bsize * bsize + vec_entry_index;
#pragma unroll
                    for (int m = 0; m < bsize; m++)
                    {
                        temp[m] = LU_nonzero_values[offset + bsize * m];
                    }
                }

                // Do matrix multiply
                s_offset = cta_blockrow_id * bsize;
#pragma unroll
                for (int m = 0; m < bsize; m++)
                {
                    bmAx -= temp[m] * s_delta_temp[s_offset++];
                }
            }
        }

        delta[i * bsize + vec_entry_index] = bmAx;
        blockrow_id += blockrows_per_cta * gridDim.x;
    }
}
#endif

#ifdef EXPERIMENTAL_LU_FORWARD

template<typename IndexType, typename ValueTypeA, typename ValueTypeB, int CtaSize, int bsize, bool ROW_MAJOR, bool hasDiag>
__global__
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
__launch_bounds__( CtaSize, 16 )
#elif defined(__CUDA_ARCH__)
__launch_bounds__( CtaSize, 8 )
#endif
void LU_forward_3x3_kernel_warp( const IndexType *LU_row_offsets,
                                 const IndexType *LU_smaller_color_offsets,
                                 const IndexType *LU_column_indices,
                                 const ValueTypeA *LU_nonzero_values,
                                 const IndexType *A_row_offsets,
                                 const IndexType *A_column_indices,
                                 const ValueTypeA *A_nonzero_values,
                                 const IndexType *A_dia_indices,
                                 const ValueTypeB *x,
                                 const ValueTypeB *b,
                                 ValueTypeB *delta,
                                 const int *sorted_rows_by_color,
                                 const int num_rows_per_color,
                                 const int current_color,
                                 bool xIsZero )
{
    const int nHalfWarps = CtaSize / 16; // Number of half warps per Cta
    const int warpId = utils::warp_id();
    const int laneId = utils::lane_id();
    const int halfWarpId = threadIdx.x / 16;
    const int halfLaneId = threadIdx.x % 16;
    const int halfLaneId_div_4 = halfLaneId / 4;
    const int halfLaneId_mod_4 = halfLaneId % 4;
#if __CUDA_ARCH__ < 300
    // Shared memory to broadcast column IDs.
    __shared__ volatile int s_aColIds[CtaSize];
    // Each thread keeps its own pointer.
    volatile int *my_s_aColIds = &s_aColIds[16 * halfWarpId];
#else
    const int upperHalf = 16 * (laneId / 16);
#endif
    // Shared memory needed to exchange X and delta.
    __shared__ volatile ValueTypeB s_mem[CtaSize];
    // Each thread keeps its own pointer to shared memory to avoid some extra computations.
    volatile ValueTypeB *my_s_mem = &s_mem[16 * halfWarpId];

    // Iterate over the rows of the matrix. One warp per row.
    for ( int aRowIt = blockIdx.x * nHalfWarps + halfWarpId ; aRowIt < num_rows_per_color ; aRowIt += gridDim.x * nHalfWarps )
    {
        int aRowId = sorted_rows_by_color[aRowIt];
        // Load one block of B.
        ValueTypeB my_bmAx(0);

        unsigned int active_mask = utils::activemask();

        if ( ROW_MAJOR )
        {
            if ( halfLaneId_mod_4 == 0 && halfLaneId_div_4 != 3 )
            {
                my_bmAx = __cachingLoad(&b[3 * aRowId + halfLaneId_div_4]);
            }
        }
        else
        {
            // if ( halfLaneId_div_4 == 0 )
            // {
            //     my_bmAx = __cachingLoad(&b[3 * aRowId + halfLaneId_mod_4]);
            // }
        }

        // Don't do anything if X is zero.
        if ( !xIsZero )
        {
            int aColBegin = A_row_offsets[aRowId  ];
            int aColEnd   = A_row_offsets[aRowId + 1];
            int aColMax = aColEnd;

            if ( hasDiag )
            {
                ++aColMax;
            }

            // Each warp load column indices of 32 nonzero blocks
            for ( ; utils::any( aColBegin < aColMax, active_mask ) ; aColBegin += 16 )
            {
                int aColIt = aColBegin + halfLaneId;
                // Get the ID of the column.
                int aColId = -1;

                if ( aColIt < aColEnd )
                {
                    aColId = A_column_indices[aColIt];
                }

                if ( hasDiag && aColIt == aColEnd )
                {
                    aColId = aRowId;
                }

#if __CUDA_ARCH__ < 300
                my_s_aColIds[halfLaneId] = aColId;
#endif
                // Count the number of active columns.
                int vote =  utils::ballot(aColId != -1, active_mask);
                // The number of iterations.
                int nCols = max( __popc( vote & 0x0000ffff ), __popc( vote & 0xffff0000 ) );

                // Loop over columns. We compute 8 columns per iteration.
                for ( int k = 0 ; k < nCols ; k += 4 )
                {
                    int my_k = k + halfLaneId_div_4;
                    // Load 8 blocks of X.
#if __CUDA_ARCH__ < 300
                    int waColId = my_s_aColIds[my_k];
#else
                    int waColId = utils::shfl( aColId, upperHalf + my_k, warpSize, active_mask );
#endif
                    ValueTypeB my_x(0);

                    if ( waColId != -1  &&  halfLaneId_mod_4 != 3 )
                    {
                        my_x = __cachingLoad(&x[3 * waColId + halfLaneId_mod_4]);
                    }

                    my_s_mem[halfLaneId] = my_x;
                    utils::syncwarp();
                    // Load 8 blocks of A.
#pragma unroll
                    for ( int i = 0 ; i < 4 ; ++i )
                    {
                        int w_aColTmp = aColBegin + k + i, w_aColIt = -1;

                        if ( w_aColTmp < aColEnd )
                        {
                            w_aColIt = w_aColTmp;
                        }

                        if ( hasDiag && w_aColTmp == aColEnd )
                        {
                            w_aColIt = A_dia_indices[aRowId];
                        }

                        ValueTypeA my_val(0);

                        if ( w_aColIt != -1 )
                        {
                            if( halfLaneId_mod_4 != 3 && halfLaneId_div_4 != 3 )
                                my_val = A_nonzero_values[9 * w_aColIt + 3 * halfLaneId_div_4 + halfLaneId_mod_4];
                        }

                        if ( ROW_MAJOR )
                        {
                            if( halfLaneId_mod_4 != 3 && halfLaneId_div_4 != 3 )
                                my_bmAx -= my_val * my_s_mem[4 * i + halfLaneId_mod_4];
                        }
                        else
                        {
                            //FatalError("Haven't implemented LU_forward_3x3 for not COL_MAJOR", AMGX_ERR_NOT_SUPPORTED_TARGET);
                        }
                    }
                } // Loop over k
            } // Loop over aColIt
        } // if xIsZero

        // Contribution from each nonzero column that has color less than yours
        if ( current_color != 0 )
        {
            // TODO: Use constant or texture here
            int aColBegin = LU_row_offsets[aRowId];
            int aColEnd   = LU_smaller_color_offsets[aRowId];

            // Each warp load column indices of 32 nonzero blocks
            for ( ; utils::any( aColBegin < aColEnd, active_mask ) ; aColBegin += 16 )
            {
                int aColIt = aColBegin + halfLaneId;
                int aColId = -1;

                if ( aColIt < aColEnd )
                {
                    aColId = LU_column_indices[aColIt];
                }

#if __CUDA_ARCH__ < 300
                my_s_aColIds[halfLaneId] = aColId;
#endif

                // Count the number of active columns.
                int vote =  utils::ballot(aColId != -1, active_mask);
                // The number of iterations.
                int nCols = max( __popc( vote & 0x0000ffff ), __popc( vote & 0xffff0000 ) );

                for ( int k = 0 ; k < nCols ; k += 4 )
                {
                    int my_k = k + halfLaneId_div_4;
                    // Load 8 blocks of X.
#if __CUDA_ARCH__ < 300
                    int waColId = my_s_aColIds[my_k];
#else
                    int waColId = utils::shfl( aColId, upperHalf + my_k, warpSize, active_mask );
#endif
                    ValueTypeB my_delta(0);

                    if ( waColId != -1 )
                    {
                        if(halfLaneId_mod_4 != 3 )
                            my_delta = delta[3 * waColId + halfLaneId_mod_4];
                    }

                    my_s_mem[halfLaneId] = my_delta;
                    utils::syncwarp(); // making sure smem write propagated
                    // Update b-Ax.
#pragma unroll
                    for ( int i = 0 ; i < 4 ; ++i )
                    {
                        int w_aColTmp = aColBegin + k + i, w_aColIt = -1;

                        if ( w_aColTmp < aColEnd )
                        {
                            w_aColIt = w_aColTmp;
                        }

                        ValueTypeA my_val(0);

                        if ( w_aColIt != -1 )
                        {
                            if( halfLaneId_mod_4 != 3 && halfLaneId_div_4 != 3 )
                            my_val = LU_nonzero_values[9 * w_aColIt + 3 * halfLaneId_div_4 + halfLaneId_mod_4];
                        }

                        if ( ROW_MAJOR )
                        {
                            if( halfLaneId_mod_4 != 3 && halfLaneId_div_4 != 3 )
                                my_bmAx -= my_val * my_s_mem[4 * i + halfLaneId_mod_4];
                                //my_bmAx -= my_val;
                            // else
                            //     my_bmAx = 0.0;
                        }
                        else
                        {
                            //FatalError("Haven't implemented LU_forward_3x3 for not COL_MAJOR", AMGX_ERR_NOT_SUPPORTED_TARGET);
                        }
                    }
                } // Loop over k
            } // Loop over aColIt
        } // If current_color != 0

        // Reduce bmAx terms.
#if __CUDA_ARCH__ >= 300

        if ( ROW_MAJOR )
        {
            my_bmAx += utils::shfl_xor( my_bmAx, 1, warpSize, active_mask );
            my_bmAx += utils::shfl_xor( my_bmAx, 2, warpSize, active_mask );
        }
        else
        {
            my_bmAx += utils::shfl_xor( my_bmAx, 4, warpSize, active_mask );
            my_bmAx += utils::shfl_xor( my_bmAx, 8, warpSize, active_mask );
        }

#else
        s_mem[threadIdx.x] = my_bmAx;

        if ( ROW_MAJOR )
        {
            if ( laneId < 31 ) { s_mem[threadIdx.x] = my_bmAx += s_mem[threadIdx.x + 1]; }

            if ( laneId < 30 ) { s_mem[threadIdx.x] = my_bmAx += s_mem[threadIdx.x + 2]; }
        }
        else
        {
            if ( laneId < 28 ) { s_mem[threadIdx.x] = my_bmAx += s_mem[threadIdx.x + 4]; }

            if ( laneId < 24 ) { s_mem[threadIdx.x] = my_bmAx += s_mem[threadIdx.x + 8]; }
        }

#endif

        // Store the results.
        if ( ROW_MAJOR )
        {
            if ( (halfLaneId_mod_4 == 0) && (halfLaneId_div_4 != 3) )
            {
                delta[3 * aRowId + halfLaneId_div_4] = my_bmAx;
            }
        }
        else
        {
                //FatalError("Haven't implemented LU_forward_3x3 for not COL_MAJOR", AMGX_ERR_NOT_SUPPORTED_TARGET);
        }
    }
}
#else
    FatalError("Haven't implemented LU_forward_3x3 for not EXPERIMENTAL_LU", AMGX_ERR_NOT_SUPPORTED_TARGET);
#endif

#ifdef EXPERIMENTAL_LU_FORWARD

template<typename IndexType, typename ValueTypeA, typename ValueTypeB, int CtaSize, int bsize, bool ROW_MAJOR, bool hasDiag>
__global__
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
__launch_bounds__( CtaSize, 16 )
#elif defined(__CUDA_ARCH__)
__launch_bounds__( CtaSize, 8 )
#endif
void LU_forward_5x5_kernel_warp( const IndexType *LU_row_offsets,
                                 const IndexType *LU_smaller_color_offsets,
                                 const IndexType *LU_column_indices,
                                 const ValueTypeA *LU_nonzero_values,
                                 const IndexType *A_row_offsets,
                                 const IndexType *A_column_indices,
                                 const ValueTypeA *A_nonzero_values,
                                 const IndexType *A_dia_indices,
                                 const ValueTypeB *x,
                                 const ValueTypeB *b,
                                 ValueTypeB *delta,
                                 const int *sorted_rows_by_color,
                                 const int num_rows_per_color,
                                 const int current_color,
                                 bool xIsZero )
{
    const int nWarps = CtaSize / 32; // Number of warps per Cta
    const int warpId = utils::warp_id();
    const int laneId = utils::lane_id();
    const int laneId_div_5 = laneId / 5 ;
    const int laneId_mod_5 = laneId % 5 ;

// #if __CUDA_ARCH__ < 300
//     // // Shared memory to broadcast column IDs.
//     // __shared__ volatile int s_aColIds[CtaSize];
//     // // Each thread keeps its own pointer.
//     // volatile int *my_s_aColIds = &s_aColIds[16 * halfWarpId];
// #else
//     // const int upperHalf = 16 * (laneId / 16);
// #endif
    // Shared memory needed to exchange X and delta.
    __shared__ volatile ValueTypeB s_mem[CtaSize];
    // Each thread keeps its own pointer to shared memory to avoid some extra computations.
    volatile ValueTypeB *my_s_mem = &s_mem[32 * warpId];

    // Iterate over the rows of the matrix. One warp per row.
    for ( int aRowIt = blockIdx.x * nWarps + warpId ; aRowIt < num_rows_per_color ; aRowIt += gridDim.x * nWarps )
    {
        int aRowId = sorted_rows_by_color[aRowIt];
        // Load one block of B.
        ValueTypeB my_bmAx(0);
        ValueTypeB my_bmAx_2(0);

        unsigned int active_mask = utils::activemask();

        if ( ROW_MAJOR )
        {
            if ( laneId_mod_5 == 0 && laneId_div_5 < 5)
            {
                my_bmAx = __cachingLoad(&b[5 * aRowId + laneId_div_5]);

            }
        }
        else
        {
            // if ( halfLaneId_div_4 == 0 )
            // {
            //     my_bmAx = __cachingLoad(&b[4 * aRowId + halfLaneId_mod_4]);
            // }
        }

        // Don't do anything if X is zero.
        if ( !xIsZero )
        {
            int aColBegin = A_row_offsets[aRowId  ];
            int aColEnd   = A_row_offsets[aRowId + 1];
            int aColMax = aColEnd;

            if ( hasDiag )
            {
                ++aColMax;
            }

            // Each warp load column indices of 32 nonzero blocks
            for ( ; utils::any( aColBegin < aColMax, active_mask ) ; aColBegin += 32 )
            {
                int aColIt = aColBegin + laneId;
                // Get the ID of the column.
                int aColId = -1;

                if ( aColIt < aColEnd )
                {
                    aColId = A_column_indices[aColIt];
                }

                if ( hasDiag && aColIt == aColEnd )
                {
                    aColId = aRowId;
                }

#if __CUDA_ARCH__ < 300
                // my_s_aColIds[halfLaneId] = aColId;
#endif
                // Count the number of active columns.
                int vote =  utils::ballot(aColId != -1, active_mask);
                // The number of iterations.
                // int nCols = max( __popc( vote & 0x0000ffff ), __popc( vote & 0xffff0000 ) );
                int nCols = __popc( vote & 0xffffffff );

                // Loop over columns. We compute 5 columns per iteration.
                for ( int k = 0 ; k < nCols; k += 5 )
                {
                    int my_k = k + laneId_div_5;
                    // Load 8 blocks of X.
#if __CUDA_ARCH__ < 300
                    int waColId = -1; 
                    // int waColId = my_s_aColIds[my_k];
#else
                    int waColId = utils::shfl( aColId, my_k, warpSize, active_mask );
#endif
                    if ( ! (laneId_div_5 < 5 ) || ! (my_k < nCols ) )
                        waColId = -1;

                    ValueTypeB my_x(0);

                    if ( waColId != -1 )
                    {
                        if (laneId_div_5 < 5)
                            my_x = __cachingLoad(&x[5 * waColId + laneId_mod_5 ]);
                    }

                    my_s_mem[laneId] = my_x;
                    utils::syncwarp();
                    // Load 5 blocks of A.
#pragma unroll
                    for ( int i = 0 ; i < 5 ; ++i )
                    {
                        int w_aColTmp = aColBegin + k + i, w_aColIt = -1;

                        if ( w_aColTmp < aColEnd )
                        {
                            w_aColIt = w_aColTmp;
                        }

                        if ( hasDiag && w_aColTmp == aColEnd )
                        {
                            w_aColIt = A_dia_indices[aRowId];
                        }

                        ValueTypeA my_val(0);

                        if ( w_aColIt != -1 )
                        {
                            if ( laneId_div_5 < 5 )
                                my_val = A_nonzero_values[25 * w_aColIt + laneId];
                        }

                        if ( ROW_MAJOR )
                        {
                            if ( laneId_div_5 < 5 )
                                my_bmAx -= my_val * my_s_mem[5 * i + laneId_mod_5];
                        }
                        else
                        {
                            // my_bmAx -= my_val * my_s_mem[4 * i + halfLaneId_div_4];
                        }
                    }
                } // Loop over k
            } // Loop over aColIt
        } // if xIsZero

        // Contribution from each nonzero column that has color less than yours
        if ( current_color != 0 )
        {
            // TODO: Use constant or texture here
            int aColBegin = LU_row_offsets[aRowId];
            int aColEnd   = LU_smaller_color_offsets[aRowId];

            // Each warp load column indices of 32 nonzero blocks
            for ( ; utils::any( aColBegin < aColEnd, active_mask ) ; aColBegin += 32 )
            {
                int aColIt = aColBegin + laneId;
                int aColId = -1;

                if ( aColIt < aColEnd )
                {
                    aColId = LU_column_indices[aColIt];
                }

#if __CUDA_ARCH__ < 300
                // my_s_aColIds[halfLaneId] = aColId;
#endif

                // Count the number of active columns.
                int vote =  utils::ballot(aColId != -1, active_mask);
                // The number of iterations.
                int nCols = __popc( vote & 0xffffffff );

                for ( int k = 0 ; k < nCols ; k += 5 )
                {
                    int my_k = k + laneId_div_5;
                    // Load 8 blocks of X.
#if __CUDA_ARCH__ < 300
                    int waColId = -1; 
                    // int waColId = my_s_aColIds[my_k];
#else

                    int waColId = utils::shfl( aColId, my_k, warpSize, active_mask );

#endif
                    if ( ! (laneId_div_5 < 5 ) || ! (my_k < nCols ) )
                        waColId = -1;

                    ValueTypeB my_delta(0);

                    if ( waColId != -1 )
                    {
                        if(laneId_div_5 < 5)
                            my_delta = delta[5 * waColId + laneId_mod_5];
                    }

                    my_s_mem[laneId] = my_delta;
                    utils::syncwarp(); // making sure smem write propagated
                    // Update b-Ax.
#pragma unroll
                    for ( int i = 0 ; i < 5 ; ++i )
                    {
                        int w_aColTmp = aColBegin + k + i, w_aColIt = -1;

                        if ( w_aColTmp < aColEnd )
                        {
                            w_aColIt = w_aColTmp;
                        }

                        ValueTypeA my_val(0);

                        if ( w_aColIt != -1 )
                        {
                            if(laneId_div_5 < 5)
                                my_val = LU_nonzero_values[25 * w_aColIt + laneId];
                        }

                        if ( ROW_MAJOR )
                        {
                            if(laneId_div_5 < 5)
                                my_bmAx -= my_val *  my_s_mem[5 * i + laneId_mod_5];
                        }
                        else
                        {
                            // my_bmAx -= my_val * my_s_mem[5 * i + halfLaneId_div_4];
                        }
                    }
                } // Loop over k
            } // Loop over aColIt
        } // If current_color != 0



        // Reduce bmAx terms.
#if __CUDA_ARCH__ >= 300
        my_bmAx_2 = my_bmAx;
        s_mem[threadIdx.x] = my_bmAx_2;

        if ( ROW_MAJOR )
        {
            // __syncwarp();

            // if ( laneId < 24 ) { s_mem[threadIdx.x] = my_bmAx_2 += s_mem[threadIdx.x + 1]; }
            
            // __syncwarp();

            // if ( laneId < 23 ) { s_mem[threadIdx.x] = my_bmAx_2 += s_mem[threadIdx.x + 2]; }
            
            // __syncwarp();

            // if ( laneId < 21 ) { s_mem[threadIdx.x] = my_bmAx += s_mem[threadIdx.x + 1]; }
            
            // __syncwarp();

__syncwarp();

                if ( laneId < 24 ) { my_bmAx_2 += s_mem[threadIdx.x + 1]; }
                
                __syncwarp();

                if ( laneId < 24 ) { s_mem[threadIdx.x] = my_bmAx_2; }
                
                __syncwarp();

                if ( laneId < 23 ) { my_bmAx_2 += s_mem[threadIdx.x + 2]; }
                
                __syncwarp();

                if ( laneId < 23 ) { s_mem[threadIdx.x] = my_bmAx_2; }
                
                __syncwarp();

                if ( laneId < 21 ) { my_bmAx += s_mem[threadIdx.x + 1]; }
                
                __syncwarp();

                if ( laneId < 21 ) { s_mem[threadIdx.x] = my_bmAx; }
                
                __syncwarp();
            

        }
        else
        {
            // my_bmAx += utils::shfl_xor( my_bmAx, 4, warpSize, active_mask );
            // my_bmAx += utils::shfl_xor( my_bmAx, 8, warpSize, active_mask );
        }

#else
        // s_mem[threadIdx.x] = my_bmAx_2;

        // if ( ROW_MAJOR )
        // {
        //     if ( laneId < 31 ) { s_mem[threadIdx.x] = my_bmAx_2 += s_mem[threadIdx.x + 1]; }

        //     if ( laneId < 30 ) { s_mem[threadIdx.x] = my_bmAx_2 += s_mem[threadIdx.x + 2]; }
        //     if ( laneId < 31 ) { s_mem[threadIdx.x] = my_bmAx += s_mem[threadIdx.x + 1]; }
        // }
        // else
        // {
        //     if ( laneId < 28 ) { s_mem[threadIdx.x] = my_bmAx += s_mem[threadIdx.x + 4]; }

        //     if ( laneId < 24 ) { s_mem[threadIdx.x] = my_bmAx += s_mem[threadIdx.x + 8]; }
        // }

#endif

        // Store the results.
        if ( ROW_MAJOR )
        {
            if ( laneId_mod_5 == 0 && laneId_div_5 < 5 )
            {
                delta[5 * aRowId + laneId_div_5] = my_bmAx;
            }
        }
        else
        {
            // if ( halfLaneId_div_4 == 0 )
            // {
            //     delta[4 * aRowId + halfLaneId_mod_4] = my_bmAx;
            // }
        }
    }
}

#else
        FatalError("Haven't implemented LU_forward_5x5 for not EXPERIMENTAL_LU", AMGX_ERR_NOT_SUPPORTED_TARGET);
#endif


#ifdef EXPERIMENTAL_LU_BACKWARD

template< typename IndexType, typename ValueTypeA, typename ValueTypeB, int CtaSize, bool ROW_MAJOR >
__global__
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
__launch_bounds__( CtaSize, 16 )
#elif defined(__CUDA_ARCH__)
__launch_bounds__( CtaSize, 8 )
#endif
void LU_backward_4x4_kernel_warp( const IndexType *row_offsets,
                                  const IndexType *larger_color_offsets,
                                  const IndexType *column_indices,
                                  const IndexType *dia_indices,
                                  const ValueTypeA *nonzero_values,
                                  const ValueTypeB *delta,
                                  ValueTypeB *Delta,
                                  ValueTypeB *x,
                                  const int *sorted_rows_by_color,
                                  const int num_rows_per_color,
                                  const int current_color,
                                  const int num_colors,
                                  const ValueTypeB weight,
                                  bool xIsZero )
{
    const int nHalfWarps = CtaSize / 16; // Number of half warps per CTA.
    const int warpId = utils::warp_id();
    const int laneId = utils::lane_id();
    const int halfWarpId = threadIdx.x / 16;
    const int halfLaneId = threadIdx.x % 16;
    const int halfLaneId_div_4 = halfLaneId / 4;
    const int halfLaneId_mod_4 = halfLaneId % 4;
#if __CUDA_ARCH__ < 300
    // Shared memory to broadcast column IDs.
    __shared__ volatile int s_aColIds[CtaSize];
    // Each thread keeps its own pointer.
    volatile int *my_s_aColIds = &s_aColIds[16 * halfWarpId];
#else
    const int upperHalf = 16 * (laneId / 16);
#endif
    // Shared memory needed to exchange X and delta.
    __shared__ volatile ValueTypeB s_mem[CtaSize];
    // Each thread keeps its own pointer to shared memory to avoid some extra computations.
    volatile ValueTypeB *my_s_mem = &s_mem[16 * halfWarpId];

    // Iterate over the rows of the matrix. One warp per two rows.
    for ( int aRowIt = blockIdx.x * nHalfWarps + halfWarpId ; aRowIt < num_rows_per_color ; aRowIt += gridDim.x * nHalfWarps )
    {
        int aRowId = sorted_rows_by_color[aRowIt];
        unsigned int active_mask = utils::activemask();
        // Load one block of B.
        ValueTypeB my_bmAx(0);

        if ( ROW_MAJOR )
        {
            if ( halfLaneId_mod_4 == 0 )
            {
                my_bmAx = delta[4 * aRowId + halfLaneId_div_4];
            }
        }
        else
        {
            if ( halfLaneId_div_4 == 0 )
            {
                my_bmAx = delta[4 * aRowId + halfLaneId_mod_4];
            }
        }

        // Don't do anything if the color is not the interesting one.
        if ( current_color != num_colors - 1 )
        {
            // The range of the rows.
            int aColBegin = larger_color_offsets[aRowId], aColEnd = row_offsets[aRowId + 1];

            // Each warp load column indices of 16 nonzero blocks
            for ( ; utils::any( aColBegin < aColEnd, active_mask ) ; aColBegin += 16 )
            {
                int aColIt = aColBegin + halfLaneId;
                // Get the ID of the column.
                int aColId = -1;

                if ( aColIt < aColEnd )
                {
                    aColId = column_indices[aColIt];
                }

#if __CUDA_ARCH__ < 300
                my_s_aColIds[halfLaneId] = aColId;
#endif

                // Loop over columns. We compute 8 columns per iteration.
                for ( int k = 0 ; k < 16 ; k += 4 )
                {
                    int my_k = k + halfLaneId_div_4;
                    // Exchange column indices.
#if __CUDA_ARCH__ < 300
                    int waColId = my_s_aColIds[my_k];
#else
                    int waColId = utils::shfl( aColId, upperHalf + my_k, warpSize, active_mask );
#endif
                    // Load 8 blocks of X if needed.
                    ValueTypeB *my_ptr = Delta;

                    if ( xIsZero )
                    {
                        my_ptr = x;
                    }

                    ValueTypeB my_x(0);

                    if ( waColId != -1 )
                    {
                        my_x = my_ptr[4 * waColId + halfLaneId_mod_4];
                    }

                    my_s_mem[halfLaneId] = my_x;
                    utils::syncwarp();
                    // Load 8 blocks of A.
#pragma unroll
                    for ( int i = 0 ; i < 4 ; ++i )
                    {
                        int w_aColTmp = aColBegin + k + i, w_aColIt = -1;

                        if ( w_aColTmp < aColEnd )
                        {
                            w_aColIt = w_aColTmp;
                        }

                        ValueTypeA my_val(0);

                        if ( w_aColIt != -1 )
                        {
                            my_val = nonzero_values[16 * w_aColIt + halfLaneId];
                        }

                        if ( ROW_MAJOR )
                        {
                            my_bmAx -= my_val * my_s_mem[4 * i + halfLaneId_mod_4];
                        }
                        else
                        {
                            my_bmAx -= my_val * my_s_mem[4 * i + halfLaneId_div_4];
                        }
                    }
                } // Loop over k
            } // Loop over aColIt

            // Reduce bmAx terms.
#if __CUDA_ARCH__ >= 300

            if ( ROW_MAJOR )
            {
                my_bmAx += utils::shfl_xor( my_bmAx, 1, warpSize, active_mask );
                my_bmAx += utils::shfl_xor( my_bmAx, 2, warpSize, active_mask );
            }
            else
            {
                my_bmAx += utils::shfl_xor( my_bmAx, 4, warpSize, active_mask );
                my_bmAx += utils::shfl_xor( my_bmAx, 8, warpSize, active_mask );
            }

#else
            s_mem[threadIdx.x] = my_bmAx;

            if ( ROW_MAJOR )
            {
                if ( laneId < 31 ) { s_mem[threadIdx.x] = my_bmAx += s_mem[threadIdx.x + 1]; }

                if ( laneId < 30 ) { s_mem[threadIdx.x] = my_bmAx += s_mem[threadIdx.x + 2]; }
            }
            else
            {
                if ( laneId < 28 ) { s_mem[threadIdx.x] = my_bmAx += s_mem[threadIdx.x + 4]; }

                if ( laneId < 24 ) { s_mem[threadIdx.x] = my_bmAx += s_mem[threadIdx.x + 8]; }
            }

#endif
        } // if current_color != num_colors-1

        // Update the shared terms.
        if ( ROW_MAJOR )
        {
            if ( halfLaneId_mod_4 == 0 )
            {
                my_s_mem[halfLaneId_div_4] = my_bmAx;
            }
        }
        else
        {
            if ( halfLaneId_div_4 == 0 )
            {
                my_s_mem[halfLaneId_mod_4] = my_bmAx;
            }
        }

        // Update the diagonal term.
        int w_aColIt = dia_indices[aRowId];
        ValueTypeA my_val(0);
        utils::syncwarp();

        if ( w_aColIt != -1 )
        {
            my_val = nonzero_values[16 * w_aColIt + halfLaneId];
        }

        if ( ROW_MAJOR )
        {
            my_bmAx = my_val * my_s_mem[halfLaneId_mod_4];
        }
        else
        {
            my_bmAx = my_val * my_s_mem[halfLaneId_div_4];
        }

        // Regroup results.
#if __CUDA_ARCH__ >= 300

        if ( ROW_MAJOR )
        {
            my_bmAx += utils::shfl_xor( my_bmAx, 1 );
            my_bmAx += utils::shfl_xor( my_bmAx, 2 );
        }
        else
        {
            my_bmAx += utils::shfl_xor( my_bmAx, 4 );
            my_bmAx += utils::shfl_xor( my_bmAx, 8 );
        }

#else
        s_mem[threadIdx.x] = my_bmAx;

        if ( ROW_MAJOR )
        {
            if ( laneId < 31 ) { s_mem[threadIdx.x] = my_bmAx += s_mem[threadIdx.x + 1]; }

            if ( laneId < 30 ) { s_mem[threadIdx.x] = my_bmAx += s_mem[threadIdx.x + 2]; }
        }
        else
        {
            if ( laneId < 28 ) { s_mem[threadIdx.x] = my_bmAx += s_mem[threadIdx.x + 4]; }

            if ( laneId < 24 ) { s_mem[threadIdx.x] = my_bmAx += s_mem[threadIdx.x + 8]; }
        }

#endif

        // Store the results.
        if ( ROW_MAJOR )
        {
            ValueTypeB my_x(0);

            if ( !xIsZero && halfLaneId_mod_4 == 0 )
            {
                my_x = x[4 * aRowId + halfLaneId_div_4];
            }

            my_x += weight * my_bmAx;

            if ( !xIsZero && halfLaneId_mod_4 == 0 )
            {
                Delta[4 * aRowId + halfLaneId_div_4] = my_bmAx;
            }

            if ( halfLaneId_mod_4 == 0 )
            {
                x[4 * aRowId + halfLaneId_div_4] = my_x;
            }
        }
        else
        {
            ValueTypeB my_x(0);

            if ( !xIsZero && halfLaneId_div_4 == 0 )
            {
                my_x = x[4 * aRowId + halfLaneId_mod_4];
            }

            my_x += weight * my_bmAx;

            if ( !xIsZero && halfLaneId_div_4 == 0 )
            {
                Delta[4 * aRowId + halfLaneId_mod_4] = my_bmAx;
            }

            if ( halfLaneId_div_4 == 0 )
            {
                x[4 * aRowId + halfLaneId_mod_4] = my_x;
            }
        }
    }
}

#else

template<typename IndexType, typename ValueTypeA, typename ValueTypeB, int blockrows_per_cta, int blockrows_per_warp, int bsize, bool ROW_MAJOR>
__global__
void LU_backward_4x4_kernel(const IndexType *row_offsets, const IndexType *larger_color_offsets, const IndexType *column_indices, const IndexType *dia_indices, const ValueTypeA *nonzero_values,
                            const ValueTypeB *delta,  ValueTypeB *Delta, ValueTypeB *x, const int *sorted_rows_by_color,
                            const int num_rows_per_color, const int current_color, const int num_colors, const ValueTypeB weight, bool xIsZero)


{
    int warp_id = threadIdx.x / 32;
    int warp_thread_id = threadIdx.x & 31;

    // padding row blocks to fit in a single warp
    if ( warp_thread_id >= blockrows_per_warp * bsize ) { return; }

    // new thread id with padding
    int tid = warp_id * blockrows_per_warp * bsize + warp_thread_id;
    // Here we use one thread per row (not block row)
    int cta_blockrow_id = (tid) / bsize;
    int blockrow_id = blockIdx.x * blockrows_per_cta + cta_blockrow_id;
    const int vec_entry_index = tid - cta_blockrow_id * bsize;
    volatile __shared__ ValueTypeB s_x_temp[ bsize * blockrows_per_cta];
    int offset, s_offset, i;
    ValueTypeB bmAx, temp[bsize];

    while (blockrow_id < num_rows_per_color &&  cta_blockrow_id < blockrows_per_cta)
    {
        i = sorted_rows_by_color[blockrow_id];
        // Load RHS and x
        offset = i * bsize + vec_entry_index;
        bmAx = delta[offset];

        // Contribution from each nonzero column that has color less than yours
        if (current_color != num_colors)
        {
            int jmin = larger_color_offsets[i];
            int jmax = row_offsets[i + 1];

            for (int jind = jmin; jind < jmax; jind++)
            {
                IndexType jcol = column_indices[jind];
                offset = jcol * bsize + vec_entry_index;

                if (xIsZero)
                {
                    s_x_temp[tid] = ld_cg(x + offset);
                }
                else
                {
                    s_x_temp[tid] = ld_cg(Delta + offset);
                }

                // Load nonzero_values
                if (ROW_MAJOR)
                {
                    offset = jind * bsize * bsize + vec_entry_index * bsize;
                    loadAsVector<bsize>(nonzero_values + offset, temp);
                }
                else
                {
                    offset = jind * bsize * bsize + vec_entry_index;
#pragma unroll
                    for (int m = 0; m < bsize; m++)
                    {
                        temp[m] = nonzero_values[offset + bsize * m];
                    }
                }

                // Do matrix multiply
                s_offset = cta_blockrow_id * bsize;
#pragma unroll
                for (int m = 0; m < bsize; m++)
                {
                    bmAx -= temp[m] * s_x_temp[s_offset++];
                }
            }
        }

        s_x_temp[tid] = bmAx;
        bmAx = 0.;

        // Load diagonals (which store the inverse)
        if (ROW_MAJOR)
        {
            offset = dia_indices[i] * bsize * bsize + vec_entry_index * bsize;
            loadAsVector<bsize>(nonzero_values + offset, temp);
        }
        else
        {
            offset = dia_indices[i] * bsize * bsize + vec_entry_index;
#pragma unroll
            for (int m = 0; m < bsize; m++)
            {
                temp[m] = nonzero_values[offset + bsize * m];
            }
        }

        // Do matrix-vector multiply
        s_offset = cta_blockrow_id * bsize;
#pragma unroll
        for (int m = 0; m < bsize; m++)
        {
            bmAx += temp[m] * s_x_temp[s_offset++];
        }

        offset = i * bsize + vec_entry_index;

        if (xIsZero)
        {
            x[offset] = weight * bmAx;
        }
        else
        {
            Delta[offset] = bmAx;
            x[offset] += weight * bmAx ;
        }

        blockrow_id += blockrows_per_cta * gridDim.x;
    }
}

#endif

#ifdef EXPERIMENTAL_LU_BACKWARD

template< typename IndexType, typename ValueTypeA, typename ValueTypeB, int CtaSize, bool ROW_MAJOR >
__global__
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
__launch_bounds__( CtaSize, 16 )
#elif defined(__CUDA_ARCH__)
__launch_bounds__( CtaSize, 8 )
#endif
void LU_backward_5x5_kernel_warp( const IndexType *row_offsets,
                                  const IndexType *larger_color_offsets,
                                  const IndexType *column_indices,
                                  const IndexType *dia_indices,
                                  const ValueTypeA *nonzero_values,
                                  const ValueTypeB *delta,
                                  ValueTypeB *Delta,
                                  ValueTypeB *x,
                                  const int *sorted_rows_by_color,
                                  const int num_rows_per_color,
                                  const int current_color,
                                  const int num_colors,
                                  const ValueTypeB weight,
                                  bool xIsZero )
{
    const int nWarps = CtaSize / 32; // Number of half warps per CTA.
    const int warpId = utils::warp_id();
    const int laneId = utils::lane_id();
    const int laneId_div_5 = laneId / 5;
    const int laneId_mod_5 = laneId % 5;

// #if __CUDA_ARCH__ < 300
//     // // Shared memory to broadcast column IDs.
//     // __shared__ volatile int s_aColIds[CtaSize];
//     // // Each thread keeps its own pointer.
//     // volatile int *my_s_aColIds = &s_aColIds[16 * halfWarpId];
// #else
//     // const int upperHalf = 16 * (laneId / 16);
// #endif
    // Shared memory needed to exchange X and delta.
    __shared__ volatile ValueTypeB s_mem[CtaSize];
    // Each thread keeps its own pointer to shared memory to avoid some extra computations.
    volatile ValueTypeB *my_s_mem = &s_mem[32 * warpId];

    // Iterate over the rows of the matrix. One warp per two rows.
    for ( int aRowIt = blockIdx.x * nWarps + warpId ; aRowIt < num_rows_per_color ; aRowIt += gridDim.x * nWarps )
    {
        int aRowId = sorted_rows_by_color[aRowIt];
        unsigned int active_mask = utils::activemask();
        // Load one block of B.
        ValueTypeB temp_mul(0);
        ValueTypeB my_bmAx(0);
        ValueTypeB my_bmAx_2(0);
                    utils::syncwarp();
        if ( ROW_MAJOR )
        {
            if ( laneId_mod_5 == 0 && laneId_div_5 < 5 )
            {
                my_bmAx = delta[5 * aRowId + laneId_div_5];
            }
        }
        else
        {
            // if ( halfLaneId_div_4 == 0 )
            // {
            //     my_bmAx = delta[4 * aRowId + halfLaneId_mod_4];
            // }
        }
                    utils::syncwarp();
        // Don't do anything if the color is not the interesting one.
        if ( current_color != num_colors - 1 )
        {
            // The range of the rows.
            int aColBegin = larger_color_offsets[aRowId];
            int aColEnd   = row_offsets[aRowId + 1];

            // Each warp load column indices of 16 nonzero blocks
            for ( ; utils::any( aColBegin < aColEnd, active_mask ) ; aColBegin += 32 )
            {
                int aColIt = aColBegin + laneId;
                // Get the ID of the column.
                int aColId = -1;

                if ( aColIt < aColEnd )
                {
                    aColId = column_indices[aColIt];
                }
                    utils::syncwarp();
#if __CUDA_ARCH__ < 300
                // my_s_aColIds[halfLaneId] = aColId;
#endif
                int vote =  utils::ballot(aColId != -1, active_mask);
                int nCols = __popc( vote & 0xffffffff );
                    utils::syncwarp();
                // Loop over columns. We compute 8 columns per iteration.
                for ( int k = 0 ; k <  nCols; k += 5 )
                {
                    int my_k = k + laneId_div_5;
                    utils::syncwarp();
                    // Exchange column indices.
#if __CUDA_ARCH__ < 300
                    int waColId = -1;
                    // int waColId = my_s_aColIds[my_k];
#else
                    int waColId = utils::shfl( aColId, my_k, warpSize, active_mask );
#endif
                    if ( ! (laneId_div_5 < 5 ) || ! (my_k < nCols ) )
                        waColId = -1;
                    utils::syncwarp();                    
                    // Load 8 blocks of X if needed.
                    ValueTypeB *my_ptr = Delta;

                    if ( xIsZero )
                    {
                        my_ptr = x;
                    }

                    ValueTypeB my_x(0);
                    utils::syncwarp();
                    if ( waColId != -1 )
                    {
                        if(laneId_div_5 < 5)
                            my_x = my_ptr[5 * waColId + laneId_mod_5];
                    }

                    my_s_mem[laneId] = my_x;
                    utils::syncwarp();
                    // Load 8 blocks of A.
#pragma unroll
                    for ( int i = 0 ; i < 5 ; ++i )
                    {
                        int w_aColTmp = aColBegin + k + i, w_aColIt = -1;

                        if ( w_aColTmp < aColEnd )
                        {
                            w_aColIt = w_aColTmp;
                        }

                        ValueTypeA my_val(0);
                            utils::syncwarp();
                        if ( w_aColIt != -1 )
                        {
                            if ( laneId_div_5 < 5 )
                                my_val = nonzero_values[25 * w_aColIt + laneId];
                        }
                            utils::syncwarp();
                        if ( ROW_MAJOR )
                        {
                            if ( laneId_div_5 < 5 )
                                {
                                    my_bmAx -= my_val * my_s_mem[5 * i + laneId_mod_5];
                                    //  temp_mul = (1.0 + my_val - my_val) * my_val * my_s_mem[5 * i + laneId_mod_5];
                                    // my_bmAx = -temp_mul + my_bmAx;
                                }
                            utils::syncwarp();
                        }
                        else
                        {
                            // my_bmAx -= my_val * my_s_mem[4 * i + halfLaneId_div_4];
                        }
                    }
                } // Loop over k
            } // Loop over aColIt

            // Reduce bmAx terms.
#if __CUDA_ARCH__ >= 300
            my_bmAx_2 = my_bmAx;
            s_mem[threadIdx.x] = my_bmAx_2;
            if ( ROW_MAJOR )
            {
                // __syncwarp();

                // if ( laneId < 24 ) { s_mem[threadIdx.x] = my_bmAx_2 += s_mem[threadIdx.x + 1]; }
                
                // __syncwarp();

                // if ( laneId < 23 ) { s_mem[threadIdx.x] = my_bmAx_2 += s_mem[threadIdx.x + 2]; }
                
                // __syncwarp();

                // if ( laneId < 21 ) { s_mem[threadIdx.x] = my_bmAx += s_mem[threadIdx.x + 1]; }
                
                // __syncwarp();

                __syncwarp();

                if ( laneId < 24 ) { my_bmAx_2 += s_mem[threadIdx.x + 1]; }
                
                __syncwarp();

                if ( laneId < 24 ) { s_mem[threadIdx.x] = my_bmAx_2; }
                
                __syncwarp();

                if ( laneId < 23 ) { my_bmAx_2 += s_mem[threadIdx.x + 2]; }
                
                __syncwarp();

                if ( laneId < 23 ) { s_mem[threadIdx.x] = my_bmAx_2; }
                
                __syncwarp();

                if ( laneId < 21 ) { my_bmAx += s_mem[threadIdx.x + 1]; }
                
                __syncwarp();

                if ( laneId < 21 ) { s_mem[threadIdx.x] = my_bmAx; }
                
                __syncwarp();




            }
            else
            {
                // my_bmAx += utils::shfl_xor( my_bmAx, 4, warpSize, active_mask );
                // my_bmAx += utils::shfl_xor( my_bmAx, 8, warpSize, active_mask );
            }

#else
            // s_mem[threadIdx.x] = my_bmAx;

            // if ( ROW_MAJOR )
            // {
            //     if ( laneId < 31 ) { s_mem[threadIdx.x] = my_bmAx += s_mem[threadIdx.x + 1]; }

            //     if ( laneId < 30 ) { s_mem[threadIdx.x] = my_bmAx += s_mem[threadIdx.x + 2]; }
            // }
            // else
            // {
            //     if ( laneId < 28 ) { s_mem[threadIdx.x] = my_bmAx += s_mem[threadIdx.x + 4]; }

            //     if ( laneId < 24 ) { s_mem[threadIdx.x] = my_bmAx += s_mem[threadIdx.x + 8]; }
            // }

#endif
        } // if current_color != num_colors-1

        // Update the shared terms.
        // if ( ROW_MAJOR )
        // {
        //     if ( laneId_mod_5 == 0 && laneId_div_5 < 5)
        //     {
        //         my_s_mem[laneId_div_5] = my_bmAx;
        //         // if(aRowId == 62114)
        //         //     printf("%lf %d\n",my_bmAx*1e12,laneId_div_5);
        //     }
        // }
        // else
        // {
        //     // if ( halfLaneId_div_4 == 0 )
        //     // {
        //     //     my_s_mem[halfLaneId_mod_4] = my_bmAx;
        //     // }
        // }

        // // Update the diagonal term.
        // int w_aColIt = dia_indices[aRowId];
        // ValueTypeA my_val(0);
        // utils::syncwarp();

        // if ( w_aColIt != -1 )
        // {
        //     if(laneId_div_5 < 5)
        //        my_val = nonzero_values[25 * w_aColIt + laneId];
        // }
        // utils::syncwarp();
        // if ( ROW_MAJOR )
        // {
        //     if(laneId_div_5 < 5)
        //     {
        //          my_bmAx = my_val * my_s_mem[laneId_mod_5];
        //     }   
            
        // }
        // else
        // {
        //     // if(laneId_div_5 < 5)
        //     //     my_bmAx = my_val * my_s_mem[laneId_div_5];
        // }
        // utils::syncwarp();
//         // Regroup results.
// #if __CUDA_ARCH__ >= 300
//         my_bmAx_2 = my_bmAx;
//         s_mem[threadIdx.x] = my_bmAx_2;

//         if ( ROW_MAJOR )
//         {
//             // __syncwarp();

//             // if ( laneId < 24 ) { s_mem[threadIdx.x] = my_bmAx_2 += s_mem[threadIdx.x + 1]; }
            
//             // __syncwarp();

//             // if ( laneId < 23 ) { s_mem[threadIdx.x] = my_bmAx_2 += s_mem[threadIdx.x + 2]; }
            
//             // __syncwarp();

//             // if ( laneId < 21 ) { s_mem[threadIdx.x] = my_bmAx += s_mem[threadIdx.x + 1]; }
            
//             // __syncwarp();

//                 __syncwarp();

//                 if ( laneId < 24 ) { my_bmAx_2 += s_mem[threadIdx.x + 1]; }
                
//                 __syncwarp();

//                 if ( laneId < 24 ) { s_mem[threadIdx.x] = my_bmAx_2; }
                
//                 __syncwarp();

//                 if ( laneId < 23 ) { my_bmAx_2 += s_mem[threadIdx.x + 2]; }
                
//                 __syncwarp();

//                 if ( laneId < 23 ) { s_mem[threadIdx.x] = my_bmAx_2; }
                
//                 __syncwarp();

//                 if ( laneId < 21 ) { my_bmAx += s_mem[threadIdx.x + 1]; }
                
//                 __syncwarp();

//                 if ( laneId < 21 ) { s_mem[threadIdx.x] = my_bmAx; }
                
//                 __syncwarp();

//         }
//         else
//         {
//             // my_bmAx += utils::shfl_xor( my_bmAx, 4 );
//             // my_bmAx += utils::shfl_xor( my_bmAx, 8 );
//         }

// #else
//         // s_mem[threadIdx.x] = my_bmAx;

//         // if ( ROW_MAJOR )
//         // {
//         //     if ( laneId < 31 ) { s_mem[threadIdx.x] = my_bmAx += s_mem[threadIdx.x + 1]; }

//         //     if ( laneId < 30 ) { s_mem[threadIdx.x] = my_bmAx += s_mem[threadIdx.x + 2]; }
//         // }
//         // else
//         // {
//         //     if ( laneId < 28 ) { s_mem[threadIdx.x] = my_bmAx += s_mem[threadIdx.x + 4]; }

//         //     if ( laneId < 24 ) { s_mem[threadIdx.x] = my_bmAx += s_mem[threadIdx.x + 8]; }
//         // }

// #endif

        // Store the results.
        if ( ROW_MAJOR )
        {
            ValueTypeB my_x(0);
            utils::syncwarp();
            if ( !xIsZero && laneId_mod_5 == 0 && laneId_div_5 < 5)
            {
                my_x = x[5 * aRowId + laneId_div_5];
            }

            // my_x += weight * my_bmAx;

            
            if ( !xIsZero && laneId_mod_5 == 0 && laneId_div_5 < 5)
            {
                Delta[5 * aRowId + laneId_div_5] = my_bmAx;
            }
            utils::syncwarp();
            if ( laneId_mod_5 == 0 && laneId_div_5 < 5)
            {
                x[5 * aRowId + laneId_div_5] = my_bmAx;
                // if(aRowId == 62114)
                //     printf("%lf %d\n",my_x*1e12,laneId_div_5);
            }
        }
        else
        {
            // ValueTypeB my_x(0);

            // if ( !xIsZero && halfLaneId_div_4 == 0 )
            // {
            //     my_x = x[4 * aRowId + halfLaneId_mod_4];
            // }

            // my_x += weight * my_bmAx;

            // if ( !xIsZero && halfLaneId_div_4 == 0 )
            // {
            //     Delta[4 * aRowId + halfLaneId_mod_4] = my_bmAx;
            // }

            // if ( halfLaneId_div_4 == 0 )
            // {
            //     x[4 * aRowId + halfLaneId_mod_4] = my_x;
            // }
        }
    }

}
#else
    FatalError("Haven't implemented LU_backward_5x5 for not EXPERIMENTAL_LU", AMGX_ERR_NOT_SUPPORTED_TARGET);
#endif


#ifdef EXPERIMENTAL_LU_BACKWARD

template< typename IndexType, typename ValueTypeA, typename ValueTypeB, int CtaSize, bool ROW_MAJOR >
__global__
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
__launch_bounds__( CtaSize, 16 )
#elif defined(__CUDA_ARCH__)
__launch_bounds__( CtaSize, 8 )
#endif
void LU_backward_3x3_kernel_warp( const IndexType *row_offsets,
                                  const IndexType *larger_color_offsets,
                                  const IndexType *column_indices,
                                  const IndexType *dia_indices,
                                  const ValueTypeA *nonzero_values,
                                  const ValueTypeB *delta,
                                  ValueTypeB *Delta,
                                  ValueTypeB *x,
                                  const int *sorted_rows_by_color,
                                  const int num_rows_per_color,
                                  const int current_color,
                                  const int num_colors,
                                  const ValueTypeB weight,
                                  bool xIsZero )
{
    const int nHalfWarps = CtaSize / 16; // Number of half warps per CTA.
    const int warpId = utils::warp_id();
    const int laneId = utils::lane_id();
    const int halfWarpId = threadIdx.x / 16;
    const int halfLaneId = threadIdx.x % 16;
    const int halfLaneId_div_4 = halfLaneId / 4;
    const int halfLaneId_mod_4 = halfLaneId % 4;

#if __CUDA_ARCH__ < 300
    // Shared memory to broadcast column IDs.
    __shared__ volatile int s_aColIds[CtaSize];
    // Each thread keeps its own pointer.
    volatile int *my_s_aColIds = &s_aColIds[16 * halfWarpId];
#else
    const int upperHalf = 16 * (laneId / 16);
#endif
    // Shared memory needed to exchange X and delta.
    __shared__ volatile ValueTypeB s_mem[CtaSize];
    // Each thread keeps its own pointer to shared memory to avoid some extra computations.
    volatile ValueTypeB *my_s_mem = &s_mem[16 * halfWarpId];

    // Iterate over the rows of the matrix. One warp per two rows.
    for ( int aRowIt = blockIdx.x * nHalfWarps + halfWarpId ; aRowIt < num_rows_per_color ; aRowIt += gridDim.x * nHalfWarps )
    {
        int aRowId = sorted_rows_by_color[aRowIt];
        unsigned int active_mask = utils::activemask();
        // Load one block of B.
        ValueTypeB my_bmAx(0);

        if ( ROW_MAJOR )
        {
            if ( (halfLaneId_mod_4 == 0) && (halfLaneId_div_4 != 3) )
            {
                my_bmAx = delta[3 * aRowId + halfLaneId_div_4];
            }
        }
        else
        {
            // if ( halfLaneId_div_4 == 0 )
            // {
            //     my_bmAx = delta[4 * aRowId + halfLaneId_mod_4];
            // }
        }

        //Don't do anything if the color is not the interesting one.
        if ( current_color != num_colors - 1 )
        {
            // The range of the rows.
            int aColBegin = larger_color_offsets[aRowId], aColEnd = row_offsets[aRowId + 1];

            // Each warp load column indices of 16 nonzero blocks
            for ( ; utils::any( aColBegin < aColEnd, active_mask ) ; aColBegin += 16 )
            {
                int aColIt = aColBegin + halfLaneId;
                // Get the ID of the column.
                int aColId = -1;

                if ( aColIt < aColEnd )
                {
                    aColId = column_indices[aColIt];
                }

#if __CUDA_ARCH__ < 300
                my_s_aColIds[halfLaneId] = aColId;
#endif

                // Loop over columns. We compute 8 columns per iteration.
                for ( int k = 0 ; k < 16 ; k += 4 )
                {
                    int my_k = k + halfLaneId_div_4;
                    // Exchange column indices.
#if __CUDA_ARCH__ < 300
                    int waColId = my_s_aColIds[my_k];
#else
                    int waColId = utils::shfl( aColId, upperHalf + my_k, warpSize, active_mask );
#endif
                    // Load 8 blocks of X if needed.
                    ValueTypeB *my_ptr = Delta;

                    if ( xIsZero )
                    {
                        my_ptr = x;
                    }

                    ValueTypeB my_x(0);

                    //if ( waColId != -1  && halfLaneId_mod_4 != 3 && halfLaneId_div_4 != 3 )
                    if ( waColId != -1  && halfLaneId_mod_4 != 3 )
                    {
                        my_x = my_ptr[3 * waColId + halfLaneId_mod_4];
                    }
                    
                    my_s_mem[halfLaneId] = my_x;
                    utils::syncwarp();
                    // Load 8 blocks of A.
#pragma unroll
                    for ( int i = 0 ; i < 4 ; ++i )
                    {
                        int w_aColTmp = aColBegin + k + i, w_aColIt = -1;

                        if ( w_aColTmp < aColEnd )
                        {
                            w_aColIt = w_aColTmp;
                        }

                        ValueTypeA my_val(0);

                        if ( w_aColIt != -1 )
                        {
                            if( halfLaneId_mod_4 != 3 && halfLaneId_div_4 != 3 )
                            my_val = nonzero_values[9 * w_aColIt + 3 * halfLaneId_div_4 + halfLaneId_mod_4];
                        }

                        if ( ROW_MAJOR )
                        {
                            if( halfLaneId_mod_4 != 3 && halfLaneId_div_4 != 3  )
                                my_bmAx -= my_val * my_s_mem[4 * i + halfLaneId_mod_4];
                        }
                        else
                        {
                            // my_bmAx -= my_val * my_s_mem[4 * i + halfLaneId_div_4];
                        }
                    }
                } // Loop over k
            } // Loop over aColIt

            // Reduce bmAx terms.
#if __CUDA_ARCH__ >= 300

            if ( ROW_MAJOR )
            {
                my_bmAx += utils::shfl_xor( my_bmAx, 1, warpSize, active_mask );
                my_bmAx += utils::shfl_xor( my_bmAx, 2, warpSize, active_mask );
            }
            else
            {
                my_bmAx += utils::shfl_xor( my_bmAx, 4, warpSize, active_mask );
                my_bmAx += utils::shfl_xor( my_bmAx, 8, warpSize, active_mask );
            }

#else
            s_mem[threadIdx.x] = my_bmAx;

            if ( ROW_MAJOR )
            {
                if ( laneId < 31 ) { s_mem[threadIdx.x] = my_bmAx += s_mem[threadIdx.x + 1]; }

                if ( laneId < 30 ) { s_mem[threadIdx.x] = my_bmAx += s_mem[threadIdx.x + 2]; }
            }
            else
            {
                if ( laneId < 28 ) { s_mem[threadIdx.x] = my_bmAx += s_mem[threadIdx.x + 4]; }

                if ( laneId < 24 ) { s_mem[threadIdx.x] = my_bmAx += s_mem[threadIdx.x + 8]; }
            }

#endif
        } // if current_color != num_colors-1

        // Update the shared terms.
        if ( ROW_MAJOR )
        {
            if ( (halfLaneId_mod_4 == 0) && (halfLaneId_div_4 != 3) )
            {
                my_s_mem[halfLaneId_div_4] = my_bmAx;
            }
        }
        else
        {
            // if ( halfLaneId_div_4 == 0 )
            // {
            //     my_s_mem[halfLaneId_mod_4] = my_bmAx;
            // }
        }

        // Update the diagonal term.
        int w_aColIt = dia_indices[aRowId];
        ValueTypeA my_val(0);
        utils::syncwarp();

        if ( w_aColIt != -1 )
        {
            if( halfLaneId_mod_4 != 3 && halfLaneId_div_4 != 3)
                my_val = nonzero_values[9 * w_aColIt + 3 * halfLaneId_div_4 + halfLaneId_mod_4];
        }

        if ( ROW_MAJOR )
        {
            if( halfLaneId_mod_4 != 3 && halfLaneId_div_4 != 3){
                my_bmAx = my_val * my_s_mem[halfLaneId_mod_4];
            }
            else{
                my_bmAx = 0.0;
            }
        }
        else
        {
            // my_bmAx = my_val * my_s_mem[halfLaneId_div_4];
        }

        // Regroup results.
#if __CUDA_ARCH__ >= 300

        if ( ROW_MAJOR )
        {
            my_bmAx += utils::shfl_xor( my_bmAx, 1 );
            my_bmAx += utils::shfl_xor( my_bmAx, 2 );
        }
        else
        {
            my_bmAx += utils::shfl_xor( my_bmAx, 4 );
            my_bmAx += utils::shfl_xor( my_bmAx, 8 );
        }

#else
        s_mem[threadIdx.x] = my_bmAx;

        if ( ROW_MAJOR )
        {
            if ( laneId < 31 ) { s_mem[threadIdx.x] = my_bmAx += s_mem[threadIdx.x + 1]; }

            if ( laneId < 30 ) { s_mem[threadIdx.x] = my_bmAx += s_mem[threadIdx.x + 2]; }
        }
        else
        {
            if ( laneId < 28 ) { s_mem[threadIdx.x] = my_bmAx += s_mem[threadIdx.x + 4]; }

            if ( laneId < 24 ) { s_mem[threadIdx.x] = my_bmAx += s_mem[threadIdx.x + 8]; }
        }

#endif

        // Store the results.
        if ( ROW_MAJOR )
        {
            ValueTypeB my_x(0);

            if ( !xIsZero && halfLaneId_mod_4 == 0 && halfLaneId_div_4 != 3 )
            {
                my_x = x[3 * aRowId + halfLaneId_div_4];
            }

            my_x += weight * my_bmAx;

            if ( !xIsZero && halfLaneId_mod_4 == 0 && halfLaneId_div_4 != 3 )
            {
                Delta[3 * aRowId + halfLaneId_div_4] = my_bmAx;
            }
            
            if ( halfLaneId_mod_4 == 0 && halfLaneId_div_4 != 3)
            {
                x[3 * aRowId + halfLaneId_div_4] = my_x;
            }
        }
        else
        {
            // ValueTypeB my_x(0);

            // if ( !xIsZero && halfLaneId_div_4 == 0 )
            // {
            //     my_x = x[4 * aRowId + halfLaneId_mod_4];
            // }

            // my_x += weight * my_bmAx;

            // if ( !xIsZero && halfLaneId_div_4 == 0 )
            // {
            //     Delta[4 * aRowId + halfLaneId_mod_4] = my_bmAx;
            // }

            // if ( halfLaneId_div_4 == 0 )
            // {
            //     x[4 * aRowId + halfLaneId_mod_4] = my_x;
            // }
        }
    }
}

#else
    FatalError("Haven't implemented LU_backward_3x3 for not EXPERIMENTAL_LU", AMGX_ERR_NOT_SUPPORTED_TARGET);
#endif


// Assumptions:
// CtaSize must be multiple of 32
// SMemSize should be larger than the maximum number of columns in the matrix
// Matrix B is superset of matrix A

template< int CtaSize, int SMemSize>
__global__ __launch_bounds__( CtaSize )
void
computeAtoLUmapping_kernel( int A_nRows,
                            const int *__restrict A_row_offsets,
                            const int *__restrict A_col_indices,
                            const int *__restrict B_row_offsets,
                            const int *__restrict B_col_indices,
                            int *__restrict AtoBmapping,
                            int *wk_returnValue )
{
    const int nWarps = CtaSize / 32; // Number of warps per Cta
    const int warpId = utils::warp_id();
    const int laneId = utils::lane_id();
    // Rows are stored in SMEM. Linear storage.
    __shared__ volatile int s_colInd[nWarps][SMemSize];
    // The row this warp is responsible for
    int aRowId = blockIdx.x * nWarps + warpId;

    // Loop over rows of A.
    for ( ; aRowId < A_nRows ; aRowId += nWarps * gridDim.x )
    {
        // Insert all the column indices of matrix B in the shared memory table
        int bColBeg = B_row_offsets[aRowId];
        int bColEnd = B_row_offsets[aRowId + 1];
        // The number of columns.
        const int nCols = bColEnd - bColBeg;

        //TODO: Add fallback for cases where number of nonzeros exceed SMemSize
        if ( nCols > SMemSize )
        {
            wk_returnValue[0] = 1;
            return;
        }

        // Fill-in the local table.
        const int NUM_STEPS = SMemSize / 32;
#pragma unroll
        for ( int step = 0, k = laneId ; step < NUM_STEPS ; ++step, k += 32 )
        {
            int bColIt = bColBeg + k;
            int bColId = -1;

            if ( bColIt < bColEnd )
            {
                bColId = B_col_indices[bColIt];
            }

            s_colInd[warpId][k] = bColId;
        }

        // Now load column indices of current row of A
        int aColIt  = A_row_offsets[aRowId];
        int aColEnd = A_row_offsets[aRowId + 1];

        for ( aColIt += laneId ; utils::any(aColIt < aColEnd) ; aColIt += 32 )
        {
            // The column.
            int aColId = -1;

            if ( aColIt < aColEnd )
            {
                aColId = A_col_indices[aColIt];
            }

            // Each thread searches for its column id, and gets the corresponding bColIt
            // TODO: Try binary search or using hash table
            int foundOffset = -1;

            if ( aColId == -1 )
            {
                foundOffset = -2;
            }

            for ( int i = 0 ; i < nCols && utils::any(foundOffset == -1) ; ++i )
                if ( foundOffset == -1 && s_colInd[warpId][i] == aColId )
                {
                    foundOffset = i;
                }

            // Store the result.
            if ( aColIt < aColEnd )
            {
                AtoBmapping[aColIt] = bColBeg + foundOffset;
            }
        }
    } // if RowId < A_nRows;
}

template< int CtaSize, int SMemSize>
__global__ __launch_bounds__( CtaSize )
void
computeAtoLUmappingExtDiag_kernel( int A_nRows,
                                   const int *__restrict A_row_offsets,
                                   const int *__restrict A_col_indices,
                                   const int *__restrict A_dia_indices,
                                   const int *__restrict B_row_offsets,
                                   const int *__restrict B_col_indices,
                                   int *__restrict AtoBmapping,
                                   int *wk_returnValue )
{
    const int nWarps = CtaSize / 32; // Number of warps per Cta
    const int warpId = utils::warp_id();
    const int laneId = utils::lane_id();
    // Rows are stored in SMEM. Linear storage.
    __shared__ volatile int s_colInd[nWarps][SMemSize];
    // The row this warp is responsible for
    int aRowId = blockIdx.x * nWarps + warpId;

    // Loop over rows of A.
    for ( ; aRowId < A_nRows ; aRowId += nWarps * gridDim.x )
    {
        // Insert all the column indices of matrix B in the shared memory table
        int bColBeg = B_row_offsets[aRowId];
        int bColEnd = B_row_offsets[aRowId + 1];
        // The number of columns.
        const int nCols = bColEnd - bColBeg;

        //TODO: Add fallback for cases where number of nonzeros exceed SMemSize
        if ( nCols > SMemSize )
        {
            wk_returnValue[0] = 1;
            return;
        }

        // Fill-in the local table.
        const int NUM_STEPS = SMemSize / 32;
#pragma unroll
        for ( int step = 0, k = laneId ; step < NUM_STEPS ; ++step, k += 32 )
        {
            int bColIt = bColBeg + k;
            int bColId = -1;

            if ( bColIt < bColEnd )
            {
                bColId = B_col_indices[bColIt];
            }

            s_colInd[warpId][k] = bColId;
        }

        // Now load column indices of current row of A
        int aColIt  = A_row_offsets[aRowId];
        int aColEnd = A_row_offsets[aRowId + 1];

        for ( aColIt += laneId ; utils::any(aColIt <= aColEnd) ; aColIt += 32 )
        {
            // The column.
            int aColId = -1;

            if ( aColIt < aColEnd )
            {
                aColId = A_col_indices[aColIt];
            }

            if ( aColIt == aColEnd )
            {
                aColId = aRowId;
            }

            // Each thread searches for its column id, and gets the corresponding bColIt
            // TODO: Try binary search or using hash table
            int foundOffset = -1;

            if ( aColId == -1 )
            {
                foundOffset = -2;
            }

            for ( int i = 0 ; i < nCols && utils::any(foundOffset == -1) ; ++i )
                if ( foundOffset == -1 && s_colInd[warpId][i] == aColId )
                {
                    foundOffset = i;
                }

            // Store the result.
            int aDst = -1;

            if ( aColIt < aColEnd )
            {
                aDst = aColIt;
            }

            if ( aColIt == aColEnd )
            {
                aDst = A_dia_indices[aRowId];
            }

            if ( aDst != -1 )
            {
                AtoBmapping[aDst] = bColBeg + foundOffset;
            }
        }
    }
}

#ifdef EXPERIMENTAL_LU_FACTORS

template< typename ValueTypeA, int CtaSize, int SMemSize, bool ROW_MAJOR >
__global__
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
__launch_bounds__( CtaSize, 12 )
#elif defined(__CUDA_ARCH__)
__launch_bounds__( CtaSize, 8 )
#endif
void
compute_LU_factors_4x4_kernel_warp( int A_nRows,
                                    const int *__restrict A_row_offsets,
                                    const int *__restrict A_col_indices,
                                    const int *__restrict A_dia_indices,
                                    ValueTypeA *__restrict A_nonzero_values,
                                    const int *__restrict A_smaller_color_offsets,
                                    const int *__restrict A_larger_color_offsets,
                                    const int *sorted_rows_by_color,
                                    const int num_rows_per_color,
                                    const int current_color,
                                    int *wk_returnValue )
{
    const int nWarps = CtaSize / 32; // Number of warps per Cta
    const int warpId = utils::warp_id();
    const int laneId = utils::lane_id();
    // Lane ID in the 2 16-wide segments.
    const int lane_id_div_16 = laneId / 16;
    const int lane_id_mod_16 = laneId % 16;
    // Coordinates inside a 4x4 block of the matrix.
    const int idx_i = lane_id_mod_16 / 4;
    const int idx_j = lane_id_mod_16 % 4;
    int globalWarpId = blockIdx.x * nWarps + warpId;
    // Shared memory to store the blocks to process
    __shared__ volatile ValueTypeA s_C_mtx[nWarps][32];
    __shared__ volatile ValueTypeA s_F_mtx[nWarps][16];
    // Shared memory to store the proposed column to load
#if __CUDA_ARCH__ < 300
    __shared__ volatile int s_aColItToLoad [nWarps][32];
    __shared__ volatile int s_waColItToLoad[nWarps][32];
#else
    __shared__ volatile int s_aColSrc[nWarps][32];
#endif
    // Shared memory to store the column indices of the current row
    __shared__ volatile int s_keys[nWarps][SMemSize];

    while (globalWarpId < num_rows_per_color)
    {
        int storedRowId[2];
        int I = 0;

        for (; I < 2 && globalWarpId < num_rows_per_color ; I++)
        {
            int aRowId = sorted_rows_by_color[globalWarpId];
            storedRowId[I] = aRowId;
            int aColBeg = A_row_offsets[aRowId + 0];
            int aColEnd = A_row_offsets[aRowId + 1];
            int aColSmaller = A_smaller_color_offsets[aRowId];
            // The number of columns.
            const int nCols = aColEnd - aColBeg;

            //TODO: Add fallback for cases where number of nonzeros exceed SMemSize
            if ( nCols > SMemSize )
            {
                wk_returnValue[0] = 1;
                return;
            }

            // Fill-in the local table.
            const int NUM_STEPS = SMemSize / 32;

#pragma unroll
            for ( int step = 0, k = laneId ; step < NUM_STEPS ; ++step, k += 32 )
            {
                int aColIt = aColBeg + k;
                int aColId = -1;

                if ( aColIt < aColEnd )
                {
                    aColId = A_col_indices[aColIt];
                }

                s_keys[warpId][k] = aColId;
            }

            // Now load all column indices of neighbours that have colors smaller than yours
            for ( int aColIt = aColBeg; aColIt < aColSmaller ; aColIt++)
            {
                unsigned int active_mask = utils::activemask();
                // Read the row to process, should be a broadcast
                int waRowId = s_keys[warpId][aColIt - aColBeg];
                // Compute multiplicative factor, load C_jj in first half, C_ij in second half
                int aColIdx = aColIt;

                if ( lane_id_div_16 == 0 )
                {
                    aColIdx = A_dia_indices[waRowId];
                }

                s_C_mtx[warpId][laneId] = A_nonzero_values[16 * aColIdx + lane_id_mod_16];
                // Threads 0-15 perform the matrix product
                ValueTypeA tmp(0);

                if (ROW_MAJOR)
                {

#pragma unroll
                    for ( int m = 0 ; m < 4 ; ++m )
                    {
                        tmp += s_C_mtx[warpId][16 + 4 * idx_i + m] * s_C_mtx[warpId][4 * m + idx_j];
                    }
                }
                else
                {

#pragma unroll
                    for ( int m = 0 ; m < 4 ; ++m )
                    {
                        tmp += s_C_mtx[warpId][16 + 4 * m + idx_j] * s_C_mtx[warpId][4 * idx_i + m];
                    }
                }

                if ( lane_id_div_16 == 0 )
                {
                    s_F_mtx[warpId][laneId] = tmp;
                    A_nonzero_values[16 * aColIt + laneId] = tmp;
                }

                int waColIt  = ld_cg(A_larger_color_offsets + waRowId);
                int waColEnd = ld_cg(A_row_offsets + waRowId + 1);

                // Load the first 32 columns of waRowId
                for (waColIt += laneId ; utils::any(waColIt < waColEnd, active_mask ); waColIt += 32 )
                {
                    // Each thread loads its column id
                    int waColId = -1;

                    if ( waColIt < waColEnd )
                    {
                        waColId = A_col_indices[waColIt];
                    }

                    // Find the right column.
                    int found_aColIt = -1;

#pragma unroll 4
                    for ( int i = 0, num_keys = aColEnd - aColBeg ; i < num_keys ; ++i )
                        if ( s_keys[warpId][i] == waColId )
                        {
                            found_aColIt = i;
                        }

                    if ( found_aColIt != -1 )
                    {
                        found_aColIt += aColBeg;
                    }

                    // Store all the columns that have been found
                    const int pred = found_aColIt != -1;
                    int vote = utils::ballot( pred, active_mask );
                    const int idst = __popc(vote & utils::lane_mask_lt());

                    if (pred)
                    {
#if __CUDA_ARCH__ < 300
                        s_aColItToLoad [warpId][idst] = found_aColIt;
                        s_waColItToLoad[warpId][idst] = waColIt;
#else
                        s_aColSrc[warpId][idst] = laneId;
#endif
                    }
                    utils::syncwarp(active_mask);

                    const int n_cols = __popc( vote );

                    // Process all columns that have been found
                    for ( int k = 0 ; k < n_cols ; k += 2 )
                    {
                        const int my_k = k + lane_id_div_16;
                        // Where to get columns from.
                        int a_col_it = -1, w_col_it = -1;
                        // Load column to load
#if __CUDA_ARCH__ < 300
                        if ( my_k < n_cols )
                        {
                            a_col_it = s_aColItToLoad [warpId][my_k];
                            w_col_it = s_waColItToLoad[warpId][my_k];
                        }
#else
                        a_col_it = utils::shfl(found_aColIt, s_aColSrc[warpId][my_k], warpSize, active_mask);
                        w_col_it = utils::shfl(waColIt,      s_aColSrc[warpId][my_k], warpSize, active_mask);

                        if ( my_k >= n_cols )
                        {
                            a_col_it = -1;
                            w_col_it = -1;
                        }

#endif
                        ValueTypeA my_C(0);

                        if ( w_col_it != -1 )
                        {
                            my_C = A_nonzero_values[16 * w_col_it + lane_id_mod_16];
                        }

                        s_C_mtx[warpId][laneId] = my_C;
                        // Run the matrix-matrix product.
                        ValueTypeA tmp(0);
                        utils::syncwarp( active_mask );

                        if (ROW_MAJOR)
                        {
#pragma unroll
                            for ( int m = 0 ; m < 4 ; ++m )
                            {
                                tmp += s_F_mtx[warpId][4 * idx_i + m] * s_C_mtx[warpId][16 * lane_id_div_16 + 4 * m + idx_j];
                            }
                        }
                        else
                        {
#pragma unroll
                            for ( int m = 0 ; m < 4 ; ++m )
                            {
                                tmp += s_F_mtx[warpId][4 * m + idx_j] * s_C_mtx[warpId][16 * lane_id_div_16 + 4 * idx_i + m];
                            }
                        }

                        if ( a_col_it != -1 )
                        {
                            A_nonzero_values[16 * a_col_it + lane_id_mod_16] -= tmp;
                        }
                    } // Loop over columns that have a match (for k=0;k<n_cols)
                } // Loop over the columns of waRowId

                //}  // Loop j=0;j<32
            } // Loop over the columns of aRowId

            globalWarpId += nWarps * gridDim.x;
        } // end of loop over I

        // Now compute the inverse of the block C_jj
        if ( lane_id_div_16 == 0 || I == 2 )
        {
            const int offset = 16 * A_dia_indices[storedRowId[lane_id_div_16]] + lane_id_mod_16;
            s_C_mtx[warpId][laneId] = A_nonzero_values[offset];
            utils::syncwarp(utils::activemask());

            if (ROW_MAJOR)
            {
                compute_block_inverse_row_major4x4_formula2<int, ValueTypeA, 4, true>( s_C_mtx[warpId], 16 * lane_id_div_16, offset, idx_i, idx_j, A_nonzero_values );
            }
            else
            {
                compute_block_inverse_col_major4x4_formula2<int, ValueTypeA, 4, true>( s_C_mtx[warpId], 16 * lane_id_div_16, offset, idx_i, idx_j, A_nonzero_values );
            }
        } // End of if statement
    } // End of while loop
}

#else

template< typename ValueTypeA, int CtaSize, int SMemSize, bool ROW_MAJOR>
__global__ __launch_bounds__( CtaSize )
void
computeLUFactors_4x4_kernel( int A_nRows,
                             const int *__restrict A_row_offsets,
                             const int *__restrict A_col_indices,
                             const int *__restrict A_dia_indices,
                             ValueTypeA *__restrict A_nonzero_values,
                             const int *__restrict A_smaller_color_offsets,
                             const int *__restrict A_larger_color_offsets,
                             const int *sorted_rows_by_color,
                             const int num_rows_per_color,
                             const int current_color,
                             int *wk_returnValue )
{
    const int nWarps = CtaSize / 32; // Number of warps per Cta
    const int warpId = utils::warp_id();
    const int laneId = utils::lane_id();
    int lane_mask_lt = utils::lane_mask_lt();

    // Lane ID in the 2 16-wide segments.
    const int lane_id_div_16 = laneId / 16;
    const int lane_id_mod_16 = laneId % 16;
    // Coordinates inside a 4x4 block of the matrix.
    const int idx_i = lane_id_mod_16 / 4;
    const int idx_j = lane_id_mod_16 % 4;
    int globalWarpId = blockIdx.x * nWarps + warpId;
    // Shared memory to store the blocks to process
    __shared__ volatile ValueTypeA s_C_mtx[nWarps][32];
    // Shared memory to store the proposed column to load
    __shared__ volatile int s_aColItToLoad[nWarps][32];
    __shared__ volatile int s_waColItToLoad[nWarps][32];
    // Shared memory to store the proposed column to load
    __shared__ volatile unsigned s_aColIds[nWarps][32];
    // The size of the hash table (one per warp - shared memory).
    __shared__ volatile int s_size[nWarps][2];
    // Shared memory to store the column indices of the current row
    __shared__ volatile int s_keys[nWarps][SMemSize];

    while (globalWarpId < num_rows_per_color)
    {
        int aRowId = sorted_rows_by_color[globalWarpId];
        // Insert all the column indices in shared memory
        // TODO: Use texture here
        int aColBeg = A_row_offsets[aRowId];
        int aColEnd = A_row_offsets[aRowId + 1];
        int aColIt  = aColBeg;

        // Check if number of nonzeros will fit in shared memory
        if ( (aColEnd - aColBeg) > SMemSize )
        {
            wk_returnValue[0] = 1;
            return;
        }

        // Load the all the column indices of row into shared memory
        for ( aColIt += laneId ; utils::any( aColIt < aColEnd ) ; aColIt += 32 )
        {
            int aColId = aColIt < aColEnd ? (int) A_col_indices[aColIt] : -1;
            s_keys[warpId][aColIt - aColBeg] = aColId;
        }

        // Now load all column indices of neighbours that have colors smaller than yours
        aColIt  = aColBeg;
        int aColSmaller = A_smaller_color_offsets[aRowId];

        for ( ; utils::any( (aColIt + laneId) < aColSmaller ) ; aColIt += 32 )
        {
            int aColId = (aColIt + laneId) < aColSmaller ? (int) A_col_indices[aColIt + laneId] : -1;
            // Each thread pushes its column
            s_aColIds[warpId][laneId] = aColId;

            // Have warp collaborate to load each row
            for ( int j = 0; j < 32; j++)
            {
                // Check if row to load is valid
                if ( ( aColIt + j ) >= aColSmaller ) { break; }

                // Read the row to process, should be a broadcast
                int waRowId = s_aColIds[warpId][j];

                // Compute multiplicative factor, load C_jj in first half, C_ij in second half
                if (lane_id_div_16 == 0)
                {
                    s_C_mtx[warpId][laneId] = A_nonzero_values[ 16 * A_dia_indices[waRowId] + lane_id_mod_16 ];
                }
                else
                {
                    s_C_mtx[warpId][laneId] = A_nonzero_values[ 16 * (aColIt + j) + lane_id_mod_16 ];
                }

                // Threads 0-15 perform the matrix product
                utils::syncwarp();
                if (lane_id_div_16 == 0)
                {
                    ValueTypeA tmp(0);

                    if (ROW_MAJOR)
                    {
#pragma unroll
                        for ( int m = 0 ; m < 4 ; ++m )
                        {
                            tmp += s_C_mtx[warpId][16 + 4 * idx_i + m] * s_C_mtx[warpId][4 * m + idx_j];
                        }
                    }
                    else
                    {
#pragma unroll
                        for ( int m = 0 ; m < 4 ; ++m )
                        {
                            tmp += s_C_mtx[warpId][16 + 4 * m + idx_j] * s_C_mtx[warpId][4 * idx_i + m];
                        }
                    }

                    s_C_mtx[warpId][laneId] = tmp;
                    A_nonzero_values[16 * (aColIt + j) + laneId] = tmp;
                }

                int waColIt  = A_larger_color_offsets[waRowId];
                int waColEnd = A_row_offsets[waRowId + 1];

                //// Load the first 32 columns of waRowId
                for (waColIt += laneId ; utils::any(waColIt < waColEnd ); waColIt += 32 )
                {
                    // Each thread loads its column id
                    int waColId = waColIt < waColEnd ? A_col_indices[waColIt] : int (-1);
                    // TODO: Try binary search if columns are ordered
                    int found_aColIt = -1;

                    //TODO: if invalid waColId, don't search
                    for (int i = 0 ; utils::any(found_aColIt == -1) && i < aColEnd - aColBeg ; i++)
                    {
                        if (s_keys[warpId][i] == waColId) { found_aColIt = aColBeg + i; }
                    }

                    // Store all the columns that have been found
                    const int pred = found_aColIt != -1;
                    const int vote = utils::ballot( pred );
                    const int idst = __popc(vote & lane_mask_lt);

                    if (pred)
                    {
                        s_aColItToLoad [warpId][idst] = found_aColIt;
                        s_waColItToLoad[warpId][idst] = waColIt;
                    }

                    const int n_cols = __popc( vote );

                    // Process all columns that have been found
                    for ( int k = 0 ; k < n_cols ; k++ )
                    {
                        // Load column to load
                        const int a_col_it = k < n_cols ? s_aColItToLoad [warpId][k] : -1;
                        const int w_col_it = k < n_cols ? s_waColItToLoad[warpId][k] : -1;

                        if (lane_id_div_16 == 1)
                        {
                            s_C_mtx[warpId][laneId] = A_nonzero_values[16 * w_col_it + lane_id_mod_16];
                            // Run the matrix-matrix product.
                            ValueTypeA tmp(0);
                            utils::syncwarp(utils::activemask());

                            if (ROW_MAJOR)
                            {

#pragma unroll
                                for ( int m = 0 ; m < 4 ; ++m )
                                {
                                    tmp += s_C_mtx[warpId][4 * idx_i + m] * s_C_mtx[warpId][16 + 4 * m + idx_j];
                                }
                            }
                            else
                            {

#pragma unroll
                                for ( int m = 0 ; m < 4 ; ++m )
                                {
                                    tmp += s_C_mtx[warpId][4 * m + idx_j] * s_C_mtx[warpId][16 + 4 * idx_i + m];
                                }
                            }

                            A_nonzero_values[16 * a_col_it + lane_id_mod_16] -= tmp;
                        }
                    } // Loop over columns that have a match (for k=0;k<n_cols)
                } // Loop over the columns of waRowId
            }  // Loop j=0;j<32
        } // Loop over the columns of aRowId

        // TODO: Have one warp deal with two rows
        // Now compute the inverse of the block C_jj
        if (lane_id_div_16 == 0)
        {
            const int offset = 16 * A_dia_indices[aRowId] + lane_id_mod_16;
            s_C_mtx[warpId][laneId] = A_nonzero_values[offset];
            utils::syncwarp(utils::activemask());

            if (ROW_MAJOR)
            {
                compute_block_inverse_row_major<int, ValueTypeA, 0, 4, 16>
                (s_C_mtx[warpId], 0, offset, idx_i, idx_j, A_nonzero_values);
            }
            else
            {
                compute_block_inverse_col_major<int, ValueTypeA, 0, 4, 16>
                (s_C_mtx[warpId], 0, offset, idx_i, idx_j, A_nonzero_values);
            }
        }

        globalWarpId += nWarps * gridDim.x;
    } // if RowId < A_nRows;
}
#endif
#ifdef EXPERIMENTAL_LU_FACTORS



#ifdef EXPERIMENTAL_LU_FACTORS

template< typename ValueTypeA, int CtaSize, int SMemSize, bool ROW_MAJOR >
__global__
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
__launch_bounds__( CtaSize, 12 )
#elif defined(__CUDA_ARCH__)
__launch_bounds__( CtaSize, 8 )
#endif
void
compute_LU_factors_1x1_kernel_warp( int A_nRows,
                                    const int *__restrict A_row_offsets,
                                    const int *__restrict A_col_indices,
                                    const int *__restrict A_dia_indices,
                                    ValueTypeA *__restrict A_nonzero_values,
                                    const int *__restrict A_smaller_color_offsets,
                                    const int *__restrict A_larger_color_offsets,
                                    const int *sorted_rows_by_color,
                                    const int num_rows_per_color,
                                    const int current_color,
                                    int *wk_returnValue )
{
    const int nWarps = CtaSize / 32; // Number of warps per Cta
    const int warpId = utils::warp_id();
    const int laneId = utils::lane_id();
    // Lane ID in the 3 9-wide segments.
    const int lane_id_div_1 = laneId / 1;
    const int lane_id_mod_1 = laneId % 1;
    // Coordinates inside a 3x3 block of the matrix.
    const int idx_i = lane_id_mod_1 / 1;
    const int idx_j = lane_id_mod_1 % 1;
    int globalWarpId = blockIdx.x * nWarps + warpId;
    // Shared memory to store the blocks to process
    __shared__ volatile ValueTypeA s_C_mtx[nWarps][32];
    __shared__ volatile ValueTypeA s_F_mtx[nWarps][16];
    // Shared memory to store the proposed column to load
#if __CUDA_ARCH__ < 300
    __shared__ volatile int s_aColItToLoad [nWarps][32];
    __shared__ volatile int s_waColItToLoad[nWarps][32];
#else
    __shared__ volatile int s_aColSrc[nWarps][32];
#endif
    // Shared memory to store the column indices of the current row
    __shared__ volatile int s_keys[nWarps][SMemSize];

    while (globalWarpId < num_rows_per_color)
    {
        int storedRowId[2];
        int I = 0;

        for (; I < 2 && globalWarpId < num_rows_per_color ; I++)
        {
            int aRowId = sorted_rows_by_color[globalWarpId];
            storedRowId[I] = aRowId;
            int aColBeg = A_row_offsets[aRowId + 0];
            int aColEnd = A_row_offsets[aRowId + 1];
            int aColSmaller = A_smaller_color_offsets[aRowId];
            // The number of columns.
            const int nCols = aColEnd - aColBeg;

            //TODO: Add fallback for cases where number of nonzeros exceed SMemSize
            if ( nCols > SMemSize )
            {
                wk_returnValue[0] = 1;
                return;
            }

            // Fill-in the local table.
            const int NUM_STEPS = SMemSize / 32;

#pragma unroll
            for ( int step = 0, k = laneId ; step < NUM_STEPS ; ++step, k += 32 )
            {
                int aColIt = aColBeg + k;
                int aColId = -1;

                if ( aColIt < aColEnd )
                {
                    aColId = A_col_indices[aColIt];
                }

                s_keys[warpId][k] = aColId;
            }

            // Now load all column indices of neighbours that have colors smaller than yours
            for ( int aColIt = aColBeg; aColIt < aColSmaller && lane_id_div_1 < 2; aColIt++)
            {
                unsigned int active_mask = utils::activemask();
                // Read the row to process, should be a broadcast
                int waRowId = s_keys[warpId][aColIt - aColBeg];
                // Compute multiplicative factor, load C_jj in first half, C_ij in second half
                int aColIdx = aColIt;


                if ( lane_id_div_1 == 0 )
                {
                    aColIdx = A_dia_indices[waRowId];
                }


                s_C_mtx[warpId][laneId] = A_nonzero_values[1 * aColIdx + lane_id_mod_1];

                // Threads 0-8 perform the matrix product
                ValueTypeA tmp(0);


                if (ROW_MAJOR)
                {

#pragma unroll
                    for ( int m = 0 ; m < 1 ; ++m )
                    {
                        tmp += s_C_mtx[warpId][1 + 1 * idx_i + m] * s_C_mtx[warpId][1 * m + idx_j];
                    }
                }
                else
                {

// #pragma unroll
//                         for ( int m = 0 ; m < 3 ; ++m )
//                         {
//                             tmp += s_C_mtx[warpId][9 + 3 * m + idx_j] * s_C_mtx[warpId][3 * idx_i + m];
//                         }
                }


            utils::syncwarp(active_mask);

                if ( lane_id_div_1 == 0 )
                {
                    s_F_mtx[warpId][laneId] = tmp;
                    A_nonzero_values[1 * aColIt + laneId] = tmp;
                }

                int waColIt  = ld_cg(A_larger_color_offsets + waRowId);
                int waColEnd = ld_cg(A_row_offsets + waRowId + 1);

                // Load the first 32 columns of waRowId
                for (waColIt += laneId ; utils::any(waColIt < waColEnd, active_mask ); waColIt += 2 )
                {
                    // Each thread loads its column id
                    int waColId = -1;

                    if ( waColIt < waColEnd )
                    {
                        waColId = A_col_indices[waColIt];
                    }

                    // Find the right column.
                    int found_aColIt = -1;

#pragma unroll 4
                    for ( int i = 0, num_keys = aColEnd - aColBeg ; i < num_keys ; ++i )
                        if ( s_keys[warpId][i] == waColId )
                        {
                            found_aColIt = i;
                        }

                    if ( found_aColIt != -1 )
                    {
                        found_aColIt += aColBeg;
                    }

                    // Store all the columns that have been found
                    const int pred = found_aColIt != -1;
                    int vote = utils::ballot( pred, active_mask );
                    const int idst = __popc(vote & utils::lane_mask_lt());

                    if (pred)
                    {
#if __CUDA_ARCH__ < 300
                        s_aColItToLoad [warpId][idst] = found_aColIt;
                        s_waColItToLoad[warpId][idst] = waColIt;
#else
                        s_aColSrc[warpId][idst] = laneId;
#endif
                    }
                    utils::syncwarp(active_mask);

                    const int n_cols = __popc( vote );

                    // Process all columns that have been found
                    for ( int k = 0 ; k < n_cols ; k += 2 )
                    {
                        const int my_k = k + lane_id_div_1;
                        // Where to get columns from.
                        int a_col_it = -1, w_col_it = -1;
                        // Load column to load
#if __CUDA_ARCH__ < 300
                        if ( my_k < n_cols )
                        {
                            a_col_it = s_aColItToLoad [warpId][my_k];
                            w_col_it = s_waColItToLoad[warpId][my_k];
                        }
#else
                        a_col_it = utils::shfl(found_aColIt, s_aColSrc[warpId][my_k], warpSize, active_mask);
                        w_col_it = utils::shfl(waColIt,      s_aColSrc[warpId][my_k], warpSize, active_mask);

                        if ( my_k >= n_cols )
                        {
                            a_col_it = -1;
                            w_col_it = -1;
                        }

#endif
                        ValueTypeA my_C(0);

                        if (( w_col_it != -1 ) && ( lane_id_div_1 < 2 ))
                        {
                            my_C = A_nonzero_values[1 * w_col_it + lane_id_mod_1];
                        }

                        if ( lane_id_div_1 < 2 )
                             s_C_mtx[warpId][laneId] = my_C;
                        // Run the matrix-matrix product.
                        ValueTypeA tmp(0);
                        utils::syncwarp( active_mask );

                        if (ROW_MAJOR)
                        {
#pragma unroll
                            for ( int m = 0 ; m < 1 ; ++m )
                            {
                                tmp += s_F_mtx[warpId][1 * idx_i + m] * s_C_mtx[warpId][1 * lane_id_div_1 + 1 * m + idx_j];
                            }
                        }
                        else
                        {
// #pragma unroll
//                             for ( int m = 0 ; m < 3 ; ++m )
//                             {
//                                 tmp += s_F_mtx[warpId][3 * m + idx_j] * s_C_mtx[warpId][9 * lane_id_div_1 + 3 * idx_i + m];
//                             }
                        }

                        if ( a_col_it != -1 )
                        {
                            A_nonzero_values[1 * a_col_it + lane_id_mod_1] -= tmp;
                        }
                    } // Loop over columns that have a match (for k=0;k<n_cols)
                } // Loop over the columns of waRowId

                //}  // Loop j=0;j<32
            } // Loop over the columns of aRowId

            globalWarpId += nWarps * gridDim.x;
        } // end of loop over I


        // Now compute the inverse of the block C_jj
        if ( lane_id_div_1 == 0 || I == 2 )
        {
            const int offset = 1 * A_dia_indices[storedRowId[lane_id_div_1]] + lane_id_mod_1;
            s_C_mtx[warpId][laneId] = A_nonzero_values[offset];
            utils::syncwarp(utils::activemask());
            double a =  s_C_mtx[warpId][laneId];
            if ( isNotCloseToZero(a))
                a = 1.0 / a;
            else 
                a = 1.0 / epsilon(a);
            
            A_nonzero_values[offset] = a;
            // if (ROW_MAJOR)
            // {
            //     compute_block_inverse_row_major2x2_formula2<int, ValueTypeA, 3, true>( s_C_mtx[warpId], 9 * lane_id_div_1, offset, idx_i, idx_j, A_nonzero_values );
            //     s_C_mtx[warpId][laneId] = a;
            // }
            // else
            // {
            //     //#TODO compute_block_inverse_row_major3x3_formula2
            // }
        } // End of if statement
    } // End of while loop
}

#else
    FatalError("Haven't implemented compute_LU_factors_3x3 for Not EXPERIMENTAL_LU_FACTORS", AMGX_ERR_NOT_SUPPORTED_TARGET);
#endif

#ifdef EXPERIMENTAL_LU_FACTORS

template< typename ValueTypeA, int CtaSize, int SMemSize, bool ROW_MAJOR >
__global__
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
__launch_bounds__( CtaSize, 12 )
#elif defined(__CUDA_ARCH__)
__launch_bounds__( CtaSize, 8 )
#endif
void
compute_LU_factors_2x2_kernel_warp( int A_nRows,
                                    const int *__restrict A_row_offsets,
                                    const int *__restrict A_col_indices,
                                    const int *__restrict A_dia_indices,
                                    ValueTypeA *__restrict A_nonzero_values,
                                    const int *__restrict A_smaller_color_offsets,
                                    const int *__restrict A_larger_color_offsets,
                                    const int *sorted_rows_by_color,
                                    const int num_rows_per_color,
                                    const int current_color,
                                    int *wk_returnValue )
{
    const int nWarps = CtaSize / 32; // Number of warps per Cta
    const int warpId = utils::warp_id();
    const int laneId = utils::lane_id();
    // Lane ID in the 3 9-wide segments.
    const int lane_id_div_4 = laneId / 4;
    const int lane_id_mod_4 = laneId % 4;
    // Coordinates inside a 3x3 block of the matrix.
    const int idx_i = lane_id_mod_4 / 2;
    const int idx_j = lane_id_mod_4 % 2;
    int globalWarpId = blockIdx.x * nWarps + warpId;
    // Shared memory to store the blocks to process
    __shared__ volatile ValueTypeA s_C_mtx[nWarps][32];
    __shared__ volatile ValueTypeA s_F_mtx[nWarps][16];
    // Shared memory to store the proposed column to load
#if __CUDA_ARCH__ < 300
    __shared__ volatile int s_aColItToLoad [nWarps][32];
    __shared__ volatile int s_waColItToLoad[nWarps][32];
#else
    __shared__ volatile int s_aColSrc[nWarps][32];
#endif
    // Shared memory to store the column indices of the current row
    __shared__ volatile int s_keys[nWarps][SMemSize];

    while (globalWarpId < num_rows_per_color)
    {
        int storedRowId[2];
        int I = 0;

        for (; I < 2 && globalWarpId < num_rows_per_color ; I++)
        {
            int aRowId = sorted_rows_by_color[globalWarpId];
            storedRowId[I] = aRowId;
            int aColBeg = A_row_offsets[aRowId + 0];
            int aColEnd = A_row_offsets[aRowId + 1];
            int aColSmaller = A_smaller_color_offsets[aRowId];
            // The number of columns.
            const int nCols = aColEnd - aColBeg;

            //TODO: Add fallback for cases where number of nonzeros exceed SMemSize
            if ( nCols > SMemSize )
            {
                wk_returnValue[0] = 1;
                return;
            }

            // Fill-in the local table.
            const int NUM_STEPS = SMemSize / 32;

#pragma unroll
            for ( int step = 0, k = laneId ; step < NUM_STEPS ; ++step, k += 32 )
            {
                int aColIt = aColBeg + k;
                int aColId = -1;

                if ( aColIt < aColEnd )
                {
                    aColId = A_col_indices[aColIt];
                }

                s_keys[warpId][k] = aColId;
            }

            // Now load all column indices of neighbours that have colors smaller than yours
            for ( int aColIt = aColBeg; aColIt < aColSmaller && lane_id_div_4 < 2; aColIt++)
            {
                unsigned int active_mask = utils::activemask();
                // Read the row to process, should be a broadcast
                int waRowId = s_keys[warpId][aColIt - aColBeg];
                // Compute multiplicative factor, load C_jj in first half, C_ij in second half
                int aColIdx = aColIt;


                if ( lane_id_div_4 == 0 )
                {
                    aColIdx = A_dia_indices[waRowId];
                }


                s_C_mtx[warpId][laneId] = A_nonzero_values[4 * aColIdx + lane_id_mod_4];

                // Threads 0-8 perform the matrix product
                ValueTypeA tmp(0);


                if (ROW_MAJOR)
                {

#pragma unroll
                    for ( int m = 0 ; m < 2 ; ++m )
                    {
                        tmp += s_C_mtx[warpId][4 + 2 * idx_i + m] * s_C_mtx[warpId][2 * m + idx_j];
                    }
                }
                else
                {

// #pragma unroll
//                         for ( int m = 0 ; m < 3 ; ++m )
//                         {
//                             tmp += s_C_mtx[warpId][9 + 3 * m + idx_j] * s_C_mtx[warpId][3 * idx_i + m];
//                         }
                }


            utils::syncwarp(active_mask);

                if ( lane_id_div_4 == 0 )
                {
                    s_F_mtx[warpId][laneId] = tmp;
                    A_nonzero_values[4 * aColIt + laneId] = tmp;
                }

                int waColIt  = ld_cg(A_larger_color_offsets + waRowId);
                int waColEnd = ld_cg(A_row_offsets + waRowId + 1);

                // Load the first 32 columns of waRowId
                for (waColIt += laneId ; utils::any(waColIt < waColEnd, active_mask ); waColIt += 8 )
                {
                    // Each thread loads its column id
                    int waColId = -1;

                    if ( waColIt < waColEnd )
                    {
                        waColId = A_col_indices[waColIt];
                    }

                    // Find the right column.
                    int found_aColIt = -1;

#pragma unroll 4
                    for ( int i = 0, num_keys = aColEnd - aColBeg ; i < num_keys ; ++i )
                        if ( s_keys[warpId][i] == waColId )
                        {
                            found_aColIt = i;
                        }

                    if ( found_aColIt != -1 )
                    {
                        found_aColIt += aColBeg;
                    }

                    // Store all the columns that have been found
                    const int pred = found_aColIt != -1;
                    int vote = utils::ballot( pred, active_mask );
                    const int idst = __popc(vote & utils::lane_mask_lt());

                    if (pred)
                    {
#if __CUDA_ARCH__ < 300
                        s_aColItToLoad [warpId][idst] = found_aColIt;
                        s_waColItToLoad[warpId][idst] = waColIt;
#else
                        s_aColSrc[warpId][idst] = laneId;
#endif
                    }
                    utils::syncwarp(active_mask);

                    const int n_cols = __popc( vote );

                    // Process all columns that have been found
                    for ( int k = 0 ; k < n_cols ; k += 2 )
                    {
                        const int my_k = k + lane_id_div_4;
                        // Where to get columns from.
                        int a_col_it = -1, w_col_it = -1;
                        // Load column to load
#if __CUDA_ARCH__ < 300
                        if ( my_k < n_cols )
                        {
                            a_col_it = s_aColItToLoad [warpId][my_k];
                            w_col_it = s_waColItToLoad[warpId][my_k];
                        }
#else
                        a_col_it = utils::shfl(found_aColIt, s_aColSrc[warpId][my_k], warpSize, active_mask);
                        w_col_it = utils::shfl(waColIt,      s_aColSrc[warpId][my_k], warpSize, active_mask);

                        if ( my_k >= n_cols )
                        {
                            a_col_it = -1;
                            w_col_it = -1;
                        }

#endif
                        ValueTypeA my_C(0);

                        if (( w_col_it != -1 ) && ( lane_id_div_4 < 2 ))
                        {
                            my_C = A_nonzero_values[4 * w_col_it + lane_id_mod_4];
                        }

                        if ( lane_id_div_4 < 2 )
                             s_C_mtx[warpId][laneId] = my_C;
                        // Run the matrix-matrix product.
                        ValueTypeA tmp(0);
                        utils::syncwarp( active_mask );

                        if (ROW_MAJOR)
                        {
#pragma unroll
                            for ( int m = 0 ; m < 2 ; ++m )
                            {
                                tmp += s_F_mtx[warpId][2 * idx_i + m] * s_C_mtx[warpId][4 * lane_id_div_4 + 2 * m + idx_j];
                            }
                        }
                        else
                        {
// #pragma unroll
//                             for ( int m = 0 ; m < 3 ; ++m )
//                             {
//                                 tmp += s_F_mtx[warpId][3 * m + idx_j] * s_C_mtx[warpId][9 * lane_id_div_4 + 3 * idx_i + m];
//                             }
                        }

                        if ( a_col_it != -1 )
                        {
                            A_nonzero_values[4 * a_col_it + lane_id_mod_4] -= tmp;
                        }
                    } // Loop over columns that have a match (for k=0;k<n_cols)
                } // Loop over the columns of waRowId

                //}  // Loop j=0;j<32
            } // Loop over the columns of aRowId

            globalWarpId += nWarps * gridDim.x;
        } // end of loop over I


        // Now compute the inverse of the block C_jj
        if ( lane_id_div_4 == 0 || I == 2 )
        {
            const int offset = 4 * A_dia_indices[storedRowId[lane_id_div_4]] + lane_id_mod_4;
            s_C_mtx[warpId][laneId] = A_nonzero_values[offset];
            utils::syncwarp(utils::activemask());

            if (ROW_MAJOR)
            {
                compute_block_inverse_row_major2x2_formula2<int, ValueTypeA, 2, true>( s_C_mtx[warpId], 4 * lane_id_div_4, offset, idx_i, idx_j, A_nonzero_values );
            }
            else
            {
                //#TODO compute_block_inverse_row_major3x3_formula2
            }
        } // End of if statement
    } // End of while loop
}

#else
    FatalError("Haven't implemented compute_LU_factors_3x3 for Not EXPERIMENTAL_LU_FACTORS", AMGX_ERR_NOT_SUPPORTED_TARGET);
#endif  

template< typename ValueTypeA, int CtaSize, int SMemSize, bool ROW_MAJOR >
__global__
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
__launch_bounds__( CtaSize, 12 )
#elif defined(__CUDA_ARCH__)
__launch_bounds__( CtaSize, 8 )
#endif
void
compute_LU_factors_3x3_kernel_warp( int A_nRows,
                                    const int *__restrict A_row_offsets,
                                    const int *__restrict A_col_indices,
                                    const int *__restrict A_dia_indices,
                                    ValueTypeA *__restrict A_nonzero_values,
                                    const int *__restrict A_smaller_color_offsets,
                                    const int *__restrict A_larger_color_offsets,
                                    const int *sorted_rows_by_color,
                                    const int num_rows_per_color,
                                    const int current_color,
                                    int *wk_returnValue )
{
    const int nWarps = CtaSize / 32; // Number of warps per Cta
    const int warpId = utils::warp_id();
    const int laneId = utils::lane_id();
    // Lane ID in the 3 9-wide segments.
    const int lane_id_div_9 = laneId / 9;
    const int lane_id_mod_9 = laneId % 9;
    // Coordinates inside a 3x3 block of the matrix.
    const int idx_i = lane_id_mod_9 / 3;
    const int idx_j = lane_id_mod_9 % 3;
    int globalWarpId = blockIdx.x * nWarps + warpId;
    // Shared memory to store the blocks to process
    __shared__ volatile ValueTypeA s_C_mtx[nWarps][32];
    __shared__ volatile ValueTypeA s_F_mtx[nWarps][16];
    // Shared memory to store the proposed column to load
#if __CUDA_ARCH__ < 300
    __shared__ volatile int s_aColItToLoad [nWarps][32];
    __shared__ volatile int s_waColItToLoad[nWarps][32];
#else
    __shared__ volatile int s_aColSrc[nWarps][32];
#endif
    // Shared memory to store the column indices of the current row
    __shared__ volatile int s_keys[nWarps][SMemSize];
    bool exShare = false;

    while (globalWarpId < num_rows_per_color)
    {
        int storedRowId[2];
        int I = 0;

        for (; I < 2 && globalWarpId < num_rows_per_color ; I++)
        {
            int aRowId = sorted_rows_by_color[globalWarpId];
            storedRowId[I] = aRowId;
            int aColBeg = A_row_offsets[aRowId + 0];
            int aColEnd = A_row_offsets[aRowId + 1];
            int aColSmaller = A_smaller_color_offsets[aRowId];
            // The number of columns.
            const int nCols = aColEnd - aColBeg;

            if ( nCols > SMemSize )
            {
                exShare = true;
                // wk_returnValue[0] = 1;
                // return;
            }

            // Fill-in the local table.
            const int NUM_STEPS = SMemSize / 32;

#pragma unroll
            for ( int step = 0, k = laneId ; step < NUM_STEPS ; ++step, k += 32 )
            {
                int aColIt = aColBeg + k;
                int aColId = -1;

                if ( aColIt < aColEnd )
                {
                    aColId = A_col_indices[aColIt];                    
                }

                s_keys[warpId][k] = aColId;
            }

            // Now load all column indices of neighbours that have colors smaller than yours
            for ( int aColIt = aColBeg; aColIt < aColSmaller; aColIt++)
            {
                unsigned int active_mask = utils::activemask();
                // Read the row to process, should be a broadcast
                int waRowId = s_keys[warpId][aColIt - aColBeg];
                // Compute multiplicative factor, load C_jj in first half, C_ij in second half
                int aColIdx = aColIt;
                

                if ( lane_id_div_9 == 0 )
                {
                    aColIdx = A_dia_indices[waRowId];
                }

                if( lane_id_div_9 < 2 )
                    s_C_mtx[warpId][laneId] = A_nonzero_values[9 * aColIdx + lane_id_mod_9];

                // Threads 0-8 perform the matrix product
                ValueTypeA tmp(0);
                

                if (ROW_MAJOR)
                {

#pragma unroll
                    for ( int m = 0 ; m < 3 && lane_id_div_9 < 2; ++m )
                    {
                        tmp += s_C_mtx[warpId][9 + 3 * idx_i + m] * s_C_mtx[warpId][3 * m + idx_j];
                    }
                }
                else
                {

// #pragma unroll
//                         for ( int m = 0 ; m < 3 ; ++m )
//                         {
//                             tmp += s_C_mtx[warpId][9 + 3 * m + idx_j] * s_C_mtx[warpId][3 * idx_i + m];
//                         }
                }


            utils::syncwarp(active_mask);

                if ( lane_id_div_9 == 0 )
                {
                    s_F_mtx[warpId][laneId] = tmp;
                    A_nonzero_values[9 * aColIt + laneId] = tmp;
                }

                int waColIt  = ld_cg(A_larger_color_offsets + waRowId);
                int waColEnd = ld_cg(A_row_offsets + waRowId + 1);

                // Load the first 32 columns of waRowId
                for (waColIt += laneId ; utils::any(waColIt < waColEnd, active_mask ); waColIt += 32 )
                {
                    // Each thread loads its column id
                    int waColId = -1;

                    if ( waColIt < waColEnd )
                    {
                        waColId = A_col_indices[waColIt];
                    }

                    // Find the right column.
                    int found_aColIt = -1;

#pragma unroll 4
                    for ( int i = 0, num_keys = aColEnd - aColBeg ; i < num_keys ; ++i )
                        if ( exShare == false )
                        {
                             if ( s_keys[warpId][i] == waColId )
                                found_aColIt = i;
                        }
                        else
                        {
                             if (  A_col_indices[aColBeg + i] == waColId )
                                found_aColIt = i;
                        }
                        

                    if ( found_aColIt != -1 )
                    {
                        found_aColIt += aColBeg;
                    }

                    // Store all the columns that have been found
                    const int pred = found_aColIt != -1;
                    int vote = utils::ballot( pred, active_mask );
                    const int idst = __popc(vote & utils::lane_mask_lt());

                    if (pred)
                    {
#if __CUDA_ARCH__ < 300
                        s_aColItToLoad [warpId][idst] = found_aColIt;
                        s_waColItToLoad[warpId][idst] = waColIt;
#else
                        s_aColSrc[warpId][idst] = laneId;
#endif
                    }
                    utils::syncwarp(active_mask);

                    const int n_cols = __popc( vote );

                    // Process all columns that have been found
                    for ( int k = 0 ; k < n_cols &&  lane_id_div_9 < 2; k += 2 )
                    {
                        unsigned int active_mask_2 = utils::activemask();
                        const int my_k = k + lane_id_div_9;
                        // Where to get columns from.
                        int a_col_it = -1, w_col_it = -1;
                        // Load column to load
#if __CUDA_ARCH__ < 300
                        if ( my_k < n_cols )
                        {
                            a_col_it = s_aColItToLoad [warpId][my_k];
                            w_col_it = s_waColItToLoad[warpId][my_k];
                        }
#else
                        a_col_it = utils::shfl(found_aColIt, s_aColSrc[warpId][my_k], warpSize, active_mask_2);
                        w_col_it = utils::shfl(waColIt,      s_aColSrc[warpId][my_k], warpSize, active_mask_2);

                        if ( my_k >= n_cols )
                        {
                            a_col_it = -1;
                            w_col_it = -1;
                        }

#endif
                        ValueTypeA my_C(0);

                        if (( w_col_it != -1 ) && ( lane_id_div_9 < 2 ))
                        {
                            my_C = A_nonzero_values[9 * w_col_it + lane_id_mod_9];
                        }

                        if ( lane_id_div_9 < 2 )
                             s_C_mtx[warpId][laneId] = my_C;
                        // Run the matrix-matrix product.
                        ValueTypeA tmp(0);
                        utils::syncwarp( active_mask_2 );

                        if (ROW_MAJOR)
                        {
#pragma unroll
                            for ( int m = 0 ; m < 3 ; ++m )
                            {
                                tmp += s_F_mtx[warpId][3 * idx_i + m] * s_C_mtx[warpId][9 * lane_id_div_9 + 3 * m + idx_j];
                            }
                        }
                        else
                        {
// #pragma unroll
//                             for ( int m = 0 ; m < 3 ; ++m )
//                             {
//                                 tmp += s_F_mtx[warpId][3 * m + idx_j] * s_C_mtx[warpId][9 * lane_id_div_9 + 3 * idx_i + m];
//                             }
                        }

                        if ( a_col_it != -1 )
                        {
                            A_nonzero_values[9 * a_col_it + lane_id_mod_9] -= tmp;
                        }
                    } // Loop over columns that have a match (for k=0;k<n_cols)
                } // Loop over the columns of waRowId

                //}  // Loop j=0;j<32
            } // Loop over the columns of aRowId

            globalWarpId += nWarps * gridDim.x;
        } // end of loop over I


        // Now compute the inverse of the block C_jj
        if ( lane_id_div_9 == 0 || I == 2 )
        {


            const int offset = 9 * A_dia_indices[storedRowId[lane_id_div_9]] + lane_id_mod_9;
            s_C_mtx[warpId][laneId] = A_nonzero_values[offset];    


            utils::syncwarp(utils::activemask());

            if (ROW_MAJOR)
            {

                compute_block_inverse_row_major3x3_formula2 <int, ValueTypeA, 3, true>( s_C_mtx[warpId], 9 * lane_id_div_9, offset, idx_i, idx_j, A_nonzero_values );              


            }
            else
            {
                //#TODO compute_block_inverse_row_major3x3_formula2
            }

        } // End of if statement
    } // End of while loop
}

#else
    FatalError("Haven't implemented compute_LU_factors_3x3 for Not EXPERIMENTAL_LU_FACTORS", AMGX_ERR_NOT_SUPPORTED_TARGET);
#endif

// #ifdef EXPERIMENTAL_LU_FACTORS

// template< typename ValueTypeA, int CtaSize, int SMemSize, bool ROW_MAJOR >
// __global__
// #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
// __launch_bounds__( CtaSize, 12 )
// #elif defined(__CUDA_ARCH__)
// __launch_bounds__( CtaSize, 8 )
// #endif
// void
// compute_LU_factors_5x5_kernel_warp( int A_nRows,
//                                     const int *__restrict A_row_offsets,
//                                     const int *__restrict A_col_indices,
//                                     const int *__restrict A_dia_indices,
//                                     ValueTypeA *__restrict A_nonzero_values,
//                                     const int *__restrict A_smaller_color_offsets,
//                                     const int *__restrict A_larger_color_offsets,
//                                     const int *sorted_rows_by_color,
//                                     const int num_rows_per_color,
//                                     const int current_color,
//                                     const int blockDim,
//                                     int *wk_returnValue )
// {
//     const int nWarps = CtaSize / 32; // Number of warps per Cta
//     const int Sq = blockDim * blockDim;
//     const int warpId = utils::warp_id();
//     const int laneId = utils::lane_id();
//     // // Lane ID in the b Sq-wide segments.
//     // const int lane_id_div_Sq = threadIdx.x / Sq;
//     // const int lane_id_mod_Sq = threadIdx.x % Sq;
//     // Coordinates inside a bxb block of the matrix.
//     const int idx_i = laneId / blockDim;
//     const int idx_j = laneId % blockDim;
//     int globalBlockId = blockIdx.x;
//     // Shared memory to store the blocks to process
//     __shared__ volatile ValueTypeA s_C_mtx[64];
//     __shared__ volatile ValueTypeA s_F_mtx[32];
//     // Shared memory to store the proposed column to load
// #if __CUDA_ARCH__ < 300
//     // __shared__ volatile int s_aColItToLoad [nWarps][32];
//     // __shared__ volatile int s_waColItToLoad[nWarps][32];
// #else
//     __shared__ volatile int s_aColIt[nWarps][32];
//     __shared__ volatile int s_waColIt[nWarps][32];
//     __shared__ volatile int s_aColSrc[nWarps][32];
// #endif
//     // Shared memory to store the column indices of the current row
//     __shared__ volatile int s_keys[SMemSize];
//     __shared__ volatile int s_numCols[32];

//     while (globalBlockId < num_rows_per_color)
//     {
//         int storedRowId[2];
//         int count = 0;
//         int seq = 1;
//         for ( int I = 0; I < 2 && globalBlockId < num_rows_per_color ; I++)
//         {
//             int aRowId = sorted_rows_by_color[globalBlockId];
//             storedRowId[I] = aRowId;
//             int aColBeg = A_row_offsets[aRowId + 0];
//             int aColEnd = A_row_offsets[aRowId + 1];
//             int aColSmaller = A_smaller_color_offsets[aRowId];
//             // The number of columns.
//             const int nCols = aColEnd - aColBeg;

//             //TODO: Add fallback for cases where number of nonzeros exceed SMemSize
//             if ( nCols > SMemSize )
//             {
//                 wk_returnValue[0] = 1;
//                 return;
//             }

//             // Fill-in the local table.
//             const int NUM_STEPS = SMemSize / 64;

//             __syncthreads();
// #pragma unroll
//             for ( int step = 0, k = threadIdx.x; step < NUM_STEPS ; ++step, k += 64 )
//             {
//                 int aColIt = aColBeg + k;
//                 int aColId = -1;
                
                
//                 if ( aColIt < aColEnd )
//                 {
//                     aColId = A_col_indices[aColIt];
//                     // printf("color:%d  %d in %d\n",current_color,aColId,aColIt);
//                 }

//                 s_keys[k] = aColId;
                
//                 __syncthreads();
//             }


//             // Now load all column indices of neighbours that have colors smaller than yours
//             for ( int aColIt = aColBeg; aColIt < aColSmaller; aColIt++)
//             {
//                 unsigned int active_mask = utils::activemask();
//                 // Read the row to process, should be a broadcast
//                 int waRowId = s_keys[aColIt - aColBeg];
//                 // Compute multiplicative factor, load C_jj in first half, C_ij in second half
//                 int aColIdx = aColIt;
//                 // if(aRowId == 116147 && laneId == 0)
//                 //     for(int kk = 0; kk < nCols;kk++)
//                 //         printf("%d  kk %d\n",s_keys[kk],kk);
//                 //     printf("aColBeg %d aColSmaller %d aColIt  %d \n",aColBeg,aColSmaller,aColIt);

//                 if ( warpId == 0 )
//                 {
//                     aColIdx = A_dia_indices[waRowId];
//                 }

//                 if ( laneId < 25 )
//                 {
//                     s_C_mtx[warpId * 32 + laneId] = A_nonzero_values[Sq * aColIdx + laneId];
//                     // if(aRowId == 1208608 && laneId == 0 && aColIt == aColBeg)
//                     // {
//                     //     printf("warpId %d %lf %lf %d %d\n",warpId,s_C_mtx[warpId * 32 + laneId], A_nonzero_values[Sq * aColIdx + laneId], aColIdx, laneId);
//                     // }
//                 }

//                 __syncthreads();
//                 // Threads 0-sq perform the matrix product
//                 ValueTypeA tmp(0);

                
//                 if(threadIdx.x < 25)
//                 {
//                     if (ROW_MAJOR)
//                     {
// #pragma unroll
//                         for ( int m = 0 ; m < blockDim ; ++m )
//                         {
//                             // tmp +=  s_C_mtx[32+  blockDim * idx_i + m] * s_C_mtx[blockDim * m + idx_j];
//                             tmp +=  A_nonzero_values[Sq * aColIt+  blockDim * idx_i + m] * A_nonzero_values[Sq * aColIdx + blockDim * m + idx_j];
//                     // if(aRowId == 1208608 && idx_i == 0 && idx_j ==  0 && aColIt == aColBeg) 
//                     // {
//                     //     printf("m %d %lf threadIdx.x %d %lf %lf  %lf   %lf\n",m,tmp,threadIdx.x,s_C_mtx[32 + blockDim * idx_i + m],s_C_mtx[blockDim * m + idx_j],A_nonzero_values[Sq * aColIdx],A_nonzero_values[Sq * aColIt]);
//                     // }

//                         }

//                     }
//                     else
//                     {

//     // #pragma unroll
//     //                         for ( int m = 0 ; m < 3 ; ++m )
//     //                         {
//     //                             tmp += s_C_mtx[warpId][9 + 3 * m + idx_j] * s_C_mtx[warpId][3 * idx_i + m];
//     //                         }
//                     }
                        
//                 }
                

//                 __syncthreads();

//                 if (threadIdx.x < 25)
//                 {
//                     s_F_mtx[laneId] = tmp;
//                     A_nonzero_values[Sq * aColIt + laneId] = tmp;
//                 }
//                 __syncthreads();

//                 int waColIt  = ld_cg(A_larger_color_offsets + waRowId);
//                 int waColEnd = ld_cg(A_row_offsets + waRowId + 1);
                
//                 // Load the first 32 columns of waRowId
//                 waColIt += threadIdx.x;
//                 // Each thread loads its column id
//                 int waColId = -1;

//                 if ( waColIt < waColEnd )
//                 {
//                     waColId = A_col_indices[waColIt];
//                 }

//                 // Find the right column.
//                 int found_aColIt = -1;

// #pragma unroll 4
//                 for ( int i = 0, num_keys = aColEnd - aColBeg ; (i < num_keys) && (warpId == 0 ); ++i ){
//                     if ( s_keys[i] == waColId )
//                     {
//                         found_aColIt = i;
                        
//                     }
//                     // if(aRowId == 116147)
//                     //     printf("i: %d thread %d found %d waColId %d waColIt %d %d warpId %d\n",i,threadIdx.x, found_aColIt,waColId,waColIt,s_keys[i],warpId);

//                 }


//                 if ( found_aColIt != -1 )
//                 {
//                     found_aColIt += aColBeg;
//                     // if(aRowId == 86621 )
//                     //     printf("found %d\n",found_aColIt);
//                 }
//                 __syncthreads();

//                 // Store all the columns that have been found
//                 const int pred = found_aColIt != -1;
//                 int vote = utils::ballot( pred, active_mask );
//                 const int idst = __popc(vote & utils::lane_mask_lt());

//                 if (pred)
//                 {
// #if __CUDA_ARCH__ < 300
//                     // s_aColItToLoad [warpId][idst] = found_aColIt;
//                     // s_waColItToLoad[warpId][idst] = waColIt;
// #else
//                     // s_aColSrc[warpId][idst] = threadIdx.x;
//                     s_aColIt [warpId][idst] = found_aColIt;
//                     s_waColIt [warpId][idst] = waColIt;
// #endif
//                 }
//                 __syncthreads();

//                 const int n_cols = __popc( vote );

//                 // s_numCols[warpId] = 0;
//                 // if(laneId == 0)
//                 //     s_numCols[warpId] = n_cols;

//                 __syncthreads();

//                 // Process all columns that have been found
//                 // for ( int j = 0 ; j < nWarps; j++ )
//                 // {
//                     // int n_cols_warp = s_numCols[j];;
//                     for ( int k = 0 ; (k < n_cols) && (warpId == 0 ) ; k += 1 )
//                     {
//                         const int my_k = k + warpId;
//                         // Where to get columns from.
//                         int a_col_it = -1, w_col_it = -1;
//                         // Load column to load
// #if __CUDA_ARCH__ < 300
//                         // if ( my_k < n_cols )
//                         // {
//                         //     a_col_it = s_aColItToLoad [warpId][my_k];
//                         //     w_col_it = s_waColItToLoad[warpId][my_k];
//                         // }
// #else
//                         // a_col_it = utils::shfl(found_aColIt, s_aColSrc[j][my_k], warpSize, active_mask);
//                         // w_col_it = utils::shfl(waColIt,      s_aColSrc[j][my_k], warpSize, active_mask);
//                         if ( my_k < n_cols )
//                         {
//                             a_col_it = s_aColIt [warpId][my_k];
//                             w_col_it = s_waColIt[warpId][my_k];
//                         }

// #endif
//                         ValueTypeA my_C(0);

//                         if ( (w_col_it != -1) && (laneId < 25) )
//                         {
//                             my_C = A_nonzero_values[Sq * w_col_it + laneId];
//                         }

//                         s_C_mtx[laneId] = my_C;
//                         // Run the matrix-matrix product.
//                         ValueTypeA tmp(0);


//                         if ( laneId < 25 )
//                         {
//                             if (ROW_MAJOR)
//                             {
// #pragma unroll
//                                 for ( int m = 0 ; m < blockDim ; ++m )
//                                 {
//                                     // tmp += s_F_mtx[blockDim * idx_i + m] * s_C_mtx[ blockDim * m + idx_j];
//                                     tmp += A_nonzero_values[Sq * aColIt + blockDim * idx_i + m] * A_nonzero_values[Sq * w_col_it + blockDim * m + idx_j];
//                                     // if(aRowId == 1208608 && idx_i == 0 && idx_j == 0 && threadIdx.x == 0) 
//                                     //     printf("tmp %lf threadIdx.x %d %lf %lf\n",tmp,threadIdx.x,s_F_mtx[blockDim * idx_i + m],s_C_mtx[ blockDim * m + idx_j]);
//                                 }
//                             }
//                             else
//                             {
// // #pragma unroll
// //                             for ( int m = 0 ; m < 3 ; ++m )
// //                             {
// //                                 tmp += s_F_mtx[warpId][3 * m + idx_j] * s_C_mtx[warpId][9 * lane_id_div_9 + 3 * idx_i + m];
// //                             }
//                             }

//                             if ( a_col_it != -1 )
//                             {
//                     //     if(aRowId == 1208608 ) 
//                     // printf("before I %d color %d %d num :%d result %lf has threadIdx.x %d tmp %lf %d block %d\n",I,current_color,globalBlockId,num_rows_per_color,A_nonzero_values[Sq * a_col_it + laneId] ,laneId,tmp,a_col_it,blockIdx.x );  
//                                 utils::syncwarp(utils::activemask());
//                                 // A_nonzero_values[Sq * a_col_it + laneId] -= tmp;     
//                     if(seq == 2)
//                         printf("wrong!!!\n");
//                     //     if(aRowId == 1208608 ) 
//                     // printf("after I %d  color %d  %d num :%d result %lf has threadIdx.x %d  tmp %lf %d block %d\n",I,current_color,globalBlockId,num_rows_per_color,A_nonzero_values[Sq * a_col_it + laneId] ,laneId,tmp,a_col_it,blockIdx.x );               
//                             }

//                         }

//                     } // Loop over columns that have a match (for k=0;k<n_cols)


//                 // }
//                 // Loop over the columns of waRowId

//                 //}  // Loop j=0;j<32
//             } // Loop over the columns of aRowId
//             count ++;
//             //  if( aRowId == 1208608 ) 
//             //     printf("block %d thread %d griddim %d count %d warpId %d %d %d\n",blockIdx.x,threadIdx.x,gridDim.x,count,warpId,storedRowId[0],storedRowId[1]);
//             globalBlockId += gridDim.x;
//         } // end of loop over I
//         __syncthreads();
//         seq = 2;
//         // Now compute the inverse of the block C_jj
//         if( (warpId == 0 || count == 2) && laneId < 25 )
//         {
//             const int offset = Sq * A_dia_indices[storedRowId[warpId]] + laneId;
//             s_C_mtx[warpId * 32 + laneId] = A_nonzero_values[offset];
//             utils::syncwarp(utils::activemask());

//             if (ROW_MAJOR)
//             {
//                 // compute_block_inverse_row_major_5_8<int, ValueTypeA>( s_C_mtx, Sq * lane_id_div_Sq, offset, idx_i, idx_j, A_nonzero_values, blockDim, lane_id_div_Sq, I);
//                 // if(storedRowId[warpId] == 1208608 ) 
//                 //     printf("I : %d 1 %d %d result %lf has threadIdx.x %d offset %d %d %d warpId %d blockIdx.x %d\n",count,current_color,globalBlockId,A_nonzero_values[offset+ laneId] ,threadIdx.x,offset,storedRowId[0],storedRowId[1],warpId,blockIdx.x);          
//                 // if(storedRowId[warpId] == 1193609 ) 
//                 //     printf("A : %d 1 %d %d result %lf has threadIdx.x %d offset %d %d %d warpId %d\n",count,current_color,globalBlockId,A_nonzero_values[offset+ laneId] ,threadIdx.x,offset,storedRowId[0],storedRowId[1],warpId);                      
//                 compute_block_inverse_row_major5x5_formula2<int, ValueTypeA, 5, true>( &s_C_mtx[warpId * 32], 0, offset, idx_i, idx_j, A_nonzero_values );
//                 // if(storedRowId[warpId] == 1208608 ) 
//                 //     printf("2 %d %d result %lf has threadIdx.x %d offset %d %d %d\n",current_color,globalBlockId,A_nonzero_values[offset+ laneId] ,laneId,offset,storedRowId[0],storedRowId[1]);          
//             }
//             else
//             {
//                 //#TODO compute_block_inverse_row_major3x3_formula2
//             }


//         }
//         __syncthreads();

//     } // End of while loop
// }

// #else
//     FatalError("Haven't implemented compute_LU_factors_5x5 for Not EXPERIMENTAL_LU_FACTORS", AMGX_ERR_NOT_SUPPORTED_TARGET);
// #endif


#ifdef EXPERIMENTAL_LU_FACTORS

template< typename ValueTypeA, int CtaSize, int SMemSize, bool ROW_MAJOR >
__global__
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
__launch_bounds__( CtaSize, 12 )
#elif defined(__CUDA_ARCH__)
__launch_bounds__( CtaSize, 8 )
#endif
void
compute_LU_factors_5x5_kernel_warp( int A_nRows,
                                    const int *__restrict A_row_offsets,
                                    const int *__restrict A_col_indices,
                                    const int *__restrict A_dia_indices,
                                    ValueTypeA *__restrict A_nonzero_values,
                                    const int *__restrict A_smaller_color_offsets,
                                    const int *__restrict A_larger_color_offsets,
                                    const int *sorted_rows_by_color,
                                    const int num_rows_per_color,
                                    const int current_color,
                                    const int blockDim,
                                    int *wk_returnValue )
{
    const int nWarps = CtaSize / 32; // Number of warps per Cta
    const int warpId = utils::warp_id();
    const int laneId = utils::lane_id();

    // Coordinates inside a bxb block of the matrix.
    const int idx_i = laneId / blockDim;
    const int idx_j = laneId % blockDim;
    int globalWarpId = blockIdx.x * nWarps + warpId;
    // Shared memory to store the blocks to process
    __shared__ volatile ValueTypeA s_C_mtx[nWarps][64];
    __shared__ volatile ValueTypeA s_F_mtx[nWarps][32];
    // Shared memory to store the proposed column to load
#if __CUDA_ARCH__ < 300
    __shared__ volatile int s_aColItToLoad [nWarps][32];
    __shared__ volatile int s_waColItToLoad[nWarps][32];
#else
    __shared__ volatile int s_aColSrc[nWarps][32];
#endif
    // Shared memory to store the column indices of the current row
    __shared__ volatile int s_keys[nWarps][SMemSize];

    while (globalWarpId < num_rows_per_color)
    {
        
        int aRowId = sorted_rows_by_color[globalWarpId];      
        int aColBeg = A_row_offsets[aRowId + 0];
        int aColEnd = A_row_offsets[aRowId + 1];
        int aColSmaller = A_smaller_color_offsets[aRowId];
        // The number of columns.
        const int nCols = aColEnd - aColBeg;

        //TODO: Add fallback for cases where number of nonzeros exceed SMemSize
        if ( nCols > SMemSize )
        {
            wk_returnValue[0] = 1;
            return;
        }

        // Fill-in the local table.
        const int NUM_STEPS = SMemSize / 32;

#pragma unroll
        for ( int step = 0, k = laneId; step < NUM_STEPS ; ++step, k += 32 )
        {
            int aColIt = aColBeg + k;
            int aColId = -1;
            
            
            if ( aColIt < aColEnd )
            {
                aColId = A_col_indices[aColIt];
                // printf("color:%d  %d in %d\n",current_color,aColId,aColIt);
            }

            s_keys[warpId][k] = aColId;
            
        }

        // Now load all column indices of neighbours that have colors smaller than yours
        for ( int aColIt = aColBeg; aColIt < aColSmaller; aColIt++)
        {
            unsigned int active_mask = utils::activemask();
            // Read the row to process, should be a broadcast
            int waRowId = s_keys[warpId][aColIt - aColBeg];
            // Compute multiplicative factor, load C_jj in first half, C_ij in second half
            int aColIdx = A_dia_indices[waRowId];

            if ( laneId < 25 )
            {
                s_C_mtx[warpId][laneId] = A_nonzero_values[25 * aColIdx + laneId];
                s_C_mtx[warpId][32 + laneId] = A_nonzero_values[25 * aColIt + laneId];
            }

            utils::syncwarp( active_mask );
            // Threads 0-25 perform the matrix product
            ValueTypeA tmp(0);

            
            if(laneId < 25)
            {
                if (ROW_MAJOR)
                {
#pragma unroll
                    for ( int m = 0 ; m < 5; ++m )
                    {
                        tmp +=  s_C_mtx[warpId][32 +  5 * idx_i + m] * s_C_mtx[warpId][5 * m + idx_j];
                        // tmp +=  A_nonzero_values[Sq * aColIt+  blockDim * idx_i + m] * A_nonzero_values[Sq * aColIdx + blockDim * m + idx_j];

                    }

                }
                else
                {
// #pragma unroll

                }
                    
            }
            

            if (laneId < 25)
            {
                s_F_mtx[warpId][laneId] = tmp;
                A_nonzero_values[25 * aColIt + laneId] = tmp;
            }


            int waColIt  = ld_cg(A_larger_color_offsets + waRowId);
            int waColEnd = ld_cg(A_row_offsets + waRowId + 1);

            // Load the first 32 columns of waRowId
            for (waColIt += laneId ; utils::any(waColIt < waColEnd, active_mask ); waColIt += 32 )
            {
                // Each thread loads its column id
                int waColId = -1;

                if ( waColIt < waColEnd )
                {
                    waColId = A_col_indices[waColIt];
                }

                // Find the right column.
                int found_aColIt = -1;

#pragma unroll 4
                for ( int i = 0, num_keys = aColEnd - aColBeg ; i < num_keys ; ++i )
                    if ( s_keys[warpId][i] == waColId )
                    {
                        found_aColIt = i;
                    }

                if ( found_aColIt != -1 )
                {
                    found_aColIt += aColBeg;
                }

                // Store all the columns that have been found
                const int pred = found_aColIt != -1;
                int vote = utils::ballot( pred, active_mask );
                const int idst = __popc(vote & utils::lane_mask_lt());

                if (pred)
                {
#if __CUDA_ARCH__ < 300
                    s_aColItToLoad [warpId][idst] = found_aColIt;
                    s_waColItToLoad[warpId][idst] = waColIt;
#else
                    s_aColSrc[warpId][idst] = laneId;
#endif
                }
                utils::syncwarp(active_mask);

                const int n_cols = __popc( vote );

                // Process all columns that have been found
                for ( int k = 0 ; k < n_cols ; k ++ )
                {
                    
                    int a_col_it = -1, w_col_it = -1;
                    // Load column to load
#if __CUDA_ARCH__ < 300
                    // if ( my_k < n_cols )
                    // {
                    //     a_col_it = s_aColItToLoad [warpId][my_k];
                    //     w_col_it = s_waColItToLoad[warpId][my_k];
                    // }
#else
                    a_col_it = utils::shfl(found_aColIt, s_aColSrc[warpId][k], warpSize, active_mask);
                    w_col_it = utils::shfl(waColIt,      s_aColSrc[warpId][k], warpSize, active_mask);

#endif
                    ValueTypeA my_C(0);

                    if ( w_col_it != -1 )
                    {
                        if(laneId < 25)
                            my_C = A_nonzero_values[25 * w_col_it + laneId];
                    }

                    s_C_mtx[warpId][laneId] = my_C;
                    // Run the matrix-matrix product.
                    ValueTypeA tmp(0);
                    utils::syncwarp( active_mask );

                    if(laneId < 25)
                    {
                        if (ROW_MAJOR)
                        {
#pragma unroll
                            for ( int m = 0 ; m < 5 ; ++m )
                            {
                                tmp += s_F_mtx[warpId][5 * idx_i + m] * s_C_mtx[warpId][5 * m + idx_j];
                            }
                        }
                        else
                        {
// #pragma unroll
//                             for ( int m = 0 ; m < 4 ; ++m )
//                             {
//                                 tmp += s_F_mtx[warpId][4 * m + idx_j] * s_C_mtx[warpId][16 * lane_id_div_16 + 4 * idx_i + m];
//                             }
                        }

                        if ( a_col_it != -1 )
                        {
                            A_nonzero_values[25 * a_col_it + laneId] -= tmp;
                        }
                    }
                } // Loop over columns that have a match (for k=0;k<n_cols)
            } // Loop over the columns of waRowId


        } // Loop over the columns of aRowId

        globalWarpId += nWarps * gridDim.x;


        // Now compute the inverse of the block C_jj
        if( laneId < 25 )
        {
            const int offset = 25 * A_dia_indices[aRowId] + laneId;
            s_C_mtx[warpId][laneId] = A_nonzero_values[offset];
            utils::syncwarp(utils::activemask());


            if (ROW_MAJOR)
            {

                compute_block_inverse_row_major5x5_formula2<int, ValueTypeA, 5, true>( s_C_mtx[warpId], 0, offset, idx_i, idx_j, A_nonzero_values );
            }
            else
            {
                //#TODO compute_block_inverse_row_major3x3_formula2
            }
            


        }


    } // End of while loop
}

#else
    FatalError("Haven't implemented compute_LU_factors_5x5 for Not EXPERIMENTAL_LU_FACTORS", AMGX_ERR_NOT_SUPPORTED_TARGET);
#endif


#ifdef EXPERIMENTAL_LU_FACTORS

template< typename ValueTypeA, int CtaSize, int SMemSize, bool ROW_MAJOR >
__global__
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
__launch_bounds__( CtaSize, 12 )
#elif defined(__CUDA_ARCH__)
__launch_bounds__( CtaSize, 8 )
#endif
void
compute_LU_factors_6_8x6_8_kernel_warp( int A_nRows,
                                    const int *__restrict A_row_offsets,
                                    const int *__restrict A_col_indices,
                                    const int *__restrict A_dia_indices,
                                    ValueTypeA *__restrict A_nonzero_values,
                                    const int *__restrict A_smaller_color_offsets,
                                    const int *__restrict A_larger_color_offsets,
                                    const int *sorted_rows_by_color,
                                    const int num_rows_per_color,
                                    const int current_color,
                                    const int blockDim,
                                    int *wk_returnValue )
{
    const int nWarps = CtaSize / 32; // Number of warps per Cta
    const int Sq = blockDim * blockDim;
    const int warpId = utils::warp_id();
    const int laneId = utils::lane_id();
    // Lane ID in the b Sq-wide segments.
    const int lane_id_div_Sq = threadIdx.x / Sq;
    const int lane_id_mod_Sq = threadIdx.x % Sq;
    // Coordinates inside a bxb block of the matrix.
    const int idx_i = lane_id_mod_Sq / blockDim;
    const int idx_j = lane_id_mod_Sq % blockDim;
    int globalBlockId = blockIdx.x;
    // Shared memory to store the blocks to process
    __shared__ volatile ValueTypeA s_C_mtx[128];
    __shared__ volatile ValueTypeA s_F_mtx[64];
    // Shared memory to store the proposed column to load
#if __CUDA_ARCH__ < 300
    // __shared__ volatile int s_aColItToLoad [nWarps][32];
    // __shared__ volatile int s_waColItToLoad[nWarps][32];
#else
    __shared__ volatile int s_aColIt[nWarps][32];
    __shared__ volatile int s_waColIt[nWarps][32];
#endif
    // Shared memory to store the column indices of the current row
    __shared__ volatile int s_keys[SMemSize];
    __shared__ volatile int s_numCols[nWarps];

    while (globalBlockId < num_rows_per_color)
    {
        int storedRowId[2];
        int I = 0;

        for (; I < 2 && globalBlockId < num_rows_per_color ; I++)
        {
            int aRowId = sorted_rows_by_color[globalBlockId];
            storedRowId[I] = aRowId;
            int aColBeg = A_row_offsets[aRowId + 0];
            int aColEnd = A_row_offsets[aRowId + 1];
            int aColSmaller = A_smaller_color_offsets[aRowId];
            // The number of columns.
            const int nCols = aColEnd - aColBeg;

            //TODO: Add fallback for cases where number of nonzeros exceed SMemSize
            if ( nCols > SMemSize )
            {
                wk_returnValue[0] = 1;
                return;
            }

            int aColIt = aColBeg + threadIdx.x;
            int aColId = -1;

            if ( aColIt < aColEnd )
            {
                aColId = A_col_indices[aColIt];
            }

            s_keys[threadIdx.x] = aColId;
            __syncthreads();

            // Now load all column indices of neighbours that have colors smaller than yours
            for ( int aColIt = aColBeg; aColIt < aColSmaller; aColIt++)
            {
                unsigned int active_mask = utils::activemask();
                // Read the row to process, should be a broadcast
                int waRowId = s_keys[aColIt - aColBeg];
                // Compute multiplicative factor, load C_jj in first half, C_ij in second half
                int aColIdx = aColIt;


                if ( lane_id_div_Sq == 0 )
                {
                    aColIdx = A_dia_indices[waRowId];
                }

                if ( lane_id_div_Sq < 2 )
                {
                    s_C_mtx[threadIdx.x] = A_nonzero_values[Sq * aColIdx + lane_id_mod_Sq];
                }

                __syncthreads();
                // Threads 0-sq perform the matrix product
                ValueTypeA tmp(0);

                if (lane_id_div_Sq == 0)
                {
                    if (ROW_MAJOR)
                    {

                        for ( int m = 0 ; m < blockDim ; ++m )
                        {
                            tmp += s_C_mtx[Sq + blockDim * idx_i + m] * s_C_mtx[blockDim * m + idx_j];
                        }
                    }
                    else
                    {

    // #pragma unroll
    //                         for ( int m = 0 ; m < 3 ; ++m )
    //                         {
    //                             tmp += s_C_mtx[warpId][9 + 3 * m + idx_j] * s_C_mtx[warpId][3 * idx_i + m];
    //                         }
                    }
                }

                __syncthreads();

                if ( lane_id_div_Sq == 0 )
                {
                    s_F_mtx[threadIdx.x] = tmp;
                    A_nonzero_values[Sq * aColIt + threadIdx.x] = tmp;
                }
                __syncthreads();

                int waColIt  = ld_cg(A_larger_color_offsets + waRowId);
                int waColEnd = ld_cg(A_row_offsets + waRowId + 1);

                // Load the first 32 columns of waRowId
                waColIt += threadIdx.x;
                // Each thread loads its column id
                int waColId = -1;

                if ( waColIt < waColEnd )
                {
                    waColId = A_col_indices[waColIt];
                }

                // Find the right column.
                int found_aColIt = -1;

#pragma unroll 4
                for ( int i = 0, num_keys = aColEnd - aColBeg ; i < num_keys ; ++i )
                    if ( s_keys[i] == waColId )
                    {
                        found_aColIt = i;
                    }

                if ( found_aColIt != -1 )
                {
                    found_aColIt += aColBeg;
                }

                // Store all the columns that have been found
                const int pred = found_aColIt != -1;
                int vote = utils::ballot( pred, active_mask );
                const int idst = __popc(vote & utils::lane_mask_lt());

                if (pred)
                {
#if __CUDA_ARCH__ < 300
                    // s_aColItToLoad [warpId][idst] = found_aColIt;
                    // s_waColItToLoad[warpId][idst] = waColIt;
#else
                    // s_aColSrc[warpId][idst] = threadIdx.x;
                    s_aColIt [warpId][idst] = found_aColIt;
                    s_waColIt [warpId][idst] = waColIt;
#endif
                }
                utils::syncwarp(active_mask);

                const int n_cols = __popc( vote );
                s_numCols[warpId] = 0;
                if(laneId == 0)
                s_numCols[warpId] = n_cols;

                __syncthreads();

                // Process all columns that have been found
                for ( int j = 0 ; j < nWarps ; j++ )
                {
                    int n_cols_warp = s_numCols[j];
                    for ( int k = 0 ; k < n_cols_warp; k += 2 )
                    {
                        const int my_k = k + lane_id_div_Sq;
                        // Where to get columns from.
                        int a_col_it = -1, w_col_it = -1;
                        // Load column to load
#if __CUDA_ARCH__ < 300
                        // if ( my_k < n_cols )
                        // {
                        //     a_col_it = s_aColItToLoad [warpId][my_k];
                        //     w_col_it = s_waColItToLoad[warpId][my_k];
                        // }
#else
                        // a_col_it = utils::shfl(found_aColIt, s_aColSrc[warpId][my_k], warpSize, active_mask);
                        // w_col_it = utils::shfl(waColIt,      s_aColSrc[warpId][my_k], warpSize, active_mask);
                        if ( my_k < n_cols_warp && lane_id_div_Sq < 2 )
                        {
                            a_col_it = s_aColIt [j][my_k];
                            w_col_it = s_waColIt[j][my_k];
                        }

#endif
                        ValueTypeA my_C(0);

                        if ( w_col_it != -1 && lane_id_div_Sq < 2 )
                        {
                            my_C = A_nonzero_values[Sq * w_col_it + lane_id_mod_Sq];
                        }

                        if (lane_id_div_Sq < 2)
                             s_C_mtx[threadIdx.x] = my_C;
                        // Run the matrix-matrix product.
                        ValueTypeA tmp(0);
                        // utils::syncwarp( active_mask );
                        __syncthreads();



                        if ( lane_id_div_Sq < 2 )
                        {
                            if (ROW_MAJOR)
                            {
                                for ( int m = 0 ; m < blockDim ; ++m )
                                {
                                    tmp += s_F_mtx[blockDim * idx_i + m] * s_C_mtx[Sq * lane_id_div_Sq + blockDim * m + idx_j];
                                }
                            }
                            else
                            {
// #pragma unroll
//                             for ( int m = 0 ; m < 3 ; ++m )
//                             {
//                                 tmp += s_F_mtx[warpId][3 * m + idx_j] * s_C_mtx[warpId][9 * lane_id_div_9 + 3 * idx_i + m];
//                             }
                            }

                            if ( a_col_it != -1 )
                            {
                                A_nonzero_values[Sq * a_col_it + lane_id_mod_Sq] -= tmp;
                            }

                        }

                    } // Loop over columns that have a match (for k=0;k<n_cols)


                }
                // Loop over the columns of waRowId

                //}  // Loop j=0;j<32
            } // Loop over the columns of aRowId

            globalBlockId += gridDim.x;
        } // end of loop over I

        // Now compute the inverse of the block C_jj
         int offset = 0;
         if ( (lane_id_div_Sq == 0 || I == 2) &&  lane_id_div_Sq < 2 )
         {
            offset = Sq * A_dia_indices[storedRowId[lane_id_div_Sq]] + lane_id_mod_Sq;
            s_C_mtx[threadIdx.x] = A_nonzero_values[offset];
         } // End of if statement
         __syncthreads();


        if (ROW_MAJOR)
        {
            compute_block_inverse_row_major_5_8<int, ValueTypeA>( s_C_mtx, Sq * lane_id_div_Sq, offset, idx_i, idx_j, A_nonzero_values, blockDim, lane_id_div_Sq, I);
        }
        else
        {
            //#TODO compute_block_inverse_row_major3x3_formula2
        }


    } // End of while loop
}

#else
    FatalError("Haven't implemented compute_LU_factors_6~8x6~8 for Not EXPERIMENTAL_LU_FACTORS", AMGX_ERR_NOT_SUPPORTED_TARGET);
#endif

// ----------
// Methods
// ----------

// Constructor
template<class T_Config>
MulticolorILUSolver_Base<T_Config>::MulticolorILUSolver_Base( AMG_Config &cfg, const std::string &cfg_scope) : Solver<T_Config>( cfg, cfg_scope)
{
    m_sparsity_level = cfg.AMG_Config::getParameter<int>("ilu_sparsity_level", cfg_scope);
    m_weight = cfg.AMG_Config::getParameter<double>("relaxation_factor", cfg_scope);
    this->m_reorder_cols_by_color_desired = (cfg.AMG_Config::getParameter<int>("reorder_cols_by_color", cfg_scope) != 0);
    this->m_insert_diagonal_desired = (cfg.AMG_Config::getParameter<int>("insert_diag_while_reordering", cfg_scope) != 0);

    if (cfg.AMG_Config::getParameter<int>("use_bsrxmv", cfg_scope))
    {
        this->m_use_bsrxmv = 1;
    }
    else
    {
        this->m_use_bsrxmv = 0;
    }

    if (m_weight == ValueTypeB(0.))
    {
        m_weight = 1.;
        amgx_printf("Warning, setting weight to 1 instead of estimating largest_eigen_value in Multicolor DILU smoother\n");
    }
}

// Destructor
template<class T_Config>
MulticolorILUSolver_Base<T_Config>::~MulticolorILUSolver_Base()
{
    m_LU.set_initialized(0);
    m_A_to_LU_mapping.clear();
    m_A_to_LU_mapping.shrink_to_fit();
    m_LU.resize(0, 0, 0, 1);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void MulticolorILUSolver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::computeAtoLUmapping()
{
    FatalError("Haven't implemented Multicolor ILU smoother for host format", AMGX_ERR_NOT_SUPPORTED_TARGET);
}


template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void MulticolorILUSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::computeAtoLUmapping()
{
    Matrix<TConfig_d> &m_A = *this->m_explicit_A;
    const int CtaSize = 128; // Number of threads per CTA
    const int SMemSize = 128;  // per warp
    const int nWarps = CtaSize / 32;
    int GridSize = std::min( AMGX_GRID_MAX_SIZE, ( this->m_explicit_A->get_num_rows( ) + nWarps - 1 ) / nWarps );
    // Global memory workspaces
    device_vector_alloc<int> returnValue(1);
    returnValue[0] = 0;

    if (this->m_explicit_A->hasProps(DIAG))
    {
        computeAtoLUmappingExtDiag_kernel<CtaSize, SMemSize> <<< GridSize, CtaSize >>> (
            m_A.get_num_rows( ),
            thrust::raw_pointer_cast( &m_A.row_offsets[0] ),
            thrust::raw_pointer_cast( &m_A.col_indices[0] ),
            thrust::raw_pointer_cast( &m_A.diag[0] ),
            thrust::raw_pointer_cast( &this->m_LU.row_offsets[0] ),
            thrust::raw_pointer_cast( &this->m_LU.col_indices[0] ),
            thrust::raw_pointer_cast( &this->m_A_to_LU_mapping[0] ),
            thrust::raw_pointer_cast( &returnValue[0] ));
    }
    else
    {
        computeAtoLUmapping_kernel<CtaSize, SMemSize> <<< GridSize, CtaSize >>> (
            m_A.get_num_rows( ),
            thrust::raw_pointer_cast( &m_A.row_offsets[0] ),
            thrust::raw_pointer_cast( &m_A.col_indices[0] ),
            thrust::raw_pointer_cast( &this->m_LU.row_offsets[0] ),
            thrust::raw_pointer_cast( &this->m_LU.col_indices[0] ),
            thrust::raw_pointer_cast( &this->m_A_to_LU_mapping[0] ),
            thrust::raw_pointer_cast( &returnValue[0] ));
    }

    cudaCheckError();

    // fallback path that allows 1024 nonzeros per row
    if (returnValue[0] == 1)
    {
        returnValue[0] = 0;
        const int SMemSize2 = 1024 ;  // per warp

        if (this->m_explicit_A->hasProps(DIAG))
        {
            computeAtoLUmappingExtDiag_kernel<CtaSize, SMemSize2> <<< GridSize, CtaSize >>> (
                m_A.get_num_rows( ),
                thrust::raw_pointer_cast( &m_A.row_offsets[0] ),
                thrust::raw_pointer_cast( &m_A.col_indices[0] ),
                thrust::raw_pointer_cast( &m_A.diag[0] ),
                thrust::raw_pointer_cast( &this->m_LU.row_offsets[0] ),
                thrust::raw_pointer_cast( &this->m_LU.col_indices[0] ),
                thrust::raw_pointer_cast( &this->m_A_to_LU_mapping[0] ),
                thrust::raw_pointer_cast( &returnValue[0] ));
        }
        else
        {
            computeAtoLUmapping_kernel<CtaSize, SMemSize2> <<< GridSize, CtaSize >>> (
                m_A.get_num_rows( ),
                thrust::raw_pointer_cast( &m_A.row_offsets[0] ),
                thrust::raw_pointer_cast( &m_A.col_indices[0] ),
                thrust::raw_pointer_cast( &this->m_LU.row_offsets[0] ),
                thrust::raw_pointer_cast( &this->m_LU.col_indices[0] ),
                thrust::raw_pointer_cast( &this->m_A_to_LU_mapping[0] ),
                thrust::raw_pointer_cast( &returnValue[0] ));
        }

        cudaCheckError();
    }

    if (returnValue[0] == 1)
    {
        FatalError( "Number of nonzeros per row exceeds allocated shared memory", AMGX_ERR_NO_MEMORY);
    }
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void MulticolorILUSolver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::fillLUValuesWithAValues()
{
    FatalError("Haven't implemented Multicolor ILU smoother for host format", AMGX_ERR_NOT_SUPPORTED_TARGET);
}


template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void MulticolorILUSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::fillLUValuesWithAValues()
{
    if (this->m_sparsity_level == 0)
    {
        this->m_LU.values = this->m_explicit_A->values;
    }
    else
    {
        // TODO: Should probably store the inverse mapping of AtoLUmapping instead
        //       This will allow to use unpermuteVector and have coalesced writes
        //       instead of coalesced reads
        thrust::fill(this->m_LU.values.begin(), this->m_LU.values.end(), 0.);
        cudaCheckError();

        if (this->m_explicit_A->hasProps(DIAG))
        {
            amgx::permuteVector(this->m_explicit_A->values, this->m_LU.values, this->m_A_to_LU_mapping, (this->m_explicit_A->get_num_nz() + this->m_explicit_A->get_num_rows())*this->m_explicit_A->get_block_size());
        }
        else
        {
            amgx::permuteVector(this->m_explicit_A->values, this->m_LU.values, this->m_A_to_LU_mapping, this->m_explicit_A->get_num_nz()*this->m_explicit_A->get_block_size());
        }
    }
}


template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void MulticolorILUSolver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::computeLUSparsityPattern()
{
    FatalError("Haven't implemented Multicolor ILU smoother for host format", AMGX_ERR_NOT_SUPPORTED_TARGET);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void MulticolorILUSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::computeLUSparsityPattern()
{
    // ILU0
    if (this->m_sparsity_level == 0)
    {
        // Copy everything except the values
        this->m_LU.copy_structure(*this->m_explicit_A);
    }
    // ILU1
    else if (this->m_sparsity_level == 1)
    {
        this->sparsity_wk = CSR_Multiply<TConfig_d>::csr_workspace_create( *this->m_cfg, "default" );
        CSR_Multiply<TConfig_d>::csr_sparsity_ilu1( *this->m_explicit_A, this->m_LU, this->sparsity_wk );
        CSR_Multiply<TConfig_d>::csr_workspace_delete( this->sparsity_wk );

        if (this->m_use_bsrxmv)
        {
            this->m_LU.set_initialized(0);
            this->m_LU.computeDiagonal();
            this->m_LU.set_initialized(1);
        }

        this->m_LU.setMatrixColoring(&(this->m_explicit_A->getMatrixColoring()));
    }
    else
    {
        FatalError("Haven't implemented Multicolor ILU smoother for this sparsity level. ", AMGX_ERR_NOT_IMPLEMENTED);
    }
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void MulticolorILUSolver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::computeLUFactors()
{
    FatalError("Haven't implemented Multicolor ILU smoother for host format", AMGX_ERR_NOT_SUPPORTED_TARGET);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void MulticolorILUSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::computeLUFactors()
{
    const int CtaSize = 128; // Number of threads per CTA
    const int SMemSize = 128;
    const int nWarps = CtaSize / 32;
    device_vector_alloc<int> returnValue(1);
    returnValue[0] = 0;
    int num_colors = this->m_LU.getMatrixColoring().getNumColors();
    const IndexType *LU_sorted_rows_by_color_ptr = this->m_LU.getMatrixColoring().getSortedRowsByColor().raw();

    for (int i = 0; i < num_colors; i++)
    {
        const IndexType color_offset = this->m_LU.getMatrixColoring().getOffsetsRowsPerColor()[i];
        const IndexType num_rows_per_color = this->m_LU.getMatrixColoring().getOffsetsRowsPerColor()[i + 1] - color_offset;
#ifdef EXPERIMENTAL_LU_FACTORS
        int GridSize = std::min( 2048, ( num_rows_per_color + nWarps - 1 ) / nWarps );

        if ( GridSize == 0 )
        {
            continue;    // if perfect coloring (color 0 has no vertices)
        }
    
        if ( this->m_LU.get_block_dimx() == 4 && this->m_LU.get_block_dimy() == 4 )
        {
            if ( this->m_explicit_A->getBlockFormat() == ROW_MAJOR )
            {
                compute_LU_factors_4x4_kernel_warp<ValueTypeA, CtaSize, SMemSize, true> <<< GridSize, CtaSize>>>(
                    this->m_LU.get_num_rows( ),
                    thrust::raw_pointer_cast( &this->m_LU.row_offsets[0] ),
                    thrust::raw_pointer_cast( &this->m_LU.col_indices[0] ),
                    thrust::raw_pointer_cast( &this->m_LU.diag[0] ),
                    thrust::raw_pointer_cast( &this->m_LU.values[0] ),
                    thrust::raw_pointer_cast( &this->m_LU.m_smaller_color_offsets[0] ),
                    thrust::raw_pointer_cast( &this->m_LU.m_larger_color_offsets[0] ),
                    LU_sorted_rows_by_color_ptr + color_offset,
                    num_rows_per_color,
                    i,
                    thrust::raw_pointer_cast( &returnValue[0] ) );
            }
            else
            {
                compute_LU_factors_4x4_kernel_warp<ValueTypeA, CtaSize, SMemSize, false> <<< GridSize, CtaSize>>>(
                    this->m_LU.get_num_rows( ),
                    thrust::raw_pointer_cast( &this->m_LU.row_offsets[0] ),
                    thrust::raw_pointer_cast( &this->m_LU.col_indices[0] ),
                    thrust::raw_pointer_cast( &this->m_LU.diag[0] ),
                    thrust::raw_pointer_cast( &this->m_LU.values[0] ),
                    thrust::raw_pointer_cast( &this->m_LU.m_smaller_color_offsets[0] ),
                    thrust::raw_pointer_cast( &this->m_LU.m_larger_color_offsets[0] ),
                    LU_sorted_rows_by_color_ptr + color_offset,
                    num_rows_per_color,
                    i,
                    thrust::raw_pointer_cast( &returnValue[0] ) );
            }

            cudaCheckError();
        }
        else if( this->m_LU.get_block_dimx() == 1 && this->m_LU.get_block_dimy() == 1 )
        {
            if ( this->m_explicit_A->getBlockFormat() == ROW_MAJOR )
            {
                compute_LU_factors_1x1_kernel_warp<ValueTypeA, CtaSize, SMemSize, true> <<< GridSize, CtaSize>>>(
                    this->m_LU.get_num_rows( ),
                    thrust::raw_pointer_cast( &this->m_LU.row_offsets[0] ),
                    thrust::raw_pointer_cast( &this->m_LU.col_indices[0] ),
                    thrust::raw_pointer_cast( &this->m_LU.diag[0] ),
                    thrust::raw_pointer_cast( &this->m_LU.values[0] ),
                    thrust::raw_pointer_cast( &this->m_LU.m_smaller_color_offsets[0] ),
                    thrust::raw_pointer_cast( &this->m_LU.m_larger_color_offsets[0] ),
                    LU_sorted_rows_by_color_ptr + color_offset,
                    num_rows_per_color,
                    i,
                    thrust::raw_pointer_cast( &returnValue[0] ) );

                    // if(i == num_colors-1){
                    //     const char* filename = "LU_3x3";
                    //     printMatrix< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> > PM;
                    //     PM.print(this->m_LU,filename);
                    //     FatalError("Unsupported 3x3 COL_MAJOR block for Multicolor ILU solver, computeLUFactors", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
                    // }

            }
            else
            {
            FatalError("Unsupported 3x3 COL_MAJOR block for Multicolor ILU solver, computeLUFactors", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
            }

            cudaCheckError();
        }
        else if( this->m_LU.get_block_dimx() == 2 && this->m_LU.get_block_dimy() == 2 )
        {
            if ( this->m_explicit_A->getBlockFormat() == ROW_MAJOR )
            {
                compute_LU_factors_2x2_kernel_warp<ValueTypeA, CtaSize, SMemSize, true> <<< GridSize, CtaSize>>>(
                    this->m_LU.get_num_rows( ),
                    thrust::raw_pointer_cast( &this->m_LU.row_offsets[0] ),
                    thrust::raw_pointer_cast( &this->m_LU.col_indices[0] ),
                    thrust::raw_pointer_cast( &this->m_LU.diag[0] ),
                    thrust::raw_pointer_cast( &this->m_LU.values[0] ),
                    thrust::raw_pointer_cast( &this->m_LU.m_smaller_color_offsets[0] ),
                    thrust::raw_pointer_cast( &this->m_LU.m_larger_color_offsets[0] ),
                    LU_sorted_rows_by_color_ptr + color_offset,
                    num_rows_per_color,
                    i,
                    thrust::raw_pointer_cast( &returnValue[0] ) );

                    // if(i == num_colors-1){
                    //     const char* filename = "LU_3x3";
                    //     printMatrix< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> > PM;
                    //     PM.print(this->m_LU,filename);
                    //     FatalError("Unsupported 3x3 COL_MAJOR block for Multicolor ILU solver, computeLUFactors", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
                    // }

            }
            else
            {
            FatalError("Unsupported 3x3 COL_MAJOR block for Multicolor ILU solver, computeLUFactors", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
            }

            cudaCheckError();
        }
        else if( this->m_LU.get_block_dimx() == 3 && this->m_LU.get_block_dimy() == 3 )
        {
            if ( this->m_explicit_A->getBlockFormat() == ROW_MAJOR )
            {
                // if( i == 0 )
                // {
                //     time_t start,stop;
                //     start = time(NULL);
                //     const char* filename1 = "./debugdata/row_ptr.dat";
                //     const char* filename2 = "./debugdata/col_ind.dat";
                //     const char* filename3 = "./debugdata/value.dat";
                //     const char* filename4 = "./debugdata/diag.dat"; 
                //     const char* filename5 = "./debugdata/smaller_color.dat";
                //     const char* filename6 = "./debugdata/larger_color.dat";
                //     const char* filename7 = "./debugdata/sorted_rows.dat";
                //     const char* filename8 = "./debugdata/offset_rows_per_color.dat";
                //     const char* filename9 = "./debugdata/color_X_cols.dat";

                //     writeVectorBinary(filename1,this->m_LU.row_offsets);
                //     printf("1\n");
                //     writeVectorBinary(filename2,this->m_LU.col_indices);
                //     printf("2\n");
                //     writeVectorBinary(filename3,this->m_LU.values);
                //     printf("3\n");
                //     writeVectorBinary(filename4,this->m_LU.diag);
                //     printf("4\n");
                //     writeVectorBinary(filename5,this->m_LU.m_smaller_color_offsets);
                //     printf("5\n");
                //     writeVectorBinary(filename6,this->m_LU.m_larger_color_offsets);
                //     printf("6\n");
                //     writeVectorBinary(filename7,this->m_LU.getMatrixColoring().getSortedRowsByColor());
                //     printf("7\n");
                //     writeVectorBinary(filename8,this->m_LU.getMatrixColoring().getOffsetsRowsPerColor());
                //     printf("8\n");
                //     // writeVectorBycolor(filename9,this->m_LU.values,this->m_LU.row_offsets,this->m_LU.getMatrixColoring().getSortedRowsByColor(),color_offset,num_rows_per_color,i,3);
                //     printf("9\n");
                //     stop = time(NULL);
                //     printf("Use time %ld\n",stop - start);

                // }


                compute_LU_factors_3x3_kernel_warp<ValueTypeA, CtaSize, SMemSize, true> <<< GridSize, CtaSize>>>(
                    this->m_LU.get_num_rows( ),
                    thrust::raw_pointer_cast( &this->m_LU.row_offsets[0] ),
                    thrust::raw_pointer_cast( &this->m_LU.col_indices[0] ),
                    thrust::raw_pointer_cast( &this->m_LU.diag[0] ),
                    thrust::raw_pointer_cast( &this->m_LU.values[0] ),
                    thrust::raw_pointer_cast( &this->m_LU.m_smaller_color_offsets[0] ),
                    thrust::raw_pointer_cast( &this->m_LU.m_larger_color_offsets[0] ),
                    LU_sorted_rows_by_color_ptr + color_offset,
                    num_rows_per_color,
                    i,
                    thrust::raw_pointer_cast( &returnValue[0] ) );

                    // if(i == 0 ){
                    //     printf("color all: %d color %dth has %d rows\n",num_colors,i,num_rows_per_color);
                    //      const char* filename = "./LU_corelink.dat";
                    // //      const char* filename_2 = "./debugdata/LU_corelink_color0.dat";
                    //     //  printMatrix< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> > PMs;
                    //     //  PMs.print(this->m_LU,filename);
                    //     //  writeVectorBinary(filename,this->m_LU.values);
                    //     writeVectorBycolor(filename,this->m_LU.values,this->m_LU.row_offsets,this->m_LU.getMatrixColoring().getSortedRowsByColor(),color_offset,num_rows_per_color,i,3);
                    //     FatalError("Unsupported 3x3 COL_MAJOR block for Multicolor ILU solver, computeLUFactors", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
                    // }

            
            }
            else
            {
            FatalError("Unsupported 3x3 COL_MAJOR block for Multicolor ILU solver, computeLUFactors", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
            }

            cudaCheckError();
        }
        else if( this->m_LU.get_block_dimx() == 5 && this->m_LU.get_block_dimy() == 5 )
        {
            if ( this->m_explicit_A->getBlockFormat() == ROW_MAJOR )
            {
                // if( i == 0 )
                // {
                //     time_t start,stop;
                //     start = time(NULL);
                //     const char* filename1 = "./debugdata/row_ptr.dat";
                //     const char* filename2 = "./debugdata/col_ind.dat";
                //     const char* filename3 = "./debugdata/value.dat";
                //     const char* filename4 = "./debugdata/diag.dat"; 
                //     const char* filename5 = "./debugdata/smaller_color.dat";
                //     const char* filename6 = "./debugdata/larger_color.dat";
                //     const char* filename7 = "./debugdata/sorted_rows.dat";
                //     const char* filename8 = "./debugdata/offset_rows_per_color.dat";
                //     const char* filename9 = "./debugdata/color_X_cols.dat";

                //     writeVectorBinary(filename1,this->m_LU.row_offsets);
                //     printf("1\n");
                //     writeVectorBinary(filename2,this->m_LU.col_indices);
                //     printf("2\n");
                //     writeVectorBinary(filename3,this->m_LU.values);
                //     printf("3\n");
                //     writeVectorBinary(filename4,this->m_LU.diag);
                //     printf("4\n");
                //     writeVectorBinary(filename5,this->m_LU.m_smaller_color_offsets);
                //     printf("5\n");
                //     writeVectorBinary(filename6,this->m_LU.m_larger_color_offsets);
                //     printf("6\n");
                //     writeVectorBinary(filename7,this->m_LU.getMatrixColoring().getSortedRowsByColor());
                //     printf("7\n");
                //     writeVectorBinary(filename8,this->m_LU.getMatrixColoring().getOffsetsRowsPerColor());
                //     printf("8\n");
                //     // writeVectorBycolor(filename9,this->m_LU.values,this->m_LU.row_offsets,this->m_LU.getMatrixColoring().getSortedRowsByColor(),color_offset,num_rows_per_color,i,5);
                //     printf("9\n");
                //     stop = time(NULL);
                //     printf("Use time %ld\n",stop - start);

                // }


                compute_LU_factors_5x5_kernel_warp<ValueTypeA, CtaSize, SMemSize, true> <<< GridSize, CtaSize>>>(
                    this->m_LU.get_num_rows( ),
                    thrust::raw_pointer_cast( &this->m_LU.row_offsets[0] ),
                    thrust::raw_pointer_cast( &this->m_LU.col_indices[0] ),
                    thrust::raw_pointer_cast( &this->m_LU.diag[0] ),
                    thrust::raw_pointer_cast( &this->m_LU.values[0] ),
                    thrust::raw_pointer_cast( &this->m_LU.m_smaller_color_offsets[0] ),
                    thrust::raw_pointer_cast( &this->m_LU.m_larger_color_offsets[0] ),
                    LU_sorted_rows_by_color_ptr + color_offset,
                    num_rows_per_color,
                    i,
                    this->m_LU.get_block_dimx(),
                    thrust::raw_pointer_cast( &returnValue[0] ) );
 
                // if(i == num_colors - 1 ){
                //     printf("color all: %d color %dth has %d rows\n",num_colors,i,num_rows_per_color);
                //     const char* filename = "./debugdata/value_complete.dat";
                //     writeVectorBinary(filename,this->m_LU.values);
                //     // writeVectorBycolor(filename,this->m_LU.values,this->m_LU.row_offsets,this->m_LU.getMatrixColoring().getSortedRowsByColor(),color_offset,num_rows_per_color,i,5);
                //     FatalError("Unsupported 5x5 COL_MAJOR block for Multicolor ILU solver, computeLUFactors", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
                // }
                    
            }
            else
            {
            FatalError("Unsupported 5x5 COL_MAJOR block for Multicolor ILU solver, computeLUFactors", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
            }

            cudaCheckError();
        }
        else if( this->m_LU.get_block_dimx() > 5 && this->m_LU.get_block_dimy() > 5 && this->m_LU.get_block_dimx() <= 8 && this->m_LU.get_block_dimy() <=  8 )
        {
            if ( this->m_explicit_A->getBlockFormat() == ROW_MAJOR )
            {
                int GridSize = std::min( 2048,  num_rows_per_color );

                if ( GridSize == 0 )
                {
                    continue;    // if perfect coloring (color 0 has no vertices)
                }

                compute_LU_factors_6_8x6_8_kernel_warp<ValueTypeA, CtaSize, SMemSize, true> <<< GridSize, CtaSize>>>(
                    this->m_LU.get_num_rows( ),
                    thrust::raw_pointer_cast( &this->m_LU.row_offsets[0] ),
                    thrust::raw_pointer_cast( &this->m_LU.col_indices[0] ),
                    thrust::raw_pointer_cast( &this->m_LU.diag[0] ),
                    thrust::raw_pointer_cast( &this->m_LU.values[0] ),
                    thrust::raw_pointer_cast( &this->m_LU.m_smaller_color_offsets[0] ),
                    thrust::raw_pointer_cast( &this->m_LU.m_larger_color_offsets[0] ),
                    LU_sorted_rows_by_color_ptr + color_offset,
                    num_rows_per_color,
                    i,
                    this->m_LU.get_block_dimx(),
                    thrust::raw_pointer_cast( &returnValue[0] ) );

                    // if(i == 1){
                    //     const char* filename = "LU_6x6";
                    //     printMatrix< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> > PM;
                    //     PM.print(this->m_LU,filename);
                    //     FatalError("Unsupported 3x3 COL_MAJOR block for Multicolor ILU solver, computeLUFactors", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
                    // }

            }
            else
            {
            FatalError("Unsupported 6～8x6～8 COL_MAJOR block for Multicolor ILU solver, computeLUFactors", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
            }

            cudaCheckError();
        }
        else
        {
            FatalError("Unsupported block size for Multicolor ILU solver, computeLUFactors", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
        }

#else
        int GridSize = std::min( AMGX_GRID_MAX_SIZE, ( num_rows_per_color + nWarps - 1 ) / nWarps );

        if ( GridSize == 0 )
        {
            continue;    // if perfect coloring (color 0 has no vertices)
        }

        if ( this->m_LU.get_block_dimx() == 4 && this->m_LU.get_block_dimy() == 4 )
        {
            //computeLUFactors_4x4_kernel<ValueTypeA,CtaSize,SMemSize> <<< GridSize, CtaSize>>> (
            if (this->m_explicit_A->getBlockFormat() == ROW_MAJOR)
            {
                computeLUFactors_4x4_kernel<ValueTypeA, CtaSize, SMemSize, true> <<< GridSize, CtaSize>>> (
                    this->m_LU.get_num_rows( ),
                    thrust::raw_pointer_cast( &this->m_LU.row_offsets[0] ),
                    thrust::raw_pointer_cast( &this->m_LU.col_indices[0] ),
                    thrust::raw_pointer_cast( &this->m_LU.diag[0] ),
                    thrust::raw_pointer_cast( &this->m_LU.values[0] ),
                    thrust::raw_pointer_cast( &this->m_LU.m_smaller_color_offsets[0] ),
                    thrust::raw_pointer_cast( &this->m_LU.m_larger_color_offsets[0] ),
                    LU_sorted_rows_by_color_ptr + color_offset,
                    num_rows_per_color,
                    i,
                    thrust::raw_pointer_cast( &returnValue[0] ) );
            }
            else
            {
                computeLUFactors_4x4_kernel<ValueTypeA, CtaSize, SMemSize, false> <<< GridSize, CtaSize>>> (
                    this->m_LU.get_num_rows( ),
                    thrust::raw_pointer_cast( &this->m_LU.row_offsets[0] ),
                    thrust::raw_pointer_cast( &this->m_LU.col_indices[0] ),
                    thrust::raw_pointer_cast( &this->m_LU.diag[0] ),
                    thrust::raw_pointer_cast( &this->m_LU.values[0] ),
                    thrust::raw_pointer_cast( &this->m_LU.m_smaller_color_offsets[0] ),
                    thrust::raw_pointer_cast( &this->m_LU.m_larger_color_offsets[0] ),
                    LU_sorted_rows_by_color_ptr + color_offset,
                    num_rows_per_color,
                    i,
                    thrust::raw_pointer_cast( &returnValue[0] ) );
            }

            cudaCheckError();
        }
        else
        {
            FatalError("Unsupported block size for Multicolor ILU solver, computeLUFactors", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
        }

#endif
    }

    // Check returnValue flag
    if ( returnValue[0] == 1 )
    {
        FatalError( "Number of nonzeros per row exceeds allocated shared memory", AMGX_ERR_NO_MEMORY);
    }
}


// Solver pre-setup
template<class T_Config>
void
MulticolorILUSolver_Base<T_Config>::pre_setup()
{
    // Check if matrix is colored
    if (this->m_explicit_A->getColoringLevel() < m_sparsity_level + 1)
    {
        FatalError("Matrix must be colored with coloring_level > sparsity_level for the multicolorILUsolver", AMGX_ERR_CONFIGURATION);
    }
    printf("~~~~~~~~~~~~~~~~~~get color level %d m_sparsity_level %d color %d~~~~~~~~~~~~~~~~~~~~~~~\n",this->m_explicit_A->getColoringLevel(),m_sparsity_level,this->m_explicit_A->getMatrixColoring().getNumColors());
    // Compute extended sparsity pattern based on coloring and matrix A
    computeLUSparsityPattern();

    if (this->m_LU.hasProps(DIAG))
    {
        FatalError("Multicolor ILU smoother does not support outside diagonal. Try setting reorder_cols_by_color=1 and insert_diag_while_reordering=1 in the multicolor_ilu solver scope in configuration file", AMGX_ERR_NOT_IMPLEMENTED);
    }

    if (m_sparsity_level == 0 && !this->m_LU.getColsReorderedByColor())
    {
        FatalError("Multicolor ILU smoother requires matrix to be reordered by color with ILU0 solver. Try setting reorder_cols_by_color=1 and insert_diag_while_reordering=1 in the multicolor_ilu solver scope in configuration file", AMGX_ERR_NOT_IMPLEMENTED);
    }

    // Reorder the columns of LU by color
    if (m_sparsity_level != 0)
    {
        // Reorder columns of LU by color
        m_LU.reorderColumnsByColor(false);

        // Compute mapping between entries in A and entries in LU
        if (this->m_explicit_A->hasProps(DIAG))
        {
            m_A_to_LU_mapping.resize(this->m_explicit_A->get_num_nz() + this->m_explicit_A->get_num_rows());
        }
        else
        {
            m_A_to_LU_mapping.resize(this->m_explicit_A->get_num_nz());
        }

        computeAtoLUmapping();
    }

    int N = this->m_LU.get_num_rows() * this->m_LU.get_block_dimy();
    m_delta.resize(N);
    m_Delta.resize(N);
    m_Delta.set_block_dimy(this->m_explicit_A->get_block_dimy());
    m_Delta.set_block_dimx(1);
    m_delta.set_block_dimy(this->m_explicit_A->get_block_dimy());
    m_delta.set_block_dimx(1);
}

template<class T_Config>
void
MulticolorILUSolver_Base<T_Config>::printSolverParameters() const
{
    std::cout << "relaxation_factor = " << this->m_weight << std::endl;
    std::cout << "use_bsrxmv = " << this->m_use_bsrxmv << std::endl;
    std::cout << "ilu_sparsity_level = " << this->m_sparsity_level <<  std::endl;
}


// Solver setup
template<class T_Config>
void
MulticolorILUSolver_Base<T_Config>::solver_setup(bool reuse_matrix_structure)
{
    this->m_explicit_A = dynamic_cast<Matrix<T_Config>*>(this->m_A);

    if (!this->m_explicit_A)
    {
        FatalError("MulticolorILUSolver only works with explicit matrices", AMGX_ERR_INTERNAL);
    }

    if (this->m_explicit_A->getColoringLevel() < 1)
    {
        FatalError("Matrix must be colored to use multicolor ilu solver. Try setting: coloring_level=1 or coloring_level=2 in the configuration file", AMGX_ERR_NOT_IMPLEMENTED);
    }

    if (!reuse_matrix_structure)
    {
        this->pre_setup();
    }

    // Fill LU sparsity pattern
    fillLUValuesWithAValues();
    // Compute LU factors in place (update LU.values)
    computeLUFactors();
}

//
template<class T_Config>
void
MulticolorILUSolver_Base<T_Config>::solve_init( VVector &b, VVector &x, bool xIsZero )
{}

// Solve one iteration
template<class T_Config>
bool
MulticolorILUSolver_Base<T_Config>::solve_iteration( VVector &b, VVector &x, bool xIsZero )
{
    if ( !m_use_bsrxmv && (this->m_LU.get_block_dimx() == 4 && this->m_LU.get_block_dimy() == 4) )
    {
        smooth_4x4(b, x, xIsZero);
    }
    else if ( !m_use_bsrxmv && (this->m_LU.get_block_dimx() == 5 && this->m_LU.get_block_dimy() == 5) )
    {
        // smooth_5x5(b, x, xIsZero);
        smooth_bxb(b, x, xIsZero);
        // printf("smooth_5x5 over\n");
    }
    else if ( !m_use_bsrxmv && (this->m_LU.get_block_dimx() == 3 && this->m_LU.get_block_dimy() == 3) )
    {
        // smooth_3x3(b, x, xIsZero);
        smooth_bxb(b, x, xIsZero);
    }
    else
    {
        smooth_bxb(b, x, xIsZero);
    }

    // Do we converge ?
    return this->converged(b, x);
}

template<class T_Config>
void
MulticolorILUSolver_Base<T_Config>::solve_finalize( VVector &b, VVector &x )
{}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void MulticolorILUSolver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::smooth_4x4(const VVector &b, VVector &x, bool xIsZero)
{
    FatalError("Haven't implemented Multicolor DILU smoother for host format", AMGX_ERR_NOT_SUPPORTED_TARGET);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void MulticolorILUSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::smooth_4x4(const VVector &b, VVector &x, bool xIsZero)
{
    Matrix<TConfig_d> &m_LU = this->m_LU;
    Matrix<TConfig_d> &m_A = *this->m_explicit_A;
    int N = m_LU.get_num_rows() * m_LU.get_block_dimy();
    cudaCheckError();

    if (!m_LU.getColsReorderedByColor())
    {
        FatalError("ILU solver currently only works if columns are reordered by color. Try setting reordering_cols_by_color=1 in the multicolor_ilu solver scope in the configuration file", AMGX_ERR_NOT_IMPLEMENTED);
    }

    // ---------------------------------------------------------
    // Solving Lower triangular system, with identity diagonal
    // ---------------------------------------------------------
    const IndexType *LU_sorted_rows_by_color_ptr = m_LU.getMatrixColoring().getSortedRowsByColor().raw();
    int num_colors = this->m_LU.getMatrixColoring().getNumColors();

    for (int i = 0; i < num_colors; i++)
    {
        const IndexType color_offset = m_LU.getMatrixColoring().getOffsetsRowsPerColor()[i];
        const IndexType num_rows_per_color = m_LU.getMatrixColoring().getOffsetsRowsPerColor()[i + 1] - color_offset;
#ifdef EXPERIMENTAL_LU_FORWARD
        const int CtaSize = 128; // Number of threads per CTA
        const int nHalfWarps = CtaSize / 16;
        int GridSize = std::min( 2048, ( num_rows_per_color + nHalfWarps - 1 ) / nHalfWarps );

        if ( GridSize == 0 )
        {
            continue;    // if perfect coloring (color 0 has no vertices)
        }

        if ( this->m_explicit_A->getBlockFormat() == ROW_MAJOR )
        {
            if (m_A.hasProps(DIAG))
            {
                LU_forward_4x4_kernel_warp<IndexType, ValueTypeA, ValueTypeB, CtaSize, 4, true, true> <<< GridSize, CtaSize>>>(
                    m_LU.row_offsets.raw(),
                    m_LU.m_smaller_color_offsets.raw(),
                    m_LU.col_indices.raw(),
                    m_LU.values.raw(),
                    m_A.row_offsets.raw(),
                    m_A.col_indices.raw(),
                    m_A.values.raw(),
                    m_A.diag.raw(),
                    x.raw(),
                    b.raw(),
                    this->m_delta.raw(),
                    LU_sorted_rows_by_color_ptr + color_offset,
                    num_rows_per_color,
                    i,
                    xIsZero );
            }
            else
            {
                LU_forward_4x4_kernel_warp<IndexType, ValueTypeA, ValueTypeB, CtaSize, 4, true, false> <<< GridSize, CtaSize>>>(
                    m_LU.row_offsets.raw(),
                    m_LU.m_smaller_color_offsets.raw(),
                    m_LU.col_indices.raw(),
                    m_LU.values.raw(),
                    m_A.row_offsets.raw(),
                    m_A.col_indices.raw(),
                    m_A.values.raw(),
                    m_A.diag.raw(),
                    x.raw(),
                    b.raw(),
                    this->m_delta.raw(),
                    LU_sorted_rows_by_color_ptr + color_offset,
                    num_rows_per_color,
                    i,
                    xIsZero );
            }
        }
        else
        {
            // COL_MAJOR
            if (m_A.hasProps(DIAG))
            {
                LU_forward_4x4_kernel_warp<IndexType, ValueTypeA, ValueTypeB, CtaSize, 4, false, true> <<< GridSize, CtaSize>>>(
                    m_LU.row_offsets.raw(),
                    m_LU.m_smaller_color_offsets.raw(),
                    m_LU.col_indices.raw(),
                    m_LU.values.raw(),
                    m_A.row_offsets.raw(),
                    m_A.col_indices.raw(),
                    m_A.values.raw(),
                    m_A.diag.raw(),
                    x.raw(),
                    b.raw(),
                    this->m_delta.raw(),
                    LU_sorted_rows_by_color_ptr + color_offset,
                    num_rows_per_color,
                    i,
                    xIsZero );
            }
            else
            {
                LU_forward_4x4_kernel_warp<IndexType, ValueTypeA, ValueTypeB, CtaSize, 4, false, false> <<< GridSize, CtaSize>>>(
                    m_LU.row_offsets.raw(),
                    m_LU.m_smaller_color_offsets.raw(),
                    m_LU.col_indices.raw(),
                    m_LU.values.raw(),
                    m_A.row_offsets.raw(),
                    m_A.col_indices.raw(),
                    m_A.values.raw(),
                    m_A.diag.raw(),
                    x.raw(),
                    b.raw(),
                    this->m_delta.raw(),
                    LU_sorted_rows_by_color_ptr + color_offset,
                    num_rows_per_color,
                    i,
                    xIsZero );
            }
        }

#else
        const int CtaSize = 128;
        const int blockrows_per_cta = CtaSize / 4;
        const int GridSize = min( AMGX_GRID_MAX_SIZE, (int) (num_rows_per_color + blockrows_per_cta - 1) / blockrows_per_cta);

        if ( GridSize == 0 )
        {
            continue;    // if perfect coloring (color 0 has no vertices)
        }

        if (this->m_explicit_A->hasProps(DIAG))
        {
            FatalError("this implementation of LU forward solve does not support A with external diagonal", AMGX_ERR_NOT_IMPLEMENTED);
        }

        if (this->m_explicit_A->getBlockFormat() == ROW_MAJOR)
        {
            LU_forward_4x4_kernel<IndexType, ValueTypeA, ValueTypeB, blockrows_per_cta, 8, 4, true> <<< GridSize, CtaSize>>>
            (m_LU.row_offsets.raw(),
             m_LU.m_smaller_color_offsets.raw(),
             m_LU.col_indices.raw(),
             m_LU.values.raw(),
             m_A.row_offsets.raw(),
             m_A.col_indices.raw(),
             m_A.values.raw(),
             x.raw(),
             b.raw(),
             delta.raw(),
             LU_sorted_rows_by_color_ptr + color_offset,
             num_rows_per_color,
             i,
             xIsZero);
        }
        else
        {
            LU_forward_4x4_kernel<IndexType, ValueTypeA, ValueTypeB, blockrows_per_cta, 8, 4, false> <<< GridSize, CtaSize>>>
            (m_LU.row_offsets.raw(),
             m_LU.m_smaller_color_offsets.raw(),
             m_LU.col_indices.raw(),
             m_LU.values.raw(),
             m_A.row_offsets.raw(),
             m_A.col_indices.raw(),
             m_A.values.raw(),
             x.raw(),
             b.raw(),
             delta.raw(),
             LU_sorted_rows_by_color_ptr + color_offset,
             num_rows_per_color,
             i,
             xIsZero);
        }

#endif
        cudaCheckError();
    }

    // --------------------
    // Backward Sweep
    // --------------------
    for (int i = num_colors - 1; i >= 0; i--)
    {
        const IndexType color_offset = m_LU.getMatrixColoring().getOffsetsRowsPerColor()[i];
        const IndexType num_rows_per_color = m_LU.getMatrixColoring().getOffsetsRowsPerColor()[i + 1] - color_offset;
#ifdef EXPERIMENTAL_LU_BACKWARD
        const int CtaSize = 128; // Number of threads per CTA
        const int nHalfWarps = CtaSize / 16;
        int GridSize = std::min( 2048, ( num_rows_per_color + nHalfWarps - 1 ) / nHalfWarps );

        if ( GridSize == 0 )
        {
            continue;    // if perfect coloring (color 0 has no vertices)
        }

        if (this->m_explicit_A->getBlockFormat() == ROW_MAJOR)
        {
            LU_backward_4x4_kernel_warp<IndexType, ValueTypeA, ValueTypeB, CtaSize, true> <<< GridSize, CtaSize>>>(
                m_LU.row_offsets.raw(),
                m_LU.m_larger_color_offsets.raw(),
                m_LU.col_indices.raw(),
                m_LU.diag.raw(),
                m_LU.values.raw(),
                this->m_delta.raw(),
                this->m_Delta.raw(),
                x.raw(),
                LU_sorted_rows_by_color_ptr + color_offset,
                num_rows_per_color,
                i,
                num_colors,
                this->m_weight,
                xIsZero);
        }
        else
        {
            LU_backward_4x4_kernel_warp<IndexType, ValueTypeA, ValueTypeB, CtaSize, false> <<< GridSize, CtaSize>>>(
                m_LU.row_offsets.raw(),
                m_LU.m_larger_color_offsets.raw(),
                m_LU.col_indices.raw(),
                m_LU.diag.raw(),
                m_LU.values.raw(),
                this->m_delta.raw(),
                this->m_Delta.raw(),
                x.raw(),
                LU_sorted_rows_by_color_ptr + color_offset,
                num_rows_per_color,
                i,
                num_colors,
                this->m_weight,
                xIsZero);
        }

#else
        const int CtaSize = 128;
        const int blockrows_per_cta = CtaSize / 4;
        const int GridSize = min( AMGX_GRID_MAX_SIZE, (int) (num_rows_per_color + blockrows_per_cta - 1) / blockrows_per_cta);

        if ( GridSize == 0 )
        {
            continue;    // if perfect coloring (color 0 has no vertices)
        }

        if (this->m_explicit_A->getBlockFormat() == ROW_MAJOR)
        {
            LU_backward_4x4_kernel<IndexType, ValueTypeA, ValueTypeB, blockrows_per_cta, 8, 4, true> <<< GridSize, CtaSize>>>
            (m_LU.row_offsets.raw(),
             m_LU.m_larger_color_offsets.raw(),
             m_LU.col_indices.raw(),
             m_LU.diag.raw(),
             m_LU.values.raw(),
             this->m_delta.raw(),
             this->m_Delta.raw(),
             x.raw(),
             LU_sorted_rows_by_color_ptr + color_offset,
             num_rows_per_color,
             i,
             num_colors,
             this->m_weight,
             xIsZero);
        }
        else
        {
            LU_backward_4x4_kernel<IndexType, ValueTypeA, ValueTypeB, blockrows_per_cta, 8, 4, false> <<< GridSize, CtaSize>>>
            (m_LU.row_offsets.raw(),
             m_LU.m_larger_color_offsets.raw(),
             m_LU.col_indices.raw(),
             m_LU.diag.raw(),
             m_LU.values.raw(),
             this->m_delta.raw(),
             this->m_Delta.raw(),
             x.raw(),
             LU_sorted_rows_by_color_ptr + color_offset,
             num_rows_per_color,
             i,
             num_colors,
             this->m_weight,
             xIsZero);
        }

#endif
    }

    cudaCheckError();
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void MulticolorILUSolver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::smooth_3x3(const VVector &b, VVector &x, bool xIsZero)
{
    FatalError("Haven't implemented Multicolor smooth3x3 smoother for host format", AMGX_ERR_NOT_SUPPORTED_TARGET);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void MulticolorILUSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::smooth_3x3(const VVector &b, VVector &x, bool xIsZero)
{
    Matrix<TConfig_d> &m_LU = this->m_LU;
    Matrix<TConfig_d> &m_A = *this->m_explicit_A;
    int N = m_LU.get_num_rows() * m_LU.get_block_dimy();
    cudaCheckError();

    if (!m_LU.getColsReorderedByColor())
    {
        FatalError("ILU solver currently only works if columns are reordered by color. Try setting reordering_cols_by_color=1 in the multicolor_ilu solver scope in the configuration file", AMGX_ERR_NOT_IMPLEMENTED);
    }

    // ---------------------------------------------------------
    // Solving Lower triangular system, with identity diagonal
    // ---------------------------------------------------------
    const IndexType *LU_sorted_rows_by_color_ptr = m_LU.getMatrixColoring().getSortedRowsByColor().raw();
    int num_colors = this->m_LU.getMatrixColoring().getNumColors();
    for (int i = 0; i < num_colors; i++)
    {
        const IndexType color_offset = m_LU.getMatrixColoring().getOffsetsRowsPerColor()[i];
        const IndexType num_rows_per_color = m_LU.getMatrixColoring().getOffsetsRowsPerColor()[i + 1] - color_offset;
#ifdef EXPERIMENTAL_LU_FORWARD
        const int CtaSize = 128; // Number of threads per CTA
        const int nHalfWarps = CtaSize / 16;
        int GridSize = std::min( 2048, ( num_rows_per_color + nHalfWarps - 1 ) / nHalfWarps );
        
        if ( GridSize == 0 )
        {
            continue;    // if perfect coloring (color 0 has no vertices)
        }

        if ( this->m_explicit_A->getBlockFormat() == ROW_MAJOR )
        {
            if (m_A.hasProps(DIAG))
            {
               FatalError("Haven't implemented compute LU_forward_3x3 for matrix has diag props", AMGX_ERR_NOT_SUPPORTED_TARGET);
            }
            else
            {
                    LU_forward_3x3_kernel_warp<IndexType, ValueTypeA, ValueTypeB, CtaSize, 4, true, false> <<< GridSize, CtaSize>>>(
                    m_LU.row_offsets.raw(),
                    m_LU.m_smaller_color_offsets.raw(),
                    m_LU.col_indices.raw(),
                    m_LU.values.raw(),
                    m_A.row_offsets.raw(),
                    m_A.col_indices.raw(),
                    m_A.values.raw(),
                    m_A.diag.raw(),
                    x.raw(),
                    b.raw(),
                    this->m_delta.raw(),
                    LU_sorted_rows_by_color_ptr + color_offset,
                    num_rows_per_color,
                    i,
                    xIsZero );
           }
        }
        else
        {
            // COL_MAJOR
            FatalError("Haven't implemented compute LU_forward_3x3 for Col Major", AMGX_ERR_NOT_SUPPORTED_TARGET);
        }

#else
            FatalError("Haven't implemented compute LU_forward_3x3 for Not EXPERIMENTAL_LU_FORWARD", AMGX_ERR_NOT_SUPPORTED_TARGET);

#endif
        cudaCheckError();
    }
    // --------------------
    // Backward Sweep
    // --------------------
    for (int i = num_colors - 1; i >= 0; i--)
    {
        const IndexType color_offset = m_LU.getMatrixColoring().getOffsetsRowsPerColor()[i];
        const IndexType num_rows_per_color = m_LU.getMatrixColoring().getOffsetsRowsPerColor()[i + 1] - color_offset;
#ifdef EXPERIMENTAL_LU_BACKWARD
        const int CtaSize = 128; // Number of threads per CTA
        const int nHalfWarps = CtaSize / 16;
        int GridSize = std::min( 2048, ( num_rows_per_color + nHalfWarps - 1 ) / nHalfWarps );

        if ( GridSize == 0 )
        {
            continue;    // if perfect coloring (color 0 has no vertices)
        }

        if (this->m_explicit_A->getBlockFormat() == ROW_MAJOR)
        {
            LU_backward_3x3_kernel_warp<IndexType, ValueTypeA, ValueTypeB, CtaSize, true> <<< GridSize, CtaSize>>>(
                m_LU.row_offsets.raw(),
                m_LU.m_larger_color_offsets.raw(),
                m_LU.col_indices.raw(),
                m_LU.diag.raw(),
                m_LU.values.raw(),
                this->m_delta.raw(),
                this->m_Delta.raw(),
                x.raw(),
                LU_sorted_rows_by_color_ptr + color_offset,
                num_rows_per_color,
                i,
                num_colors,
                this->m_weight,
                xIsZero);
                // if(i == 0){
                //     const char* filename = "v_3x3";
                //     writeVector(filename,x);   
                //     FatalError("Haven't implemented compute LU_forward_3x3 for Not EXPERIMENTAL_LU_FORWARD", AMGX_ERR_NOT_SUPPORTED_TARGET);
                // }
        }
        else
        {
            FatalError("Haven't implemented compute LU_backward_3x3 for Col Major", AMGX_ERR_NOT_SUPPORTED_TARGET);
        }

#else
        FatalError("Haven't implemented compute  LU_backward_3x3 for Not EXPERIMENTAL_LU_BACKWARD", AMGX_ERR_NOT_SUPPORTED_TARGET);
#endif
    }
    cudaCheckError();
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void MulticolorILUSolver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::smooth_5x5(const VVector &b, VVector &x, bool xIsZero)
{
    FatalError("Haven't implemented Multicolor DILU smoother for host format", AMGX_ERR_NOT_SUPPORTED_TARGET);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void MulticolorILUSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::smooth_5x5(const VVector &b, VVector &x, bool xIsZero)
{
    Matrix<TConfig_d> &m_LU = this->m_LU;
    Matrix<TConfig_d> &m_A = *this->m_explicit_A;
    int N = m_LU.get_num_rows() * m_LU.get_block_dimy();
    cudaCheckError();

    if (!m_LU.getColsReorderedByColor())
    {
        FatalError("ILU solver currently only works if columns are reordered by color. Try setting reordering_cols_by_color=1 in the multicolor_ilu solver scope in the configuration file", AMGX_ERR_NOT_IMPLEMENTED);
    }

    // ---------------------------------------------------------
    // Solving Lower triangular system, with identity diagonal
    // ---------------------------------------------------------
    const IndexType *LU_sorted_rows_by_color_ptr = m_LU.getMatrixColoring().getSortedRowsByColor().raw();
    int num_colors = this->m_LU.getMatrixColoring().getNumColors();

    for (int i = 0; i < num_colors; i++)
    {
        const IndexType color_offset = m_LU.getMatrixColoring().getOffsetsRowsPerColor()[i];
        const IndexType num_rows_per_color = m_LU.getMatrixColoring().getOffsetsRowsPerColor()[i + 1] - color_offset;
#ifdef EXPERIMENTAL_LU_FORWARD
        const int CtaSize = 128; // Number of threads per CTA
        const int nWarps = CtaSize / 32;
        int GridSize = std::min( 2048, ( num_rows_per_color + nWarps - 1 ) / nWarps );

        if ( GridSize == 0 )
        {
            continue;    // if perfect coloring (color 0 has no vertices)
        }

        if ( this->m_explicit_A->getBlockFormat() == ROW_MAJOR )
        {
            if (m_A.hasProps(DIAG))
            {
               FatalError("Haven't implemented compute LU_forward_3x3 for matrix has diag props", AMGX_ERR_NOT_SUPPORTED_TARGET);
            }
            else
            {
                if( i == 0 )
                {
                    const char* filename1 = "./debugdata/x.dat";
                    const char* filename2 = "./debugdata/b.dat";
                    const char* filename3 = "./debugdata/delta.dat";

                    writeVectorBinary(filename1,x);
                    writeVectorBinary(filename2,b);
                    writeVectorBinary(filename3,this->m_delta);
                }


                LU_forward_5x5_kernel_warp<IndexType, ValueTypeA, ValueTypeB, CtaSize, 4, true, false> <<< GridSize, CtaSize>>>(
                    m_LU.row_offsets.raw(),
                    m_LU.m_smaller_color_offsets.raw(),
                    m_LU.col_indices.raw(),
                    m_LU.values.raw(),
                    m_A.row_offsets.raw(),
                    m_A.col_indices.raw(),
                    m_A.values.raw(),
                    m_A.diag.raw(),
                    x.raw(),
                    b.raw(),
                    this->m_delta.raw(),
                    LU_sorted_rows_by_color_ptr + color_offset,
                    num_rows_per_color,
                    i,
                    xIsZero );
                
                // if(i == num_colors - 1)
                // {
                //     const char* filename4 = "./debugdata/delta_gpu_5.dat";
                //     writeVectorBinary(filename4,this->m_delta);              
                //     return ;
                //     // FatalError("Unsupported 3x3 COL_MAJOR block for Multicolor ILU solver, computeLUFactors", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
                // }
            }
        }
        else
        {
            FatalError("Haven't implemented compute LU_backward_5x5 for Col Major", AMGX_ERR_NOT_SUPPORTED_TARGET);
        }

#else
        {
           FatalError("Haven't implemented compute  LU_forward_5x5 for Col Major for Not EXPERIMENTAL_LU_BACKWARD", AMGX_ERR_NOT_SUPPORTED_TARGET);
        }
#endif
        cudaCheckError();
    }

    // --------------------
    // Backward Sweep
    // --------------------
    for (int i = num_colors - 1; i >= 0; i--)
    {
        const IndexType color_offset = m_LU.getMatrixColoring().getOffsetsRowsPerColor()[i];
        const IndexType num_rows_per_color = m_LU.getMatrixColoring().getOffsetsRowsPerColor()[i + 1] - color_offset;
#ifdef EXPERIMENTAL_LU_BACKWARD
        const int CtaSize = 128; // Number of threads per CTA
        const int nWarps = CtaSize / 32;
        int GridSize = std::min( 2048, ( num_rows_per_color + nWarps - 1 ) / nWarps );

        if ( GridSize == 0 )
        {
            continue;    // if perfect coloring (color 0 has no vertices)
        }

        if (this->m_explicit_A->getBlockFormat() == ROW_MAJOR)
        {
                if( i == num_colors - 1 )
                {
                    const char* filename3 = "./debugdata/Delta.dat";

                    writeVectorBinary(filename3,this->m_Delta);
                    // ValueTypeB m_weight
                    const char* filename = "./debugdata/weight.dat";
                    writeBinary(filename,this->m_weight);

                }

            LU_backward_5x5_kernel_warp<IndexType, ValueTypeA, ValueTypeB, CtaSize, true> <<< GridSize, CtaSize>>>(
                m_LU.row_offsets.raw(),
                m_LU.m_larger_color_offsets.raw(),
                m_LU.col_indices.raw(),
                m_LU.diag.raw(),
                m_LU.values.raw(),
                this->m_delta.raw(),
                this->m_Delta.raw(),
                x.raw(),
                LU_sorted_rows_by_color_ptr + color_offset,
                num_rows_per_color,
                i,
                num_colors,
                this->m_weight,
                xIsZero);
                        
            printf("backward %d has processed\n",i);

            // if (i == num_colors -1  )
            if (i == 0 )
            {
                const char* filename = "./debugdata/x_gpu_5.dat";
                writeVectorBinary(filename,this->m_Delta);
                return;
                // FatalError("Unsupported 3x3 COL_MAJOR block for Multicolor ILU solver, computeLUFactors", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);

            }

        }
        else
        {
           FatalError("Haven't implemented compute LU_backward_5x5 for Col Major or Not EXPERIMENTAL_LU_BACKWARD", AMGX_ERR_NOT_SUPPORTED_TARGET);
        }

#else
       {
           FatalError("Haven't implemented compute LU_backward_5x5 for Col Major or Not EXPERIMENTAL_LU_BACKWARD", AMGX_ERR_NOT_SUPPORTED_TARGET);
       }
#endif
    }

    cudaCheckError();
}




template<int CtaSize, int SMemSize, bool ROW_MAJOR, typename ValueTypeA >
__global__
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
__launch_bounds__( CtaSize, 12 )
#elif defined(__CUDA_ARCH__)
__launch_bounds__( CtaSize, 8 )
#endif
void axpy_color(const ValueTypeA *x,
                ValueTypeA *y,
                double alpha,
                const int *sorted_rows_by_color,
                const int num_rows_per_color,
                int blockDimx)
{
    // __shared__ volatile ValueTypeA s_x[SMemSize];
    // __shared__ volatile ValueTypeA s_y[SMemSize];


    for(int threadId =  blockIdx.x * blockDim.x + threadIdx.x; threadId < num_rows_per_color * blockDimx; threadId += gridDim.x * blockDim.x)
    {
        int RowIt = threadId / blockDimx;
        int VarId = threadId % blockDimx;
        int aRowId = sorted_rows_by_color[RowIt];
        y[aRowId * blockDimx + VarId] += alpha * x[aRowId * blockDimx + VarId];
    }

}


template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void MulticolorILUSolver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::smooth_bxb(const VVector &b, VVector &x, bool xIsZero)
{
    FatalError("Haven't implemented Multicolor DILU smoother for host format", AMGX_ERR_NOT_SUPPORTED_TARGET);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void MulticolorILUSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::smooth_bxb(const VVector &b, VVector &x, bool xIsZero)
{
    Matrix<TConfig_d> &m_LU = this->m_LU;
    Matrix<TConfig_d> &m_A = *this->m_explicit_A;
    int N = m_LU.get_num_rows() * m_LU.get_block_dimy();

    if (!m_LU.getColsReorderedByColor())
    {
        FatalError("ILU solver currently only works if columns are reordered by color. Try setting reorder_cols_by_color=1 in the multicolor_ilu solver scope in the configuration file", AMGX_ERR_NOT_IMPLEMENTED);
    }

    if (this->m_explicit_A->getBlockFormat() == COL_MAJOR)
    {
        FatalError("ILU solver for arbitrary block sizes only works with ROW_MAJOR matrices", AMGX_ERR_NOT_IMPLEMENTED);
    }

    // ---------------------------------------------------------
    // Solving Lower triangular system, with identity diagonal
    // ---------------------------------------------------------
    const IndexType *LU_sorted_rows_by_color_ptr = m_LU.getMatrixColoring().getSortedRowsByColor().raw();
    int num_colors = this->m_LU.getMatrixColoring().getNumColors();
    //delta = b;
    thrust::copy(b.begin(), b.end(), this->m_delta.begin());
    //delta = delta - Ax;
    Cusparse::bsrmv((ValueTypeA) - 1.0, m_A, x, (ValueTypeA)1.0, this->m_delta);
    cudaCheckError();
    // Setting Delta to zero
    thrust::fill(this->m_Delta.begin(), this->m_Delta.end(), (ValueTypeB)0.0f);
    cudaCheckError();
    bool skipped_end = false;




    for (int i = 0; i < num_colors; i++)
    {
        // if( i == 0 )
        // {
        //     const char* filename1 = "./debugdata/x.dat";
        //     const char* filename2 = "./debugdata/b.dat";
        //     const char* filename3 = "./debugdata/delta.dat";

        //     writeVectorBinary(filename1,x);
        //     writeVectorBinary(filename2,b);
        //     writeVectorBinary(filename3,this->m_delta);
        // }

        const IndexType color_offset = m_LU.getMatrixColoring().getOffsetsRowsPerColor()[i];
        const IndexType num_rows_per_color = m_LU.getMatrixColoring().getOffsetsRowsPerColor()[i + 1] - color_offset;

        if (num_rows_per_color == 0) { continue; } // if perfect coloring (color 0 has no vertices)

        if (skipped_end)
        {
            // delta = delta - LU*delta smaller colors
            Cusparse::bsrmv(Cusparse::SMALLER_COLORS, i, (ValueTypeA) - 1.0f, m_LU, this->m_delta, (ValueTypeA)1.0f, this->m_delta);
        }
        cudaCheckError();
        if (num_rows_per_color > 0)
        {
            skipped_end = true;
        }

        // if(i == num_colors - 1)
        // {
        //     const char* filename4 = "./debugdata/delta_gpu_b.dat";
        //     writeVectorBinary(filename4,this->m_delta);              
        //     // FatalError("Unsupported 3x3 COL_MAJOR block for Multicolor ILU solver, computeLUFactors", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
        // }
    }
           

    cudaCheckError();
    skipped_end = false;

    // --------------------
    // Backward Sweep
    // --------------------
    const int CtaSize = 160; // Number of threads per CTA
    const int SMemSize = 160;
    const int nRows = CtaSize / m_LU.get_block_dimx();
    for (int i = num_colors - 1; i >=  0; i--)
    {
        // if( i == num_colors - 1 )
        // {
        //     const char* filename4 = "./debugdata/Delta.dat";

        //     writeVectorBinary(filename4,this->m_Delta);
        //     // ValueTypeB m_weight
        //     const char* filename5 = "./debugdata/weight.dat";
        //     writeBinary(filename5,this->m_weight);

        // }
        const IndexType color_offset = m_LU.getMatrixColoring().getOffsetsRowsPerColor()[i];
        const IndexType num_rows_per_color = m_LU.getMatrixColoring().getOffsetsRowsPerColor()[i + 1] - color_offset;
        int GridSize = std::min( 2048, ( num_rows_per_color + nRows - 1 ) / nRows );

        if ( num_rows_per_color == 0 )  { continue;  }  // if perfect coloring (color 0 has no vertices)

        // delta = delta - LU*Delta larger colors
        if(skipped_end){
            if ( xIsZero ){
                Cusparse::bsrmv(Cusparse::LARGER_COLORS, i, (ValueTypeA) - 1.0f, m_LU, x, (ValueTypeA)1.0f, this->m_delta);
                // Cusparse::bsrmv(Cusparse::LARGER_COLORS, i, (ValueTypeA) - 1.0f, m_LU, this->m_Delta, (ValueTypeA)1.0f, this->m_delta);
            }
            else{
                Cusparse::bsrmv(Cusparse::LARGER_COLORS, i, (ValueTypeA) - 1.0f, m_LU, this->m_Delta, (ValueTypeA)1.0f, this->m_delta);
            }
        }
        cudaCheckError();    
        // Multiple by inverse stored on diagonal  Delta = delta * LU diagonal
        Cusparse::bsrmv(Cusparse::DIAG_COL, i, (ValueTypeA) 1.0f, m_LU, this->m_delta, 0.0f, this->m_Delta);
        cudaCheckError();   

        axpy_color<CtaSize, SMemSize, true> <<< GridSize, CtaSize>>>(
                    this->m_Delta.raw(),
                    x.raw(),
                    (double)this->m_weight,
                    LU_sorted_rows_by_color_ptr + color_offset,
                    num_rows_per_color,
                    m_LU.get_block_dimx());
        
        

        if (num_rows_per_color > 0)
        {
            skipped_end = true;
        }
        // printf("xIszero: %d\n",xIsZero);
        
        // if (i == 0)
        // {
        //     // const char* filename6 = "./debugdata/delta_gpu_b.dat";
        //     // writeVectorBinary(filename6,this->m_delta);
        //     const char* filename7 = "./debugdata/Delta_gpu_b.dat";
        //     writeVectorBinary(filename7,this->m_Delta);

        //     FatalError("Unsupported 3x3 COL_MAJOR block for Multicolor ILU solver, computeLUFactors", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
        // }

    }
    cudaCheckError();    
    //x = x + weight * Delta
    // axpy(this->m_Delta,x,this->m_weight,0,-1);
    // cudaCheckError();
    //     //     // if (i == 0 )
//     {
//         const char* filename6 = "./debugdata/x_gpu_b.dat";
//         writeVectorBinary(filename6,x);

//         FatalError("Unsupported 3x3 COL_MAJOR block for Multicolor ILU solver, computeLUFactors", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);

//     }
}



/****************************************
 * Explict instantiations
 ***************************************/
#define AMGX_CASE_LINE(CASE) template class MulticolorILUSolver_Base<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class MulticolorILUSolver<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

}
} // namespace amgx
