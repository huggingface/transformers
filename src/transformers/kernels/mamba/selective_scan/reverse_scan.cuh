/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cub/config.cuh>

#include <cub/util_ptx.cuh>
#include <cub/util_type.cuh>
#include <cub/block/block_raking_layout.cuh>
// #include <cub/detail/uninitialized_copy.cuh>
#include "uninitialized_copy.cuh"

/**
 * Perform a reverse sequential reduction over \p LENGTH elements of the \p input array.  The aggregate is returned.
 */
template <
    int         LENGTH,
    typename    T,
    typename    ReductionOp>
__device__ __forceinline__ T ThreadReverseReduce(const T (&input)[LENGTH], ReductionOp reduction_op) {
    static_assert(LENGTH > 0);
    T retval = input[LENGTH - 1];
    #pragma unroll
    for (int i = LENGTH - 2; i >= 0; --i) { retval = reduction_op(retval, input[i]); }
    return retval;
}

/**
 * Perform a sequential inclusive postfix reverse scan over the statically-sized \p input array, seeded with the specified \p postfix.  The aggregate is returned.
 */
template <
    int         LENGTH,
    typename    T,
    typename    ScanOp>
__device__ __forceinline__ T ThreadReverseScanInclusive(
    const T (&input)[LENGTH],
    T (&output)[LENGTH],
    ScanOp scan_op,
    const T postfix)
{
    T inclusive = postfix;
    #pragma unroll
    for (int i = LENGTH - 1; i >= 0; --i) {
        inclusive = scan_op(inclusive, input[i]);
        output[i] = inclusive;
    }
}

/**
 * Perform a sequential exclusive postfix reverse scan over the statically-sized \p input array, seeded with the specified \p postfix.  The aggregate is returned.
 */
template <
    int         LENGTH,
    typename    T,
    typename    ScanOp>
__device__ __forceinline__ T ThreadReverseScanExclusive(
    const T (&input)[LENGTH],
    T (&output)[LENGTH],
    ScanOp scan_op,
    const T postfix)
{
    // Careful, output maybe be aliased to input
    T exclusive = postfix;
    T inclusive;
    #pragma unroll
    for (int i = LENGTH - 1; i >= 0; --i) {
        inclusive = scan_op(exclusive, input[i]);
        output[i] = exclusive;
        exclusive = inclusive;
    }
    return inclusive;
}


/**
 * \brief WarpReverseScan provides SHFL-based variants of parallel postfix scan of items partitioned across a CUDA thread warp.
 *
 * LOGICAL_WARP_THREADS must be a power-of-two
 */
template <
    typename    T,                      ///< Data type being scanned
    int         LOGICAL_WARP_THREADS    ///< Number of threads per logical warp
    >
struct WarpReverseScan {
    //---------------------------------------------------------------------
    // Constants and type definitions
    //---------------------------------------------------------------------

    /// Whether the logical warp size and the PTX warp size coincide
    static constexpr bool IS_ARCH_WARP = (LOGICAL_WARP_THREADS == CUB_WARP_THREADS(0));
    /// The number of warp scan steps
    static constexpr int STEPS = cub::Log2<LOGICAL_WARP_THREADS>::VALUE;
    static_assert(LOGICAL_WARP_THREADS == 1 << STEPS);


    //---------------------------------------------------------------------
    // Thread fields
    //---------------------------------------------------------------------

    /// Lane index in logical warp
    unsigned int lane_id;

    /// Logical warp index in 32-thread physical warp
    unsigned int warp_id;

    /// 32-thread physical warp member mask of logical warp
    unsigned int member_mask;

    //---------------------------------------------------------------------
    // Construction
    //---------------------------------------------------------------------

    /// Constructor
    explicit __device__ __forceinline__
    WarpReverseScan()
        : lane_id(cub::LaneId())
        , warp_id(IS_ARCH_WARP ? 0 : (lane_id / LOGICAL_WARP_THREADS))
        , member_mask(cub::WarpMask<LOGICAL_WARP_THREADS>(warp_id))
    {
        if (!IS_ARCH_WARP) {
            lane_id = lane_id % LOGICAL_WARP_THREADS;
        }
    }


    /// Broadcast
    __device__ __forceinline__ T Broadcast(
        T               input,              ///< [in] The value to broadcast
        int             src_lane)           ///< [in] Which warp lane is to do the broadcasting
    {
        return cub::ShuffleIndex<LOGICAL_WARP_THREADS>(input, src_lane, member_mask);
    }


    /// Inclusive scan
    template <typename ScanOpT>
    __device__ __forceinline__ void InclusiveReverseScan(
        T               input,              ///< [in] Calling thread's input item.
        T               &inclusive_output,  ///< [out] Calling thread's output item.  May be aliased with \p input.
        ScanOpT         scan_op)            ///< [in] Binary scan operator
    {
        inclusive_output = input;
        #pragma unroll
        for (int STEP = 0; STEP < STEPS; STEP++) {
            int offset = 1 << STEP;
            T temp = cub::ShuffleDown<LOGICAL_WARP_THREADS>(
                inclusive_output, offset, LOGICAL_WARP_THREADS - 1, member_mask
            );
            // Perform scan op if from a valid peer
            inclusive_output = static_cast<int>(lane_id) >= LOGICAL_WARP_THREADS - offset
                ? inclusive_output : scan_op(temp, inclusive_output);
        }
    }

    /// Exclusive scan
    // Get exclusive from inclusive
    template <typename ScanOpT>
    __device__ __forceinline__ void ExclusiveReverseScan(
        T              input,              ///< [in] Calling thread's input item.
        T              &exclusive_output,  ///< [out] Calling thread's output item.  May be aliased with \p input.
        ScanOpT        scan_op,            ///< [in] Binary scan operator
        T              &warp_aggregate)    ///< [out] Warp-wide aggregate reduction of input items.
    {
        T inclusive_output;
        InclusiveReverseScan(input, inclusive_output, scan_op);
        warp_aggregate = cub::ShuffleIndex<LOGICAL_WARP_THREADS>(inclusive_output, 0, member_mask);
        // initial value unknown
        exclusive_output = cub::ShuffleDown<LOGICAL_WARP_THREADS>(
            inclusive_output, 1, LOGICAL_WARP_THREADS - 1, member_mask
        );
    }

    /**
     * \brief Computes both inclusive and exclusive reverse scans using the specified binary scan functor across the calling warp.  Because no initial value is supplied, the \p exclusive_output computed for the last <em>warp-lane</em> is undefined.
     */
    template <typename ScanOpT>
    __device__ __forceinline__ void ReverseScan(
        T               input,              ///< [in] Calling thread's input item.
        T               &inclusive_output,  ///< [out] Calling thread's inclusive-scan output item.
        T               &exclusive_output,  ///< [out] Calling thread's exclusive-scan output item.
        ScanOpT         scan_op)            ///< [in] Binary scan operator
    {
        InclusiveReverseScan(input, inclusive_output, scan_op);
        // initial value unknown
        exclusive_output = cub::ShuffleDown<LOGICAL_WARP_THREADS>(
            inclusive_output, 1, LOGICAL_WARP_THREADS - 1, member_mask
        );
    }

};

/**
 * \brief BlockReverseScan provides variants of raking-based parallel postfix scan across a CUDA thread block.
 */
template <
    typename    T,              ///< Data type being scanned
    int         BLOCK_DIM_X,    ///< The thread block length in threads along the X dimension
    bool        MEMOIZE=false   ///< Whether or not to buffer outer raking scan partials to incur fewer shared memory reads at the expense of higher register pressure
    >
struct BlockReverseScan {
    //---------------------------------------------------------------------
    // Types and constants
    //---------------------------------------------------------------------

    /// Constants
    /// The thread block size in threads
    static constexpr int BLOCK_THREADS = BLOCK_DIM_X;

    /// Layout type for padded thread block raking grid
    using BlockRakingLayout = cub::BlockRakingLayout<T, BLOCK_THREADS>;
    // The number of reduction elements is not a multiple of the number of raking threads for now
    static_assert(BlockRakingLayout::UNGUARDED);

    /// Number of raking threads
    static constexpr int RAKING_THREADS = BlockRakingLayout::RAKING_THREADS;
    /// Number of raking elements per warp synchronous raking thread
    static constexpr int SEGMENT_LENGTH = BlockRakingLayout::SEGMENT_LENGTH;
    /// Cooperative work can be entirely warp synchronous
    static constexpr bool WARP_SYNCHRONOUS = (int(BLOCK_THREADS) == int(RAKING_THREADS));

    ///  WarpReverseScan utility type
    using WarpReverseScan = WarpReverseScan<T, RAKING_THREADS>;

    /// Shared memory storage layout type
    struct _TempStorage {
        typename BlockRakingLayout::TempStorage raking_grid;     ///< Padded thread block raking grid
    };


    /// Alias wrapper allowing storage to be unioned
    struct TempStorage : cub::Uninitialized<_TempStorage> {};


    //---------------------------------------------------------------------
    // Per-thread fields
    //---------------------------------------------------------------------

    // Thread fields
    _TempStorage    &temp_storage;
    unsigned int    linear_tid;
    T               cached_segment[SEGMENT_LENGTH];


    //---------------------------------------------------------------------
    // Utility methods
    //---------------------------------------------------------------------

    /// Performs upsweep raking reduction, returning the aggregate
    template <typename ScanOp>
    __device__ __forceinline__ T Upsweep(ScanOp scan_op) {
        T *smem_raking_ptr = BlockRakingLayout::RakingPtr(temp_storage.raking_grid, linear_tid);
        // Read data into registers
        #pragma unroll
        for (int i = 0; i < SEGMENT_LENGTH; ++i) { cached_segment[i] = smem_raking_ptr[i]; }
        T raking_partial = cached_segment[SEGMENT_LENGTH - 1];
        #pragma unroll
        for (int i = SEGMENT_LENGTH - 2; i >= 0; --i) {
            raking_partial = scan_op(raking_partial, cached_segment[i]);
        }
        return raking_partial;
    }


    /// Performs exclusive downsweep raking scan
    template <typename ScanOp>
    __device__ __forceinline__ void ExclusiveDownsweep(
        ScanOp          scan_op,
        T               raking_partial)
    {
        T *smem_raking_ptr = BlockRakingLayout::RakingPtr(temp_storage.raking_grid, linear_tid);
        // Read data back into registers
        if (!MEMOIZE) {
            #pragma unroll
            for (int i = 0; i < SEGMENT_LENGTH; ++i) { cached_segment[i] = smem_raking_ptr[i]; }
        }
        ThreadReverseScanExclusive(cached_segment, cached_segment, scan_op, raking_partial);
        // Write data back to smem
        #pragma unroll
        for (int i = 0; i < SEGMENT_LENGTH; ++i) { smem_raking_ptr[i] = cached_segment[i]; }
    }


    //---------------------------------------------------------------------
    // Constructors
    //---------------------------------------------------------------------

    /// Constructor
    __device__ __forceinline__ BlockReverseScan(
        TempStorage &temp_storage)
    :
        temp_storage(temp_storage.Alias()),
        linear_tid(cub::RowMajorTid(BLOCK_DIM_X, 1, 1))
    {}


    /// Computes an exclusive thread block-wide postfix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  the call-back functor \p block_postfix_callback_op is invoked by the first warp in the block, and the value returned by <em>lane</em><sub>0</sub> in that warp is used as the "seed" value that logically postfixes the thread block's scan inputs.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
    template <
        typename ScanOp,
        typename BlockPostfixCallbackOp>
    __device__ __forceinline__ void ExclusiveReverseScan(
        T                       input,                          ///< [in] Calling thread's input item
        T                       &exclusive_output,              ///< [out] Calling thread's output item (may be aliased to \p input)
        ScanOp                  scan_op,                        ///< [in] Binary scan operator
        BlockPostfixCallbackOp  &block_postfix_callback_op)     ///< [in-out] <b>[<em>warp</em><sub>0</sub> only]</b> Call-back functor for specifying a thread block-wide postfix to be applied to all inputs.
    {
        if (WARP_SYNCHRONOUS) {
            // Short-circuit directly to warp-synchronous scan
            T block_aggregate;
            WarpReverseScan warp_scan;
            warp_scan.ExclusiveReverseScan(input, exclusive_output, scan_op, block_aggregate);
            // Obtain warp-wide postfix in lane0, then broadcast to other lanes
            T block_postfix = block_postfix_callback_op(block_aggregate);
            block_postfix = warp_scan.Broadcast(block_postfix, 0);
            exclusive_output = linear_tid == BLOCK_THREADS - 1 ? block_postfix : scan_op(block_postfix, exclusive_output);
        } else {
            // Place thread partial into shared memory raking grid
            T *placement_ptr = BlockRakingLayout::PlacementPtr(temp_storage.raking_grid, linear_tid);
            detail::uninitialized_copy(placement_ptr, input);
            cub::CTA_SYNC();
            // Reduce parallelism down to just raking threads
            if (linear_tid < RAKING_THREADS) {
                WarpReverseScan warp_scan;
                // Raking upsweep reduction across shared partials
                T upsweep_partial = Upsweep(scan_op);
                // Warp-synchronous scan
                T exclusive_partial, block_aggregate;
                warp_scan.ExclusiveReverseScan(upsweep_partial, exclusive_partial, scan_op, block_aggregate);
                // Obtain block-wide postfix in lane0, then broadcast to other lanes
                T block_postfix = block_postfix_callback_op(block_aggregate);
                block_postfix = warp_scan.Broadcast(block_postfix, 0);
                // Update postfix with warpscan exclusive partial
                T downsweep_postfix = linear_tid == RAKING_THREADS - 1
                    ? block_postfix : scan_op(block_postfix, exclusive_partial);
                // Exclusive raking downsweep scan
                ExclusiveDownsweep(scan_op, downsweep_postfix);
            }
            cub::CTA_SYNC();
            // Grab thread postfix from shared memory
            exclusive_output = *placement_ptr;

            // // Compute warp scan in each warp.
            // // The exclusive output from the last lane in each warp is invalid.
            // T inclusive_output;
            // WarpReverseScan warp_scan;
            // warp_scan.ReverseScan(input, inclusive_output, exclusive_output, scan_op);

            // // Compute the warp-wide postfix and block-wide aggregate for each warp.  Warp postfix for the last warp is invalid.
            // T block_aggregate;
            // T warp_postfix = ComputeWarpPostfix(scan_op, inclusive_output, block_aggregate);

            // // Apply warp postfix to our lane's partial
            // if (warp_id != 0) {
            //     exclusive_output = scan_op(warp_postfix, exclusive_output);
            //     if (lane_id == 0) { exclusive_output = warp_postfix; }
            // }

            // // Use the first warp to determine the thread block postfix, returning the result in lane0
            // if (warp_id == 0) {
            //     T block_postfix = block_postfix_callback_op(block_aggregate);
            //     if (lane_id == 0) {
            //         // Share the postfix with all threads
            //         detail::uninitialized_copy(&temp_storage.block_postfix,
            //                                   block_postfix);

            //         exclusive_output = block_postfix; // The block postfix is the exclusive output for tid0
            //     }
            // }

            // cub::CTA_SYNC();

            // // Incorporate thread block postfix into outputs
            // T block_postfix = temp_storage.block_postfix;
            // if (linear_tid > 0) { exclusive_output = scan_op(block_postfix, exclusive_output); }
        }
    }


    /**
     * \brief Computes an inclusive block-wide postfix scan using the specified binary \p scan_op functor.  Each thread contributes an array of consecutive input elements.  the call-back functor \p block_postfix_callback_op is invoked by the first warp in the block, and the value returned by <em>lane</em><sub>0</sub> in that warp is used as the "seed" value that logically postfixes the thread block's scan inputs.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
     */
    template <
        int             ITEMS_PER_THREAD,
        typename        ScanOp,
        typename        BlockPostfixCallbackOp>
    __device__ __forceinline__ void InclusiveReverseScan(
        T                       (&input)[ITEMS_PER_THREAD],     ///< [in] Calling thread's input items
        T                       (&output)[ITEMS_PER_THREAD],    ///< [out] Calling thread's output items (may be aliased to \p input)
        ScanOp                  scan_op,                        ///< [in] Binary scan functor
        BlockPostfixCallbackOp   &block_postfix_callback_op)    ///< [in-out] <b>[<em>warp</em><sub>0</sub> only]</b> Call-back functor for specifying a block-wide postfix to be applied to the logical input sequence.
    {
        // Reduce consecutive thread items in registers
        T thread_postfix = ThreadReverseReduce(input, scan_op);
        // Exclusive thread block-scan
        ExclusiveReverseScan(thread_postfix, thread_postfix, scan_op, block_postfix_callback_op);
        // Inclusive scan in registers with postfix as seed
        ThreadReverseScanInclusive(input, output, scan_op, thread_postfix);
    }

};