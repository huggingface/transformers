#include <torch/extension.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/Sort.h>
#include <ATen/native/Sorting.h>
#include <ATen/cuda/cub.h>
#include <ATen/ceil_div.h>

// #define ENABLE_HALF

template <typename T, const int N> using Accessor = torch::PackedTensorAccessor32<T, N, torch::RestrictPtrTraits>;


struct Dims {
    int N;
    int M;
    int L;
    int K;
};



template<const int BN, const int BM, const int BL, const int TN, const int TL, bool transpose, typename T, typename iT> __global__ void cvmm_sorted_blocktile_co_raw_kernel(Dims dims, T* __restrict__ pV, iT* __restrict__ pS, T* __restrict__ pM, T* __restrict__ pY, iT* __restrict__ order, iT* __restrict__ counts, iT* __restrict__ offsets)
{
    const int n_vects = counts[blockIdx.z];
    for (int nchunk = 0;; nchunk += gridDim.y * BN){
        const int n_within_expert = blockIdx.y * blockDim.y * TN + nchunk;
        if (n_within_expert >= n_vects)
            return;

        const int offset = blockIdx.z > 0 ? offsets[blockIdx.z - 1] : 0;
        const int block_start_l = blockIdx.x * blockDim.x * TL;
        const int block_start_fake_n = n_within_expert + offset;
        const int l = block_start_l + threadIdx.x * TL;
        const int ind = blockIdx.z * blockDim.z + threadIdx.z;
        const int thread_index = threadIdx.x + threadIdx.y * blockDim.x;
        const int N_THREADS = blockDim.x * blockDim.y * blockDim.z;
        const int MAT_CACHE_SIZE = BM * BL;
        const int INPUT_CACHE_SIZE = BN * BM;

        const int LOAD_SIZE_FACTOR = sizeof(float) * 4 / sizeof(T);
        const int N_LOAD_AT_ONCE = N_THREADS * LOAD_SIZE_FACTOR;

        const int thread_index_load = thread_index * LOAD_SIZE_FACTOR;

        __shared__ T mat_slice[MAT_CACHE_SIZE];
        __shared__ T input_slice[INPUT_CACHE_SIZE];


        #define ORDER(i) (order[block_start_fake_n + (i)])
        #define NEED_TO_WORK(i) ((n_within_expert + (i)) < n_vects)

        T subres[TN * TL] = {0.0};
        T reg_input[TN];
        T reg_mat[TL];

        const T *pmCurrent = pM + (ind * dims.M * dims.L);

        const float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        const bool thread_has_work = (threadIdx.y * TN) < n_vects && l < dims.L;

        for(int m = 0; m < dims.M; m += BM){
            for (int linaddr = 0; linaddr < MAT_CACHE_SIZE; linaddr += N_LOAD_AT_ONCE){
                const int ii = linaddr + thread_index_load;
                if (ii < MAT_CACHE_SIZE){

                    if constexpr(transpose){
                        const int y = ii % BM;
                        const int x = ii / BM;

                        const int lm = m + y;
                        const int ll = block_start_l + x;

                        T load_buffer[LOAD_SIZE_FACTOR];
                        if ((lm < dims.M) & (ll < dims.L)){
                            reinterpret_cast<float4 *>(&load_buffer)[0] = reinterpret_cast<const float4 *>(&pmCurrent[ll * dims.M + lm])[0];

                            #pragma unroll
                            for (int i=0; i<LOAD_SIZE_FACTOR; ++i){
                                mat_slice[(y + i) * BL + x] = load_buffer[i];
                            }
                        } else {
                            #pragma unroll
                            for (int i=0; i<LOAD_SIZE_FACTOR; ++i){
                                mat_slice[(y + i) * BL + x] = 0;
                            }
                        }
                    } else {
                        const int x = ii % BL;
                        const int y = ii / BL;

                        const int lm = m + y;
                        const int ll = block_start_l + x;

                        if ((lm < dims.M) & (ll < dims.L)){
                            reinterpret_cast<float4 *>(&mat_slice[ii])[0] = reinterpret_cast<const float4 *>(&pmCurrent[lm * dims.L + ll])[0];
                        } else {
                            reinterpret_cast<float4 *>(&mat_slice[ii])[0] = zero;
                        }
                    }
                }
            }

            for (int linaddr = 0; linaddr < INPUT_CACHE_SIZE; linaddr += N_LOAD_AT_ONCE){
                const int ii = linaddr + thread_index_load;
                if (ii < INPUT_CACHE_SIZE){
                    const int x = ii % BM;
                    const int y = ii / BM;

                    const int lm = m + x;
                    const int ln = ORDER(y);

                    if (NEED_TO_WORK(y) & (lm < dims.M)){
                        reinterpret_cast<float4 *>(&input_slice[ii])[0] = reinterpret_cast<const float4 *>(&pV[ln * dims.M + lm])[0];
                    } else {
                        reinterpret_cast<float4 *>(&input_slice[ii])[0] = zero;
                    }
                }
            }

            __syncthreads();

            if (thread_has_work){
                for (int di = 0; di < BM; ++di) {
                    for (int n=0; n<TN; ++n){
                        reg_input[n] = input_slice[(threadIdx.y * TN + n) * BM + di];
                    }

                    for (int n=0; n<TL; ++n){
                        reg_mat[n] = mat_slice[di * BL + threadIdx.x * TL + n];
                    }

                    #ifdef ENABLE_HALF
                    if constexpr(std::is_same<T, at::Half>::value){
                        for (int tn=0; tn<TN; tn++){
                            const half2 other = __halves2half2(reg_input[tn], reg_input[tn]);

                            for (int tl=0; tl<TL; tl+=2){
                                *reinterpret_cast<half2 *>(subres + tn * TL + tl) = __hfma2(other, *reinterpret_cast<half2 *>(reg_mat + tl), *reinterpret_cast<half2 *>(subres + tn * TL + tl));
                            }
                        }
                    } else {
                    #endif
                        for (int tn=0; tn<TN; ++tn){
                            for (int tl=0; tl<TL; ++tl){
                                subres[tn * TL + tl] += reg_input[tn] * reg_mat[tl];
                            }
                        }
                    #ifdef ENABLE_HALF
                    }
                    #endif

                }
            }

            __syncthreads();
        }


        if (thread_has_work){
            for (int tn=0; tn<TN; ++tn){
                const int ln = threadIdx.y * TN + tn;
                if (NEED_TO_WORK(ln)){
                    T* const out_row_start = pY + ORDER(ln) * dims.L;
                    if constexpr(TL*sizeof(T) % 16 == 0){
                        for (int tl=0; tl<TL; tl += LOAD_SIZE_FACTOR){
                            const int ll = l + tl;
                            reinterpret_cast<float4 *>(&out_row_start[ll])[0] = reinterpret_cast<float4 *>(&subres[tn * TL + tl])[0];
                        }
                    } else {
                        for (int tl=0; tl<TL; ++tl){
                            const int ll = l + tl;
                            out_row_start[ll] = subres[tn * TL + tl];
                        }
                    }
                }
            }
        }
    }
}


template<const int BN, const int BM, const int BL, const int TM, const int TL, const int MAX_N, typename T, typename iT, typename gT> __global__ void cvmm_sorted_blocktile_co_raw_project_grads_kernel(Dims dims, T* __restrict__ pV, iT* __restrict__ pS, T* __restrict__ pG, gT* __restrict__ pM, iT* __restrict__ order, iT* __restrict__ counts, iT* __restrict__ offsets)
{
    const int ind = blockIdx.z % dims.K;
    const int n_vects = counts[ind];
    const int n_blocks = gridDim.z*blockDim.z / dims.K;

    for (int n_offset = (blockIdx.z / dims.K) * MAX_N;; n_offset += n_blocks * MAX_N){
        if (n_vects < n_offset)
            return;

        const int offset = ind > 0 ? offsets[ind - 1] : 0;
        const int block_start_l = blockIdx.x * blockDim.x * TL;
        const int l = block_start_l + threadIdx.x * TL;
        const int block_start_m = blockIdx.y * blockDim.y * TM;
        const int m = block_start_m + threadIdx.y * TM;

        const int thread_index = threadIdx.x + threadIdx.y * blockDim.x;


        const int N_THREADS = blockDim.x * blockDim.y * blockDim.z;
        const int INPUT_CACHE_SIZE = BN * BM;
        const int GRAD_CACHE_SIZE = BN * BL;

        const int LOAD_SIZE_FACTOR = sizeof(float) * 4 / sizeof(T);
        const int N_LOAD_AT_ONCE = N_THREADS * LOAD_SIZE_FACTOR;

        const int thread_index_load = thread_index * LOAD_SIZE_FACTOR;

        __shared__ T grad_slice[GRAD_CACHE_SIZE];
        __shared__ T input_slice[INPUT_CACHE_SIZE];


        gT subres[TM * TL] = {0.0};
        T reg_input[TM];
        T reg_grad[TL];

        gT *pmCurrent = pM + (ind * dims.M * dims.L);

        const float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        const bool thread_has_work = l < dims.L && m < dims.M;

        const int end_of_chunk = min(n_vects, n_offset + MAX_N);
        for(int n = n_offset; n < end_of_chunk; n += BN){
            for (int linaddr = 0; linaddr < GRAD_CACHE_SIZE; linaddr += N_LOAD_AT_ONCE){
                const int ii = linaddr + thread_index_load;
                if (ii < GRAD_CACHE_SIZE){
                    const int x = ii % BL;
                    const int y = ii / BL;

                    const int ln = n + y;
                    const int ll = block_start_l + x;

                    const bool need_to_work = (ln < n_vects) & (ll < dims.L);

                    reinterpret_cast<float4 *>(&grad_slice[ii])[0] = need_to_work ? reinterpret_cast<const float4 *>(&pG[order[offset+ln] * dims.L + ll])[0] : zero;
                }
            }

            for (int linaddr = 0; linaddr < INPUT_CACHE_SIZE; linaddr += N_LOAD_AT_ONCE){
                const int ii = linaddr + thread_index_load;
                if (ii < INPUT_CACHE_SIZE){
                    const int x = ii % BM;
                    const int y = ii / BM;

                    const int lm = block_start_m + x;
                    const int ln = n + y;

                    const bool need_to_work = (ln < n_vects) & (lm < dims.M);

                    reinterpret_cast<float4 *>(&input_slice[ii])[0] = need_to_work ? reinterpret_cast<const float4 *>(&pV[order[offset+ln] * dims.M + lm])[0] : zero;
                }
            }

            __syncthreads();

            if (thread_has_work){
                for (int di = 0; di < BN; ++di) {
                    for (int n=0; n<TM; ++n){
                        reg_input[n] = input_slice[threadIdx.y * TM + n + di * BM];
                    }

                    for (int n=0; n<TL; ++n){
                        reg_grad[n] = grad_slice[threadIdx.x * TL + n + di * BL];
                    }

                    for (int tm=0; tm<TM; ++tm){
                        for (int tl=0; tl<TL; ++tl){
                            subres[tm * TL + tl] += reg_input[tm] * reg_grad[tl];
                        }
                    }
                }
            }

            __syncthreads();
        }

        if (thread_has_work){
            for (int tm=0; tm<TM; ++tm){
                const int lm = m + tm;
                gT* const out_row_start = pmCurrent + lm * dims.L;

                if (n_vects < MAX_N){
                    if constexpr(TL*sizeof(T) % 16 == 0){
                        for (int tl=0; tl<TL; tl += LOAD_SIZE_FACTOR){
                            const int ll = l + tl;
                            reinterpret_cast<float4 *>(&out_row_start[ll])[0] = reinterpret_cast<float4 *>(&subres[tm * TL + tl])[0];
                        }
                    } else {
                        for (int tl=0; tl<TL; ++tl){
                            const int ll = l + tl;
                            out_row_start[ll] = subres[tm * TL + tl];
                        }
                    }
                } else {
                    for (int tl=0; tl<TL; ++tl){
                        const int ll = l + tl;
                        atomicAdd(&out_row_start[ll], subres[tm * TL + tl]);
                    }
                }
            }
        }
    }
}




std::vector<torch::Tensor> sort_indices(torch::Tensor &indices, const int n_indices){
    auto sorted_indices = torch::empty_like(indices, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    auto orig_indices = torch::empty_like(indices, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    const int num_indices = indices.numel();

    std::vector<torch::Tensor> result;

    AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "sort_indices", [&] () {
        auto range = torch::arange(num_indices, indices.options());
        int64_t nbits = at::cuda::cub::get_num_bits(n_indices);
        at::cuda::cub::radix_sort_pairs(
            indices.data_ptr<index_t>(), sorted_indices.data_ptr<index_t>(),
            range.data_ptr<index_t>(), orig_indices.data_ptr<index_t>(),
            num_indices, false, 0, nbits);
    });

    result.push_back(sorted_indices);
    result.push_back(orig_indices);

    return result;
}



torch::Tensor cvmm_sorted_blocktile_co_raw(torch::Tensor &bvec, torch::Tensor &indices, torch::Tensor &mats, torch::Tensor &order, torch::Tensor &counts, torch::Tensor &offsets){
    auto out = torch::zeros({bvec.size(0), mats.size(2)}, bvec.options());

    // Dims:
    //    bvec: [N, M]
    //    indices: [N]
    //    mats: [K, M, L]
    //    order: [N]

    const int N = bvec.size(0);
    const int L = mats.size(2);

    const Dims d{N, (int)bvec.size(1), L, (int)mats.size(0)};

    const int BN = 32;
    const int BM = 16;
    const int BL = 64;
    const int TL = 4;
    const int TN = 4;

    assert(d.M % 4 == 0);
    assert(d.L % TL == 0 && d.L % 4 == 0);
    static_assert(BN % TN == 0);
    static_assert(BL % TL == 0);

    // const int max_n = counts.max().item<int>();
    // const int max_n = N;

    const int gsX = (L + BL - 1) / BL;
    const int targetGsN = (N + BN - 1) / BN;

    const int maxGridSize = 16384;
    const int maxGsN = max(int(maxGridSize / (gsX*mats.size(0))), 1);

    dim3 dimGrid(gsX, min(targetGsN, maxGsN), mats.size(0));
    dim3 dimBlock( BL / TL, BN / TN, 1);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const bool transpose = mats.stride(1) == 1;

    // AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        // bvec.scalar_type(), "cvmm_sorted_blocktile_co_raw", [&] {
         using scalar_t = float;
            AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "cvmm_sorted_blocktile_co_raw", [&]() {
                if (transpose){
                    cvmm_sorted_blocktile_co_raw_kernel<BN, BM, BL, TN, TL, true, scalar_t, index_t><<<dimGrid, dimBlock, 0, stream>>>(
                        d,
                        bvec.data_ptr<scalar_t>(),
                        indices.data_ptr<index_t>(),
                        mats.data_ptr<scalar_t>(),
                        out.data_ptr<scalar_t>(),
                        order.data_ptr<index_t>(),
                        counts.data_ptr<index_t>(),
                        offsets.data_ptr<index_t>()
                    );
                } else {
                    cvmm_sorted_blocktile_co_raw_kernel<BN, BM, BL, TN, TL, false, scalar_t, index_t><<<dimGrid, dimBlock, 0, stream>>>(
                        d,
                        bvec.data_ptr<scalar_t>(),
                        indices.data_ptr<index_t>(),
                        mats.data_ptr<scalar_t>(),
                        out.data_ptr<scalar_t>(),
                        order.data_ptr<index_t>(),
                        counts.data_ptr<index_t>(),
                        offsets.data_ptr<index_t>()
                    );
                }

                C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
    // });

    return out;
}





torch::Tensor cvmm_sorted_blocktile_co_raw_project_grads(const int n_experts, torch::Tensor &bvec, torch::Tensor &indices, torch::Tensor &grads, torch::Tensor &order, torch::Tensor &counts, torch::Tensor &offsets){
    // Dims:
    //    bvec: [N, M]
    //    indices: [N]
    //    grads: [N, L]
    //    order: [N]

    const Dims d{(int)bvec.size(0), (int)bvec.size(1), (int)grads.size(1), n_experts};

    const int BN = 8;
    const int BM = 32;
    const int BL = 64;
    const int TL = 4;
    const int TM = 4;
    const int MAX_N = 512;
    const int MAX_N_CHUNKS = 16;

    assert(d.M % 4 == 0);
    assert(d.L % TL == 0 && d.L % 4 == 0);

    int n_chunks;
    if (n_experts * MAX_N_CHUNKS > 32768){
        n_chunks = max(32768 / n_experts, 1);
    } else {
        n_chunks = MAX_N_CHUNKS;
    }

    dim3 dimGrid((d.L + BL - 1) / BL, (d.M + BM - 1) / BM, n_experts * n_chunks);
    dim3 dimBlock( BL / TL, BM / TM, 1);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto out = torch::zeros({d.K, d.M, d.L},  torch::TensorOptions().dtype(torch::kFloat32).device(bvec.device()).layout(bvec.layout()));


    // template<const int BN, const int BM, const int BL, const int TM, const int TL, typename T, typename iT> __global__ void cvmm_sorted_blocktile_co_raw_project_grads_kernel(Dims dims, T* __restrict__ pV, iT* __restrict__ pS, T* __restrict__ pG, T* __restrict__ pM, iT* __restrict__ order, iT* __restrict__ counts, iT* __restrict__ offsets)

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        bvec.scalar_type(), "cvmm_sorted_blocktile_co_raw", [&] {
        AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "cvmm_sorted_blocktile_co_raw_project_grads", [&]() {
                cvmm_sorted_blocktile_co_raw_project_grads_kernel<BN, BM, BL, TM, TL, MAX_N, scalar_t, index_t, float><<<dimGrid, dimBlock, 0, stream>>>(
                    d,
                    bvec.data_ptr<scalar_t>(),
                    indices.data_ptr<index_t>(),
                    grads.data_ptr<scalar_t>(),
                    out.data_ptr<float>(),
                    order.data_ptr<index_t>(),
                    counts.data_ptr<index_t>(),
                    offsets.data_ptr<index_t>()
                );

                C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
    });

    return out;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "cvmm_sorted_blocktile_co_raw",
        &cvmm_sorted_blocktile_co_raw,
        "Conditional vector-matrix multiplication"
    );

    m.def(
        "cvmm_sorted_blocktile_co_raw_project_grads",
        &cvmm_sorted_blocktile_co_raw_project_grads,
        "Conditional vector-matrix multiplication gradient projection"
    );

    m.def(
        "sort_indices",
        &sort_indices,
        "Sort indices"
    );
}
