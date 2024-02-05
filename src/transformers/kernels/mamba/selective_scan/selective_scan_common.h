/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <c10/util/complex.h>  // For scalar_value_type

#define MAX_DSTATE 256

using complex_t = c10::complex<float>;

inline __device__ float2 operator+(const float2 & a, const float2 & b){
    return {a.x + b.x, a.y + b.y};
}

inline __device__ float3 operator+(const float3 &a, const float3 &b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}

inline __device__ float4 operator+(const float4 & a, const float4 & b){
    return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int BYTES> struct BytesToType {};

template<> struct BytesToType<16> {
    using Type = uint4;
    static_assert(sizeof(Type) == 16);
};

template<> struct BytesToType<8> {
    using Type = uint64_t;
    static_assert(sizeof(Type) == 8);
};

template<> struct BytesToType<4> {
    using Type = uint32_t;
    static_assert(sizeof(Type) == 4);
};

template<> struct BytesToType<2> {
    using Type = uint16_t;
    static_assert(sizeof(Type) == 2);
};

template<> struct BytesToType<1> {
    using Type = uint8_t;
    static_assert(sizeof(Type) == 1);
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename scalar_t, int N>
struct Converter{
    static inline __device__ void to_float(const scalar_t (&src)[N], float (&dst)[N]) {
        #pragma unroll
        for (int i = 0; i < N; ++i) { dst[i] = src[i]; }
    }
};

template<int N>
struct Converter<at::Half, N>{
    static inline __device__ void to_float(const at::Half (&src)[N], float (&dst)[N]) {
        static_assert(N % 2 == 0);
        auto &src2 = reinterpret_cast<const half2 (&)[N / 2]>(src);
        auto &dst2 = reinterpret_cast<float2 (&)[N / 2]>(dst);
        #pragma unroll
        for (int i = 0; i < N / 2; ++i) { dst2[i] = __half22float2(src2[i]); }
    }
};

#if __CUDA_ARCH__ >= 800
template<int N>
struct Converter<at::BFloat16, N>{
    static inline __device__ void to_float(const at::BFloat16 (&src)[N], float (&dst)[N]) {
        static_assert(N % 2 == 0);
        auto &src2 = reinterpret_cast<const nv_bfloat162 (&)[N / 2]>(src);
        auto &dst2 = reinterpret_cast<float2 (&)[N / 2]>(dst);
        #pragma unroll
        for (int i = 0; i < N / 2; ++i) { dst2[i] = __bfloat1622float2(src2[i]); }
    }
};
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

// From https://stackoverflow.com/questions/9860711/cucomplex-h-and-exp
// and https://forums.developer.nvidia.com/t/complex-number-exponential-function/24696
__device__ __forceinline__ complex_t cexp2f(complex_t z) {
    float t = exp2f(z.real_);
    float c, s;
    sincosf(z.imag_, &s, &c);
    return complex_t(c * t, s * t);
}

__device__ __forceinline__ complex_t cexpf(complex_t z) {
    float t = expf(z.real_);
    float c, s;
    sincosf(z.imag_, &s, &c);
    return complex_t(c * t, s * t);
}

template<typename scalar_t> struct SSMScanOp;

template<>
struct SSMScanOp<float> {
    __device__ __forceinline__ float2 operator()(const float2 &ab0, const float2 &ab1) const {
        return make_float2(ab1.x * ab0.x, ab1.x * ab0.y + ab1.y);
    }
};

template<>
struct SSMScanOp<complex_t> {
    __device__ __forceinline__ float4 operator()(const float4 &ab0, const float4 &ab1) const {
        complex_t a0 = complex_t(ab0.x, ab0.y);
        complex_t b0 = complex_t(ab0.z, ab0.w);
        complex_t a1 = complex_t(ab1.x, ab1.y);
        complex_t b1 = complex_t(ab1.z, ab1.w);
        complex_t out_a = a1 * a0;
        complex_t out_b = a1 * b0 + b1;
        return make_float4(out_a.real_, out_a.imag_, out_b.real_, out_b.imag_);
    }
};

// A stateful callback functor that maintains a running prefix to be applied
// during consecutive scan operations.
template <typename scalar_t> struct SSMScanPrefixCallbackOp {
    using scan_t = std::conditional_t<std::is_same_v<scalar_t, float>, float2, float4>;
    scan_t running_prefix;
    // Constructor
    __device__ SSMScanPrefixCallbackOp(scan_t running_prefix_) : running_prefix(running_prefix_) {}
    // Callback operator to be entered by the first warp of threads in the block.
    // Thread-0 is responsible for returning a value for seeding the block-wide scan.
    __device__ scan_t operator()(scan_t block_aggregate) {
        scan_t old_prefix = running_prefix;
        running_prefix = SSMScanOp<scalar_t>()(running_prefix, block_aggregate);
        return old_prefix;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Ktraits>
inline __device__ void load_input(typename Ktraits::input_t *u,
                                  typename Ktraits::input_t (&u_vals)[Ktraits::kNItems],
                                  typename Ktraits::BlockLoadT::TempStorage &smem_load,
                                  int seqlen) {
    if constexpr (Ktraits::kIsEvenLen) {
        auto& smem_load_vec = reinterpret_cast<typename Ktraits::BlockLoadVecT::TempStorage&>(smem_load);
        using vec_t = typename Ktraits::vec_t;
        Ktraits::BlockLoadVecT(smem_load_vec).Load(
            reinterpret_cast<vec_t*>(u),
            reinterpret_cast<vec_t(&)[Ktraits::kNLoads]>(u_vals)
       );
    } else {
        Ktraits::BlockLoadT(smem_load).Load(u, u_vals, seqlen, 0.f);
    }
}

template<typename Ktraits>
inline __device__ void load_weight(typename Ktraits::input_t *Bvar,
                                   typename Ktraits::weight_t (&B_vals)[Ktraits::kNItems],
                                   typename Ktraits::BlockLoadWeightT::TempStorage &smem_load_weight,
                                   int seqlen) {
    constexpr int kNItems = Ktraits::kNItems;
    if constexpr (!Ktraits::kIsComplex) {
        typename Ktraits::input_t B_vals_load[kNItems];
        if constexpr (Ktraits::kIsEvenLen) {
            auto& smem_load_weight_vec = reinterpret_cast<typename Ktraits::BlockLoadWeightVecT::TempStorage&>(smem_load_weight);
            using vec_t = typename Ktraits::vec_t;
            Ktraits::BlockLoadWeightVecT(smem_load_weight_vec).Load(
                reinterpret_cast<vec_t*>(Bvar),
                reinterpret_cast<vec_t(&)[Ktraits::kNLoads]>(B_vals_load)
          );
        } else {
            Ktraits::BlockLoadWeightT(smem_load_weight).Load(Bvar, B_vals_load, seqlen, 0.f);
        }
        // #pragma unroll
        // for (int i = 0; i < kNItems; ++i) { B_vals[i] = B_vals_load[i]; }
        Converter<typename Ktraits::input_t, kNItems>::to_float(B_vals_load, B_vals);
    } else {
        typename Ktraits::input_t B_vals_load[kNItems * 2];
        if constexpr (Ktraits::kIsEvenLen) {
            auto& smem_load_weight_vec = reinterpret_cast<typename Ktraits::BlockLoadWeightVecT::TempStorage&>(smem_load_weight);
            using vec_t = typename Ktraits::vec_t;
            Ktraits::BlockLoadWeightVecT(smem_load_weight_vec).Load(
                reinterpret_cast<vec_t*>(Bvar),
                reinterpret_cast<vec_t(&)[Ktraits::kNLoads * 2]>(B_vals_load)
          );
        } else {
            Ktraits::BlockLoadWeightT(smem_load_weight).Load(Bvar, B_vals_load, seqlen, 0.f);
        }
        #pragma unroll
        for (int i = 0; i < kNItems; ++i) { B_vals[i] = complex_t(B_vals_load[i * 2], B_vals_load[i * 2 + 1]); }
    }
}

template<typename Ktraits>
inline __device__ void store_output(typename Ktraits::input_t *out,
                                    const float (&out_vals)[Ktraits::kNItems],
                                    typename Ktraits::BlockStoreT::TempStorage &smem_store,
                                    int seqlen) {
    typename Ktraits::input_t write_vals[Ktraits::kNItems];
    #pragma unroll
    for (int i = 0; i < Ktraits::kNItems; ++i) { write_vals[i] = out_vals[i]; }
    if constexpr (Ktraits::kIsEvenLen) {
        auto& smem_store_vec = reinterpret_cast<typename Ktraits::BlockStoreVecT::TempStorage&>(smem_store);
        using vec_t = typename Ktraits::vec_t;
        Ktraits::BlockStoreVecT(smem_store_vec).Store(
            reinterpret_cast<vec_t*>(out),
            reinterpret_cast<vec_t(&)[Ktraits::kNLoads]>(write_vals)
       );
    } else {
        Ktraits::BlockStoreT(smem_store).Store(out, write_vals, seqlen);
    }
}
