#include <stdio.h>
#include <assert.h>
#include "ATen/ATen.h"
#define MIN_VALUE (-1e38)
typedef at::BFloat16 bf16;

__global__ void kernel_forward_bf16(
    const int B, const int T, const int C, const float *__restrict__ const _w, const bf16 *__restrict__ const _u,
    const bf16 *__restrict__ const _k, const bf16 *__restrict__ const _v, bf16 *__restrict__ const _y
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int _b = idx / C;
    const int _c = idx % C;
    const int _offset = _b * T * C + _c;

    float u = float(_u[_c]);
    float w = _w[_c];
    const bf16 *__restrict__ const k = _k + _offset;
    const bf16 *__restrict__ const v = _v + _offset;
    bf16 *__restrict__ const y = _y + _offset;

    // aa and bb are running sums divided by exp(pp) (to avoid overflow)
    float aa = 0, bb = 0, pp = MIN_VALUE;
    for (int i = 0; i < T; i++) {
        const int ii = i * C;
        const float kk = float(k[ii]);
        const float vv = float(v[ii]);

        float ww = u + kk;
        float p = max(pp, ww);
        float e1 = exp(pp - p);
        float e2 = exp(ww - p);
        y[ii] = bf16((e1 * aa + e2 * vv) / (e1 * bb + e2));
        
        ww = w + pp;
        p = max(ww, kk);
        e1 = exp(ww - p);
        e2 = exp(kk - p);
        aa = e1 * aa + e2 * vv;
        bb = e1 * bb + e2;
        pp = p;
    }
}

__global__ void kernel_forward_with_state_bf16(
    const int B, const int T, const int C, const float *__restrict__ const _w, const bf16 *__restrict__ const _u,
    const bf16 *__restrict__ const _k, const bf16 *__restrict__ const _v, bf16 *__restrict__ const _y,
    float *__restrict__ const _s
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int _b = idx / C;
    const int _c = idx % C;
    const int _offset_s = _b * C * 3 + _c * 3;
    const int _offset = _b * T * C + _c;

    float u = float(_u[_c]);
    float w = _w[_c];
    const bf16 *__restrict__ const k = _k + _offset;
    const bf16 *__restrict__ const v = _v + _offset;
    bf16 *__restrict__ const y = _y + _offset;
    float *__restrict__ const s = _s + _offset_s;

    // aa and bb are running sums divided by exp(pp) (to avoid overflow)
    float aa = s[0], bb = s[1], pp = s[2];
    for (int i = 0; i < T; i++) {
        const int ii = i * C;
        const float kk = float(k[ii]);
        const float vv = float(v[ii]);

        float ww = u + kk;
        float p = max(pp, ww);
        float e1 = exp(pp - p);
        float e2 = exp(ww - p);
        y[ii] = bf16(e1 * aa + e2 * vv) / (e1 * bb + e2);
        
        ww = w + pp;
        p = max(ww, kk);
        e1 = exp(ww - p);
        e2 = exp(kk - p);
        aa = e1 * aa + e2 * vv;
        bb = e1 * bb + e2;
        pp = p;
    }
    s[0] = aa;
    s[1] = bb;
    s[2] = pp;
}

__global__ void kernel_backward_bf16(
    const int B, const int T, const int C, const float *__restrict__ const _w, const bf16 *__restrict__ const _u,
    const bf16 *__restrict__ const _k, const bf16 *__restrict__ const _v, const bf16 *__restrict__ const _y,
    const bf16 *__restrict__ const _gy, bf16 *__restrict__ const _gw, bf16 *__restrict__ const _gu,
    bf16 *__restrict__ const _gk, bf16 *__restrict__ const _gv
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int _b = idx / C;
    const int _c = idx % C;
    const int _offset = _b * T * C + _c;

    float u = float(_u[_c]);
    float w = _w[_c];
    const bf16 *__restrict__ const k = _k + _offset;
    const bf16 *__restrict__ const v = _v + _offset;
    const bf16 *__restrict__ const y = _y + _offset;
    const bf16 *__restrict__ const gy = _gy + _offset;
    bf16 *__restrict__ const gk = _gk + _offset;
    bf16 *__restrict__ const gv = _gv + _offset;

    float q[Tmax], r[Tmax];

    float gw = 0, gu = 0, aa = 0, bb = 0, ga = 0, gb = 0, pp = MIN_VALUE;
    for (int i = 0; i < T; i++) {
        const int ii = i * C;
        const float kk = float(k[ii]);
        const float vv = float(v[ii]);
        const float yy = float(y[ii]);

        float ww = u + kk;
        float p = max(pp, ww);
        float e1 = exp(pp - p);
        float e2 = exp(ww - p);
        const float qq = float(gy[ii]) / (e1 * bb + e2);
        gw += (ga - gb * yy) * e1 * qq;
        gu += (vv - yy) * e2 * qq;
        q[i] = qq;
        r[i] = ww - p;

        ww = w + pp;
        p = max(ww, kk);
        e1 = exp(ww - p);
        e2 = exp(kk - p);
        ga = e1 * (aa + ga);
        gb = e1 * (bb + gb);
        aa = e1 * aa + e2 * vv;
        bb = e1 * bb + e2;
        pp = p;
    }
    const int _offsetBC = _b * C + _c;
    _gw[_offsetBC] = bf16(gw * _w[_c]); // multiply by w because of w -> -exp(w) in python forward()
    _gu[_offsetBC] = bf16(gu);

    aa = 0, bb = 0, pp = MIN_VALUE;
    for (int i = T - 1; i >= 0; i--) {
        const int ii = i * C;
        const float kk = float(k[ii]);
        const float vv = float(v[ii]);
        const float yy = float(y[ii]);
        const float qq = q[i];
        const float rr = r[i];

        float e1 = qq * exp(rr);
        float e2 = exp(kk + pp);
        gk[ii] = bf16(e1 * (vv - yy) + e2 * (aa * vv + bb));
        gv[ii] = bf16(e1 + e2 * aa);

        const float ww = w + pp;
        const float www = rr - u - kk;
        const float p = max(ww, www);
        e1 = exp(ww - p);
        e2 = qq * exp(www - p);
        aa = e1 * aa + e2;
        bb = e1 * bb - e2 * yy;
        pp = p;
    }
}

void cuda_forward_bf16(int B, int T, int C, float *w, bf16 *u, bf16 *k, bf16 *v, bf16 *y) {
    dim3 threadsPerBlock( min(C, 32) ); // requires --maxrregcount 60 for optimal performance
    assert(B * C % threadsPerBlock.x == 0);
    dim3 numBlocks(B * C / threadsPerBlock.x);
    kernel_forward_bf16<<<numBlocks, threadsPerBlock>>>(B, T, C, w, u, k, v, y);
}

void cuda_forward_with_state_bf16(int B, int T, int C, float *w, bf16 *u, bf16 *k, bf16 *v, bf16 *y, float *s) {
    dim3 threadsPerBlock( min(C, 32) ); // requires --maxrregcount 60 for optimal performance
    assert(B * C % threadsPerBlock.x == 0);
    dim3 numBlocks(B * C / threadsPerBlock.x);
    kernel_forward_with_state_bf16<<<numBlocks, threadsPerBlock>>>(B, T, C, w, u, k, v, y, s);
}

void cuda_backward_bf16(int B, int T, int C, float *w, bf16 *u, bf16 *k, bf16 *v, bf16 *y, bf16 *gy, bf16 *gw, bf16 *gu, bf16 *gk, bf16 *gv) {
    dim3 threadsPerBlock( min(C, 32) ); // requires --maxrregcount 60 for optimal performance
    assert(B * C % threadsPerBlock.x == 0);
    dim3 numBlocks(B * C / threadsPerBlock.x);
    kernel_backward_bf16<<<numBlocks, threadsPerBlock>>>(B, T, C, w, u, k, v, y, gy, gw, gu, gk, gv);
}
