#include <torch/extension.h>
#include "ATen/ATen.h"
typedef at::BFloat16 bf16;

void cuda_forward(int B, int T, int C, float *w, float *u, float *k, float *v, float *y);
void cuda_forward_bf16(int B, int T, int C, float *w, bf16 *u, bf16 *k, bf16 *v, bf16 *y);
void cuda_forward_with_state(int B, int T, int C, float *w, float *u, float *k, float *v, float *y, float *s);
void cuda_forward_with_state_bf16(int B, int T, int C, float *w, bf16 *u, bf16 *k, bf16 *v, bf16 *y, float *s);
void cuda_backward(int B, int T, int C, float *w, float *u, float *k, float *v, float *y, float *gy, float *gw, float *gu, float *gk, float *gv);
void cuda_backward_bf16(int B, int T, int C, float *w, bf16 *u, bf16 *k, bf16 *v, bf16 *y, bf16 *gy, bf16 *gw, bf16 *gu, bf16 *gk, bf16 *gv);

void forward(torch::Tensor &w, torch::Tensor &u, torch::Tensor &k, torch::Tensor &v, torch::Tensor &y) {
    const int B = k.size(0);
    const int T = k.size(1);
    const int C = k.size(2);
    cuda_forward(B, T, C, w.data_ptr<float>(), u.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), y.data_ptr<float>());
}
void forward_bf16(torch::Tensor &w, torch::Tensor &u, torch::Tensor &k, torch::Tensor &v, torch::Tensor &y) {
    const int B = k.size(0);
    const int T = k.size(1);
    const int C = k.size(2);
    cuda_forward_bf16(B, T, C, w.data_ptr<float>(), u.data_ptr<bf16>(), k.data_ptr<bf16>(), v.data_ptr<bf16>(), y.data_ptr<bf16>());
}
void forward_with_state(torch::Tensor &w, torch::Tensor &u, torch::Tensor &k, torch::Tensor &v, torch::Tensor &y, torch::Tensor &s) {
    const int B = k.size(0);
    const int T = k.size(1);
    const int C = k.size(2);
    cuda_forward_with_state(B, T, C, w.data_ptr<float>(), u.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), y.data_ptr<float>(), s.data_ptr<float>());
}
void forward_with_state_bf16(torch::Tensor &w, torch::Tensor &u, torch::Tensor &k, torch::Tensor &v, torch::Tensor &y, torch::Tensor &s) {
    const int B = k.size(0);
    const int T = k.size(1);
    const int C = k.size(2);
    cuda_forward_with_state_bf16(B, T, C, w.data_ptr<float>(), u.data_ptr<bf16>(), k.data_ptr<bf16>(), v.data_ptr<bf16>(), y.data_ptr<bf16>(), s.data_ptr<float>());
}
void backward(torch::Tensor &w, torch::Tensor &u, torch::Tensor &k, torch::Tensor &v, torch::Tensor &y, torch::Tensor &gy, torch::Tensor &gw, torch::Tensor &gu, torch::Tensor &gk, torch::Tensor &gv) {
    const int B = k.size(0);
    const int T = k.size(1);
    const int C = k.size(2);
    cuda_backward(B, T, C, w.data_ptr<float>(), u.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), y.data_ptr<float>(), gy.data_ptr<float>(), gw.data_ptr<float>(), gu.data_ptr<float>(), gk.data_ptr<float>(), gv.data_ptr<float>());
}
void backward_bf16(torch::Tensor &w, torch::Tensor &u, torch::Tensor &k, torch::Tensor &v, torch::Tensor &y, torch::Tensor &gy, torch::Tensor &gw, torch::Tensor &gu, torch::Tensor &gk, torch::Tensor &gv) {
    const int B = k.size(0);
    const int T = k.size(1);
    const int C = k.size(2);
    cuda_backward_bf16(B, T, C, w.data_ptr<float>(), u.data_ptr<bf16>(), k.data_ptr<bf16>(), v.data_ptr<bf16>(), y.data_ptr<bf16>(),
        gy.data_ptr<bf16>(), gw.data_ptr<bf16>(), gu.data_ptr<bf16>(), gk.data_ptr<bf16>(), gv.data_ptr<bf16>());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "wkv forward");
    m.def("forward_bf16", &forward_bf16, "wkv forward bf16");
    m.def("forward_with_state", &forward_with_state, "wkv forward with state");
    m.def("forward_with_state_bf16", &forward_with_state_bf16, "wkv forward with state bf16");
    m.def("backward", &backward, "wkv backward");
    m.def("backward_bf16", &backward_bf16, "wkv backward bf16");
}

TORCH_LIBRARY(wkv, m) {
    m.def("forward", forward);
    m.def("forward_bf16", forward_bf16);
    m.def("forward_with_state", forward_with_state);
    m.def("forward_with_state_bf16", forward_with_state_bf16);
    m.def("backward", backward);
    m.def("backward_bf16", backward_bf16);
}
