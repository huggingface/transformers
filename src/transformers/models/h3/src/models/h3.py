import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from transformers.models.h3.src.models.ss_kernel import SSKernel


try:
    from transformers.models.h3.src.ops.fftconv import fftconv_func
except ImportError:
    fftconv_func = None


@torch.jit.script
def mul_sum(q, y):
    return (q * y).sum(dim=1)


class H3(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=64,
        l_max=None,
        head_dim=1,
        use_fast_fftconv=False,
        dropout=0.0,  # Just to absorb the kwarg
        layer_idx=None,
        device=None,
        dtype=None,
        # SSM Kernel arguments
        **kernel_args,
    ):
        """
        d_state: the dimension of the state, also denoted by N l_max: the maximum kernel length, also denoted by L. Set
        l_max=None to always use a global kernel

        See the class .kernel.SSKernel for the kernel constructor which accepts kernel_args. Relevant options that are
        worth considering and tuning include "mode" + "measure", "dt_min", "dt_max", "lr"

        Other options are all experimental and should not need to be configured
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.head_dim = head_dim
        assert d_model % head_dim == 0
        self.H = d_model // head_dim
        self.N = d_state
        self.L = l_max
        self.layer_idx = layer_idx
        self.use_fast_fftconv = use_fast_fftconv
        if self.use_fast_fftconv:
            assert fftconv_func is not None, "Need to install fftconv"

        self.q_proj = nn.Linear(self.d_model, self.d_model, **factory_kwargs)
        self.k_proj = nn.Linear(self.d_model, self.d_model, **factory_kwargs)
        self.v_proj = nn.Linear(self.d_model, self.d_model, **factory_kwargs)

        # TODO: SSKernel doesn't take device argument yet
        self.ssm_k_kernel = SSKernel(self.d_model, N=d_state, L=self.L, mode="shift", lr=kernel_args.get("lr", None))
        self.ssm_k_D = nn.Parameter(torch.randn(self.d_model))
        # S4D Kernel
        self.kernel = SSKernel(self.H, N=self.N, L=self.L, channels=1, **kernel_args)
        self.D = nn.Parameter(torch.randn(self.H, **factory_kwargs))

        # Pointwise
        # position-wise output transform to mix features
        # Don't use FusedDense since the layout is H first
        self.output_linear = nn.Linear(self.d_model, self.d_model)

    def forward(self, u, inference_params=None):
        """
        u: (B L H)

        Returns: same shape as u
        """
        L_og = u.size(-2)
        if self.use_fast_fftconv and L_og % 2 != 0:
            u = F.pad(u, (0, 0, 0, 1))
        L = u.size(-2)

        use_fast_fftconv = self.use_fast_fftconv and inference_params is None

        state_k, state = None, None
        if inference_params is not None:
            assert self.layer_idx is not None
            if self.layer_idx not in inference_params.key_value_memory_dict:
                batch_shape = (u.shape[0] * self.head_dim * self.head_dim,)
                state_k = self.ssm_k_kernel.default_state(*batch_shape)
                state = self.kernel.default_state(*batch_shape)
                inference_params.key_value_memory_dict[self.layer_idx] = (state_k, state)
            else:
                state_k, state = inference_params.key_value_memory_dict[self.layer_idx]
            if inference_params.sequence_len_offset == 0:
                self.ssm_k_kernel._setup_step()
                self.kernel._setup_step()

        if inference_params is not None and inference_params.sequence_len_offset > 0:
            y, next_state_k, next_state = self.step(u, state_k, state)
            inference_params.key_value_memory_dict[self.layer_idx][0].copy_(next_state_k)
            inference_params.key_value_memory_dict[self.layer_idx][1].copy_(next_state)
            return y

        # Compute SS Kernel
        L_kernel = L if self.L is None else min(L, self.L)
        ssm_kernel, k_state = self.kernel(L=L_kernel, state=state, rate=1.0)  # (C H L) (B C H L)
        ssm_kernel = rearrange(ssm_kernel, "1 h l -> h l")

        u = rearrange(u, "b l h -> (b l) h")
        dtype = self.q_proj.weight.dtype if not torch.is_autocast_enabled() else torch.get_autocast_gpu_dtype()
        q = self.q_proj.weight @ u.T + self.q_proj.bias.to(dtype).unsqueeze(-1)
        k = self.k_proj.weight @ u.T + self.k_proj.bias.to(dtype).unsqueeze(-1)
        v = self.v_proj.weight @ u.T + self.v_proj.bias.to(dtype).unsqueeze(-1)
        q, k, v = [rearrange(x, "h (b l) -> b h l", l=L) for x in [q, k, v]]

        k_og = k
        ssm_k_kernel, _ = self.ssm_k_kernel(L=L_kernel, state=state_k, rate=1.0)  # (C H L) (B C H L)
        ssm_k_kernel = rearrange(ssm_k_kernel, "1 h l -> h l")
        if not use_fast_fftconv:
            fft_size = L_kernel + L
            ssm_k_kernel_f = torch.fft.rfft(ssm_k_kernel, n=fft_size)  # (H 2L)
            k_f = torch.fft.rfft(k.to(ssm_kernel.dtype), n=fft_size)  # (B H 2L)
            shift_k_out = torch.fft.irfft(ssm_k_kernel_f * k_f, n=fft_size)[..., :L]
            k = shift_k_out + rearrange(self.ssm_k_D, "h -> h 1") * k
        else:
            dropout_mask = None
            # No GeLU after the SSM
            # We want output_hbl=True so that k has the same layout as q and v for the next
            # fftconv
            k = fftconv_func(k, ssm_k_kernel, self.ssm_k_D, dropout_mask, False, False, True)
            # This line below looks like it doesn't do anything, but it gets the stride right
            # for the case batch_size=1. In that case k has stride (L, L, 1), but q and v has
            # stride (H * L, L, 1). The two strides are equivalent because batch_size=1, but
            # the C++ code doesn't like that.
            k = rearrange(rearrange(k, "b h l -> h b l"), "h b l -> b h l")

        if not use_fast_fftconv:
            fft_size = L_kernel + L
            # kv = k * v
            kv = rearrange(k, "b (h d1) l -> b d1 1 h l", d1=self.head_dim) * rearrange(
                v, "b (h d2) l -> b 1 d2 h l", d2=self.head_dim
            )  # b d1 d2 h l
            kv_f = torch.fft.rfft(kv.to(dtype=ssm_kernel.dtype), n=fft_size) / fft_size
            ssm_kernel_f = torch.fft.rfft(ssm_kernel, n=fft_size)  # h L+1
            y = torch.fft.irfft(kv_f * ssm_kernel_f, n=fft_size, norm="forward")[..., :L]  # b d1 d2 h l
            y = y + kv * self.D.unsqueeze(-1)  # b d1 d2 h l
            q = rearrange(q, "b (h d1) l -> b d1 1 h l", d1=self.head_dim)
            # einsum is way slower than multiply and then sum.
            if self.head_dim > 1:
                y = mul_sum(y, q)
                y = rearrange(y, "b d h l -> b (d h) l")
            else:
                y = rearrange(y * q, "b 1 1 h l -> b h l")
        else:
            dropout_mask = None
            # No GeLU after the SSM
            # Set output_hbl_layout=True since we'll be doing a matmul right after
            y = fftconv_func(
                k, ssm_kernel, self.D, dropout_mask, False, torch.is_autocast_enabled(), True, v, self.head_dim, q
            )

        y = rearrange(y, "b h l -> b l h")

        if state is not None:
            assert inference_params is not None
            # TODO: This doesn't ever happen?
            # if inference_params.sequence_len_offset > 0:
            #     y = y + k_state
            inference_params.key_value_memory_dict[self.layer_idx][0].copy_(
                self.ssm_k_kernel.forward_state(k_og, state_k)
            )
            inference_params.key_value_memory_dict[self.layer_idx][1].copy_(
                self.kernel.forward_state(rearrange(kv, "b d1 d2 h l -> (b d1 d2) h l"), state)
            )

        # y could be in fp32 because of the SSMs
        if not torch.is_autocast_enabled():
            y = y.to(dtype=self.output_linear.weight.dtype)
        y = self.output_linear(y)
        if L_og < L:
            y = y[:, :L_og, :]

        return y

    def step(self, u, state_k, state):
        q, k, v = self.q_proj(u), self.k_proj(u), self.v_proj(u)
        shift_k, next_state_k = self.ssm_k_kernel.step(rearrange(k, "b 1 h -> b h"), state_k)
        k = shift_k + k * self.ssm_k_D
        # kv = k * v
        kv = rearrange(k, "b 1 (h d1) -> b d1 1 h", d1=self.head_dim) * rearrange(
            v, "b 1 (h d2) -> b 1 d2 h", d2=self.head_dim
        )  # b d1 d2 h
        y, next_state = self.kernel.step(rearrange(kv, "b d1 d2 h -> (b d1 d2) h"), state)
        y = rearrange(y, "(b d1 d2) 1 h -> b d1 d2 h", d1=self.head_dim, d2=self.head_dim) + kv * self.D
        q = rearrange(q, "b 1 (h d1) -> b d1 1 h", d1=self.head_dim)
        if self.head_dim > 1:
            y = mul_sum(y, q)
            y = rearrange(y, "b d h l -> b (d h) l")
        else:
            y = rearrange(y * q, "b 1 1 h -> b 1 h")
        # y could be in fp32 because of the SSMs
        if not torch.is_autocast_enabled():
            y = y.to(dtype=self.output_linear.weight.dtype)
        return self.output_linear(y), next_state_k, next_state
