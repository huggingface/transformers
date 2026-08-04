# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Which GGUF weights keep their bytes packed, and the modules that hold them.

`get_gguf_plan` answers the question from the file alone — header, arch table and kernel — so it
needs no model instance. `replace_with_gguf_modules` then settles which of the plan's entries really
stay packed: only a module that can hold blocks gets swapped, and the rest are dequantized at load.

Keeping a weight packed is only worth it when a fused dequant-matmul kernel can read it: without one,
unpacking the whole weight on every forward costs more than unpacking it once at load. That is why
`get_gguf_plan` asks `kernels.py` first, and why these modules exist only where the answer was yes.
"""

import torch
from torch import nn

from ...core_model_loading import WeightConverter, WeightRenaming, WeightTransform
from ...utils import logging
from .dequant import GGML_BLOCK
from .gguf_conversion_mapping import GGUF_ARCHS
from .kernels import MAX_GEMV_ROWS, dequantize_blocks, get_gguf_kernel, mul_mat_vec
from .reader import read_gguf_architecture, read_gguf_tensor_types


logger = logging.get_logger(__name__)


def is_gguf_arch_supported(gguf_path: str) -> bool:
    """Whether this file's architecture has a mapping here.

    Architectures without one still load through the legacy GGUF loader, which dequantizes
    everything; this path is opt-in per architecture as they are migrated.
    """
    return read_gguf_architecture(gguf_path) in GGUF_ARCHS


def get_gguf_conversion_mapping(gguf_arch: str, config) -> list[WeightTransform]:
    """Weight transforms turning a GGUF checkpoint of `gguf_arch` into transformers weights."""
    if gguf_arch not in GGUF_ARCHS:
        raise ValueError(f"GGUF architecture {gguf_arch!r} is not supported yet. Supported: {sorted(GGUF_ARCHS)}.")
    return GGUF_ARCHS[gguf_arch](config)


class GgufPlan(dict):
    """`{param_name: ggml_type}` that also remembers each entry's GGUF tensor name."""

    def __init__(self):
        super().__init__()
        self.gguf_name: dict[str, str] = {}


def get_gguf_plan(gguf_file: str, config) -> GgufPlan:
    """`{param_name: ggml_type}` for weights whose bytes can stay packed, plus their gguf names.

    Empty when nothing can stay packed: the architecture has no mapping (so the legacy loader handles
    the file), or no fused dequant-matmul kernel is available — without one, unpacking on every
    forward costs more than unpacking once at load.

    Callers that already know they want dense weights should not call this at all.
    """
    if not is_gguf_arch_supported(gguf_file):
        return GgufPlan()
    kernel = get_gguf_kernel("cuda" if torch.cuda.is_available() else "cpu")
    if kernel is None:
        logger.warning(
            "No GGUF matmul kernel is available, so weights will be dequantized at load time. "
            "Set TRANSFORMERS_GGUF_KERNEL_LIB to a built extension to keep them packed."
        )
        return GgufPlan()

    arch = read_gguf_architecture(gguf_file)
    types = read_gguf_tensor_types(gguf_file)
    mapping = get_gguf_conversion_mapping(arch, config)
    renamings = [m for m in mapping if isinstance(m, WeightRenaming)]
    converters = [m for m in mapping if isinstance(m, WeightConverter)]

    plan = GgufPlan()
    for gguf_name, ggml_type in types.items():
        if ggml_type not in GGML_BLOCK or not kernel.supports(ggml_type):
            continue
        renamed = gguf_name
        for renaming in renamings:
            renamed, _ = renaming.rename_source_key(renamed)
        # every conversion applied to this tensor must be safe on packed bytes
        for converter in converters:
            _, pattern = converter.rename_source_key(renamed)
            if pattern is not None:
                if not all(getattr(op, "supports_packed", False) for op in converter.operations):
                    renamed = None  # needs dense data (e.g. a column permute): dequantize it
                break
        if renamed is not None:
            plan[renamed] = ggml_type
            plan.gguf_name[renamed] = gguf_name
    return plan


class GgufLinear(nn.Module):
    """`nn.Linear` whose weight stays as GGUF blocks: `(out_features, bytes_per_row)` uint8."""

    dequant_chunk_elems = 64 << 20  # ~128 MB of bf16 per chunk

    def __init__(self, in_features: int, out_features: int, ggml_type: int, kernel=None, device=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ggml_type = ggml_type
        block_elems, block_bytes = GGML_BLOCK[ggml_type]
        bytes_per_row = in_features // block_elems * block_bytes
        self.weight = nn.Parameter(
            torch.empty((out_features, bytes_per_row), dtype=torch.uint8, device=device), requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pick a path from the row count: the fused kernel for a few rows, unpack-and-matmul beyond.

        The kernel reads the blocks straight out of the weight and never materializes it, but it only
        goes up to `MAX_GEMV_ROWS`. There is no fused kernel to use above that — llama.cpp's quantized
        gemm needs a ggml backend context to allocate from, so it is not ported — so prefill unpacks
        instead, which is what llama.cpp also does once a batch is large enough to amortize it.

        The kernel writes f32 whatever `x` was, so the cast back is here rather than inside the op: as
        a plain `aten` op, inductor can fuse it into whatever consumes the output.

        It is also CUDA-only, so a module whose blocks are offloaded to CPU takes the unpacking path
        at every row count. Unpacking runs wherever the blocks are.
        """
        flat = x.reshape(-1, self.in_features)
        if flat.shape[0] <= MAX_GEMV_ROWS and self.weight.is_cuda:
            out = mul_mat_vec(self.weight, flat, self.ggml_type, self.out_features)
        else:
            out = self._unpack_matmul(flat)
        return out.reshape(*x.shape[:-1], self.out_features).to(x.dtype)

    def _unpack_matmul(self, flat: torch.Tensor) -> torch.Tensor:
        """Matmul against the weight unpacked a row chunk at a time, so it is never fully materialized.

        A row slice of the blocks stands on its own because GGUF quantizes each row independently, and
        rows of the weight are columns of the result. Chunking bounds the transient: unpacking a large
        tied `lm_head` in one go costs more than the packed model itself.
        """
        rows_per_chunk = max(1, self.dequant_chunk_elems // self.in_features)
        out = torch.empty(flat.shape[0], self.out_features, dtype=flat.dtype, device=flat.device)
        for start in range(0, self.out_features, rows_per_chunk):
            rows = self.weight[start : start + rows_per_chunk]
            chunk = dequantize_blocks(rows, self.ggml_type, rows.shape[0], self.in_features, flat.dtype)
            out[:, start : start + rows_per_chunk] = flat @ chunk.T
        return out

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, ggml_type={self.ggml_type}"


class GgufEmbedding(nn.Module):
    """`nn.Embedding` whose table stays as GGUF blocks.

    GGUF quantizes each row independently, so gathering the rows for a batch of ids is exact on the
    packed bytes: only the gathered `(n_tokens, embedding_dim)` slice is ever dequantized, never the
    whole vocabulary. For a 248k-token vocabulary that is the difference between 0.5 GB of blocks and
    1.3 GB of bf16.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, ggml_type: int, dtype=None, device=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.ggml_type = ggml_type
        self.compute_dtype = dtype or torch.get_default_dtype()
        block_elems, block_bytes = GGML_BLOCK[ggml_type]
        self.weight = nn.Parameter(
            torch.empty(
                (num_embeddings, embedding_dim // block_elems * block_bytes), dtype=torch.uint8, device=device
            ),
            requires_grad=False,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Gather these ids' rows out of the packed bytes, then dequantize only those."""
        flat = input_ids.reshape(-1)
        rows = self.weight.index_select(0, flat).contiguous()
        out = dequantize_blocks(rows, self.ggml_type, flat.numel(), self.embedding_dim, self.compute_dtype)
        return out.reshape(*input_ids.shape, self.embedding_dim)

    def extra_repr(self) -> str:
        return f"{self.num_embeddings}, {self.embedding_dim}, ggml_type={self.ggml_type}"


def replace_with_gguf_modules(model, plan: GgufPlan, dtype: "torch.dtype") -> dict[str, nn.Module]:
    """Replace every module named in `plan` with one that holds GGUF blocks; return `{gguf_name: module}`.

    Keyed by GGUF tensor name because that is what the caller owes the reader: the names whose bytes
    must come back as raw blocks rather than dequantized.

    Only `<module>.weight` is looked up, since that is the one parameter `GgufLinear` and
    `GgufEmbedding` can hold. A plan entry that is not a module weight — the arch table also renames
    tensors to plain parameters, like `linear_attn.A_log` — matches nothing here and is dequantized,
    which is what we want for anything with no packed module.

    Likewise a planned weight whose module cannot hold blocks: none at that name, a `Linear` with a
    bias (`GgufLinear` has none), or anything that is not exactly an `nn.Linear`/`nn.Embedding` — a
    subclass may compute more than a plain matmul, so it is left alone rather than assumed equivalent.
    That is all decided here, by looking at the modules, so no name rule or per-architecture list is
    needed for it.
    """
    replaced: dict[str, nn.Module] = {}
    for module_name, module in model.named_modules():
        param_name = f"{module_name}.weight"
        ggml_type = plan.get(param_name)
        if ggml_type is None:
            continue

        new_module = None
        with torch.device("meta"):
            if type(module) is nn.Linear and module.bias is None:
                new_module = GgufLinear(module.in_features, module.out_features, ggml_type)
            elif type(module) is nn.Embedding:
                new_module = GgufEmbedding(module.num_embeddings, module.embedding_dim, ggml_type, dtype=dtype)
        if new_module is not None:
            model.set_submodule(module_name, new_module)
            replaced[plan.gguf_name[param_name]] = new_module

    # An empty plan is not worth a warning: either the file holds nothing quantized, or `get_gguf_plan`
    # already said why. Weights that could have stayed packed and found no module to hold them are.
    if plan and not replaced:
        logger.warning(
            "You are loading your model from a GGUF file but no module could keep its weights packed."
            " Every quantized tensor will be dequantized at load time."
        )
    return replaced


def retie_gguf_lm_head(model, embedding: GgufEmbedding) -> None:
    """Repair an `lm_head` that ordinary weight tying pointed at packed blocks.

    A tied `lm_head` has no tensor of its own in the file — it reuses `token_embd`. `tie_weights` has
    already run by now and assigned the embedding's uint8 block buffer onto the head, which is still a
    dense `nn.Linear` and cannot compute with it; the fix is a `GgufLinear` over that same buffer.

    Sharing that exact storage is the test, rather than `config.tie_word_embeddings`: it is the
    observed outcome of tying, so a head holding its own loaded `output.weight` is left alone even
    when the config claims the weights are tied and `tie_weights` declined to tie them.
    """
    head = getattr(model, "lm_head", None)
    if isinstance(head, nn.Linear) and head.weight is embedding.weight:
        tied = GgufLinear(embedding.embedding_dim, embedding.num_embeddings, embedding.ggml_type, device="meta")
        tied.weight = embedding.weight  # same blocks, no copy
        model.lm_head = tied
