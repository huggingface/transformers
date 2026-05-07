# Copyright 2026 The HuggingFace Inc. team
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
"""Resolves a `ContinuousBatchingConfig` into a fully-specified config ready for cache and runner creation. Each
helper mutates the config in place; `resolve_continuous_batching_config` orchestrates them in the required order."""

from copy import deepcopy
from math import ceil

import torch

from ...configuration_utils import PretrainedConfig
from ...generation.configuration_utils import CompileConfig, ContinuousBatchingConfig
from ...modeling_flash_attention_utils import lazy_import_paged_flash_attention
from ...utils.generic import is_flash_attention_requested
from .requests import logger
from .utils import WorkloadHints


def resolve_continuous_batching_config(
    config: PretrainedConfig,
    cb_config: ContinuousBatchingConfig,
    workload_hints: WorkloadHints | None,
    has_logit_processors: bool,
) -> ContinuousBatchingConfig:
    """Returns a deep-copied and fully-resolved `ContinuousBatchingConfig`. The original `cb_config` is not mutated."""
    cb_config = deepcopy(cb_config)

    # Look at whether the user explicitly asked for the decode fast path before we assign a default value
    user_requested_decode_path = cb_config.max_blocks_per_request is not None
    # Same for cuda graphs, if the user signals they want CUDA graphs via any padding/cached-graph parameter
    cuda_graph_requested = any(
        [cb_config.q_padding_interval_size, cb_config.kv_padding_interval_size, cb_config.max_cached_graphs]
    )

    # Resolve missing attributes for which we have hints. Must happen before no-hints resolve.
    resolve_using_hints(cb_config, workload_hints)

    # Resolve remaining missing attributes. Must happen before decode fast path is checked.
    resolve_without_hints(cb_config)

    # Check if the decode fast path is available. Must happen before the compile config.
    ensure_decode_fast_path_is_available(config, cb_config, user_requested_decode_path)

    # Decide if compile should be used. Must happen before CUDA graphs are decided.
    resolve_compile_configs(
        cb_config=cb_config,
        fallback_compile_config=getattr(config, "compile_config", None),
        is_flash_attn=is_flash_attention_requested(config),
        decode_fast_path_available=cb_config.max_blocks_per_request > 0,
    )

    # Decide if CUDA graphs should be used. Should happen after compile configs are decided.
    is_attn_mask_needed = not is_flash_attention_requested(config)
    decide_use_cuda_graphs(
        cb_config=cb_config, is_attn_mask_needed=is_attn_mask_needed, cuda_graph_requested=cuda_graph_requested
    )

    # Decide if asynchronous batching should be used. Should happen after CUDA graphs are decided.
    decide_use_async_batching(cb_config=cb_config, is_attn_mask_needed=is_attn_mask_needed)

    # Resolve the max memory percent. This can happen anytime before cache creation.
    resolve_max_memory_percent(cb_config=cb_config, has_logit_processors=has_logit_processors)
    return cb_config


def resolve_using_hints(cb_config: ContinuousBatchingConfig, workload_hints: WorkloadHints | None) -> None:
    """Fills `max_blocks_per_request` from the workload hints, when the user did not set it explicitly."""
    # The max number of blocks per request is an even number large enough to hold the max request length
    if cb_config.max_blocks_per_request is None and workload_hints is not None:
        max_sequence_length = workload_hints.max_prompt_length + workload_hints.max_generated_length
        if max_sequence_length > 0:
            blocks_per_request = int(ceil(max_sequence_length / cb_config.block_size)) + 1
            cb_config.max_blocks_per_request = blocks_per_request + (blocks_per_request % 2)


def resolve_without_hints(cb_config: ContinuousBatchingConfig) -> None:
    """Fills any remaining unset/sentinel attribute with a fallback default."""
    if cb_config.max_blocks_per_request is None:
        cb_config.max_blocks_per_request = 32
    if cb_config.q_padding_interval_size == 0:
        cb_config.q_padding_interval_size = 64
    if cb_config.kv_padding_interval_size == 0:
        cb_config.kv_padding_interval_size = 64 * 256  # 64 blocks of 256 tokens ie. 16384 tokens
    if cb_config.max_cached_graphs == 0:
        cb_config.max_cached_graphs = 32


def ensure_decode_fast_path_is_available(
    config: PretrainedConfig, cb_config: ContinuousBatchingConfig, user_requested: bool
) -> None:
    """Ensures the decode fast path is available. If it is not, set the max blocks per request to 0. If it is
    available, and no user-provided max blocks per request, set it to the fallback default."""
    # Then, if the decode fast path is not turned off, check if it is available
    if cb_config.max_blocks_per_request != 0:
        # NOTE: block table should be available with FA2 and FA3, but there seems to be an issue with FA2 atm
        if is_flash_attention_requested(config, version=3):
            flash_attn_with_kvcache = lazy_import_paged_flash_attention(config._attn_implementation)[1]
            conditions = [
                torch.cuda.is_available(),  # Block table is only supported on CUDA
                flash_attn_with_kvcache is not None,  # The `flash_attn_with_kvcache` fn is needed
            ]
            # Throw a warning only if the decode fast path was requested by the user
            if not all(conditions):
                if user_requested:
                    logger.warning(
                        f"Although {cb_config.max_blocks_per_request = }, the decode fast path is not available "
                        f"because at least one condition is not met: {conditions}."
                    )
                cb_config.max_blocks_per_request = 0
        # Specific warning for attn implementation other than FA3
        else:
            if user_requested:
                logger.warning(
                    f"Although {cb_config.max_blocks_per_request = }, the decode fast path is not available "
                    f"because the attention implementation is not FA3. Got {config._attn_implementation = }."
                )
            cb_config.max_blocks_per_request = 0


def resolve_compile_configs(
    cb_config: ContinuousBatchingConfig,
    fallback_compile_config: CompileConfig | None,
    is_flash_attn: bool,
    decode_fast_path_available: bool,
) -> None:
    """Resolve if the compile configs for varlen and decode paths, modifying these attributes in place if needed.
    Default config use full compile over regional compile, because the throughput is significantly higher (~15%)"""
    # For each config, priority is: explicit config, default config, fallback config, None
    if cb_config.varlen_compile_config is None:
        if cb_config.use_default_compile_configs:
            # We don't use compile with flash varlen, because max_seqlen_k is volatile and introduces recompilations
            if is_flash_attn:
                varlen_config = None
            else:
                varlen_config = CompileConfig(mode="max-autotune-no-cudagraphs", fullgraph=True, dynamic=True)
        elif fallback_compile_config is not None:
            varlen_config = fallback_compile_config
        else:
            varlen_config = None
    else:
        varlen_config = cb_config.varlen_compile_config

    if cb_config.decode_compile_config is None:
        if cb_config.use_default_compile_configs:
            # Paged attention is wrapped in @torch.compiler.disable so we can't use fullgraph
            decode_config = CompileConfig(mode="max-autotune-no-cudagraphs", fullgraph=False, dynamic=False)
        elif fallback_compile_config is not None:
            decode_config = fallback_compile_config
        else:
            decode_config = None
    else:
        decode_config = cb_config.decode_compile_config

    # For decode, we throw a warning if the fast decode path is not available and a compile config was found
    if not decode_fast_path_available and cb_config.decode_compile_config is not None:
        decode_config = None
        logger.warning("A decode_compile_config was set but fast decode path is not available. Ignoring it.")

    # Log what will be compiled
    if varlen_config is not None:
        logger.info(f"Varlen path will be compiled with {varlen_config.to_dict()}")
    if decode_config is not None:
        logger.info(f"Decode path will be compiled with {decode_config.to_dict()}")
    # Modify in place
    cb_config.varlen_compile_config = varlen_config
    cb_config.decode_compile_config = decode_config


def decide_use_cuda_graphs(
    cb_config: ContinuousBatchingConfig, is_attn_mask_needed: bool, cuda_graph_requested: bool
) -> None:
    """Decides whether or not to use cuda graphs for continuous batching. If the user specified this in the config
    or if they specified a parameter related to cuda graphs, they are turned on. Otherwise, we use a heuristic
    based on the attention implementation: we turn on cuda graphs if and only if no attention mask is needed.

    This function modifies the `use_cuda_graph` attribute of the config in place, to a tuple of booleans.
    """
    # If cuda is not available, we cannot use cuda graphs
    if not torch.cuda.is_available():
        intended_use_cuda_graph = any(cb_config.cuda_graph_booleans)
        if intended_use_cuda_graph:  # throw a warning only if the user intended to use cuda graphs
            logger.warning(
                f"{cb_config.use_cuda_graph = } but {torch.cuda.is_available() = }: turning off cuda graphs"
            )
        cb_config.use_cuda_graph = (False, False)

    # Else if use_cuda_graph is specified, we follow the user's choice and make sure it is a tuple of booleans
    elif cb_config.use_cuda_graph is not None:
        if isinstance(cb_config.use_cuda_graph, bool):
            cb_config.use_cuda_graph = (cb_config.use_cuda_graph, cb_config.use_cuda_graph)

    # Else if the user specified a parameter related to cuda graphs, we activate cuda graphs
    elif cuda_graph_requested:
        cb_config.use_cuda_graph = (True, True)

    # Otherwise we have a default heuristic based on the attention implementation:
    # attention implementations where an attention mask is needed suffer a lot more from the padding associated
    # with cuda graphs, so default is to turn cuda graphs off for those implementations
    else:
        use_cuda_graph = []
        for compile_config in [cb_config.varlen_compile_config, cb_config.decode_compile_config]:
            # No compile config means we decide on attention
            if compile_config is None:
                use_cuda_graph.append(not is_attn_mask_needed)
                continue
            # Otherwise we disable cuda graphs if the compile config uses them
            options = torch._inductor.list_mode_options().get(compile_config.mode, compile_config.options)
            compile_uses_cudagraphs = options.get("triton.cudagraphs", False)
            if compile_uses_cudagraphs:
                logger.warning(
                    f"Compile config {compile_config.mode = } uses cudagraphs, which usually does not work well with "
                    "continuous batching. We recommend using mode 'default' or 'max-autotune-no-cudagraphs' instead."
                )
            use_cuda_graph.append(not compile_uses_cudagraphs and not is_attn_mask_needed)
        cb_config.use_cuda_graph = tuple(use_cuda_graph)

    logger.info(f"Using cuda graphs for (varlen, decode) paths: {cb_config.use_cuda_graph}")


def decide_use_async_batching(cb_config: ContinuousBatchingConfig, is_attn_mask_needed: bool) -> None:
    """Returns whether or not to use asynchronous batching for continuous batching. If the user specified this in
    the config, we follow their choice. Otherwise, we turn on asynchronous batching if and only if CUDA graphs are
    turned on and no attention mask is needed.

    This function modifies the `use_async_batching` attribute of the config in place.
    """
    # If the user specifies to use async or not, no need to decide ourselves
    if cb_config.use_async_batching is None:
        use_cuda_graphs = any(cb_config.cuda_graph_booleans)
        cb_config.use_async_batching = use_cuda_graphs and not is_attn_mask_needed
        logger.info(
            f"No behavior specified for use_async_batching, choosing {cb_config.use_async_batching = } because "
            f"{use_cuda_graphs = } and {is_attn_mask_needed = }. If you want to save memory, you can "
            "disable asynchronous batching but it will degrade performance."
        )


def resolve_max_memory_percent(cb_config: ContinuousBatchingConfig, has_logit_processors: bool) -> None:
    if cb_config.max_memory_percent is None:
        cb_config.max_memory_percent = 0.8 if has_logit_processors else 0.9
