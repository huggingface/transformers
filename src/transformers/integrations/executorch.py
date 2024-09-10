# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import torch

from transformers import (
    PreTrainedModel,
    StaticCache,
)
from transformers.pytorch_utils import is_torch_greater_or_equal_than_2_3


class TorchExportableModuleWithStaticCache(torch.nn.Module):
    """
    A wrapper module designed to make a `PreTrainedModel` exportable with `torch.export`,
    specifically for use with static caching. This module ensures that the exported model
    is compatible with further lowering and execution in `ExecuTorch`.

    Note:
        This class is specifically designed to support export process using `torch.export`
        in a way that ensures the model can be further lowered and run efficiently in `ExecuTorch`.
    """

    def __init__(self, model: PreTrainedModel):
        """
        Initializes the wrapper module with the pretrained model.

        Args:
            model (`PreTrainedModel`): The pretrained model to wrap. The model must have caching
            enabled and use a 'static' caching implementation.

        Raises:
            AssertionError: If the pretrained model does not have caching enabled or if it does
            not use a 'static' caching implementation in `model.generation_config`.
        """
        super().__init__()

        # Sanity checks
        if model.generation_config is None:
            raise AssertionError(
                "The model must have a generation config to be exported with static caching. "
                "Please set `generation_config`."
            )

        if not model.generation_config.use_cache:
            raise AssertionError(
                "The model must have caching enabled to be exported with static caching. "
                "Please set `generation_config.use_cache=True`."
            )

        if model.generation_config.cache_implementation != "static":
            raise AssertionError(
                "The model must use a 'static' caching implementation to be exported with static caching. "
                "Please set `generation_config.cache_implementation='static'`."
            )

        self.model = model
        self.static_cache = StaticCache(
            config=self.model.config,
            batch_size=self.model.generation_config.cache_config.batch_size,
            max_cache_len=self.model.generation_config.cache_config.max_cache_len,
            dtype=self.model.config.torch_dtype,
        )
        self.is_causal = any("CausalLM" in arch for arch in self.model.config.architectures)
        if self.is_causal:
            causal_mask = torch.tril(
                torch.ones(
                    self.static_cache.max_cache_len,
                    self.static_cache.max_cache_len,
                    dtype=torch.bool,
                )
            )
            self.register_buffer("mask", causal_mask, persistent=False)

    def forward(self, input_ids: torch.Tensor, cache_position: torch.Tensor):
        """
        Forward pass of the module, which is compatible with the ExecuTorch runtime.

        Args:
            input_ids (`torch.Tensor`): Tensor representing current input token id to the module.
            cache_position (`torch.Tensor`): Tensor representing current input position in the cache.

        Returns:
            torch.Tensor: Logits output from the model.

        This forward adapter serves two primary purposes:

        1. **Making the Model `torch.export`-Compatible**:
            The adapter hides unsupported objects, such as the `Cache`, from the graph inputs and outputs,
            enabling the model to be exportable using `torch.export` without encountering issues.

        2. **Ensuring Compatibility with `ExecuTorch` runtime**:
            The adapter matches the model's forward signature with that in `executorch/extension/llm/runner`,
            ensuring that the exported model can be executed in `ExecuTorch` out-of-the-box.
        """
        _, seqlen = input_ids.shape
        attn_mask = self.mask[cache_position, :seqlen] if self.is_causal else None
        outs = self.model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            position_ids=cache_position.unsqueeze(0),
            cache_position=cache_position,
            past_key_values=self.static_cache,
            use_cache=True,
        )
        return outs.logits


def convert_and_export_with_cache(
    model: PreTrainedModel,
    example_input_ids: torch.Tensor = None,
    example_cache_position: torch.Tensor = None,
):
    """
    Convert a `PreTrainedModel` into an exportable module and export it using `torch.export`,
    ensuring the exported model is compatible with `ExecuTorch`.

    Args:
        model (`PreTrainedModel`): The pretrained model to be exported.
        example_input_ids (`torch.Tensor`): Example input token id used by `torch.export`.
        example_cache_position (`torch.Tensor`): Example current cache position used by `torch.export`.

    Returns:
        Exported program (`torch.export.ExportedProgram`): The exported program generated via `torch.export`.
    """

    if not is_torch_greater_or_equal_than_2_3:
        raise ImportError("torch >= 2.3 is required.")

    import torch.export._trace

    with torch.no_grad():
        # TODO: The default inputs only work for text models. We need to add support for vision/audio models.
        example_input_ids = (
            example_input_ids if example_input_ids is not None else torch.tensor([[1]], dtype=torch.long)
        )
        example_cache_position = (
            example_cache_position if example_cache_position is not None else torch.tensor([0], dtype=torch.long)
        )

        # Due to issue https://github.com/pytorch/pytorch/issues/128394, we need to switch to use an internal
        # export API and pre_dispatch=False. Switch to use the public API once the issue is included in 2.5 release.
        exported_program = torch.export._trace._export(
            TorchExportableModuleWithStaticCache(model),
            args=(example_input_ids,),
            kwargs={"cache_position": example_cache_position},
            pre_dispatch=False,
            strict=True,
        )
        return exported_program
