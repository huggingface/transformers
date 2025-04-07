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

from typing import Optional

import torch

from transformers.generation.configuration_utils import GenerationConfig

from ..utils.import_utils import is_torch_available


if is_torch_available():
    from transformers import PreTrainedModel, StaticCache
    from transformers.pytorch_utils import is_torch_greater_or_equal, is_torch_greater_or_equal_than_2_3


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
            max_batch_size=self.model.generation_config.cache_config.batch_size,
            max_cache_len=self.model.generation_config.cache_config.max_cache_len,
            device=self.model.generation_config.cache_config.device,
            dtype=self.model.dtype,
        )
        for i in range(len(self.static_cache.key_cache)):
            self.register_buffer(f"key_cache_{i}", self.static_cache.key_cache[i], persistent=False)
            self.register_buffer(f"value_cache_{i}", self.static_cache.value_cache[i], persistent=False)

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
        position_ids = cache_position.unsqueeze(0)
        past_key_values = self.static_cache

        outs = self.model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=past_key_values,
            use_cache=True,
        )
        return outs.logits

    @staticmethod
    def generate(
        exported_program: torch.export.ExportedProgram, prompt_token_ids: torch.Tensor, max_new_tokens: int
    ) -> torch.Tensor:
        """
        Generate a sequence of tokens using an exported program.

        This util function is designed to test exported models by simulating the generation process.
        It processes the input prompt tokens sequentially (no parallel prefill).
        This generate function is not intended to replace the original `generate` method, and the support
        for leveraging the original `generate` is potentially planed!

        Args:
            exported_program (`torch.export.ExportedProgram`): The exported program generated via `torch.export`.
            prompt_token_ids (`torch.Tensor`): Tensor representing the input prompt token IDs.
            max_new_tokens (`int`): Maximum number of new tokens to generate. Note that the total generation
                length is limited by both `max_new_tokens` and the model's cache size.

        Returns:
            torch.Tensor: A tensor containing the generated sequence of token IDs, including the original prompt tokens.
        """
        prompt_token_len = prompt_token_ids.shape[-1]
        max_generation_length = prompt_token_len + max_new_tokens
        for buffer_name, buffer in exported_program.named_buffers():
            if buffer_name.startswith("key_cache"):
                max_cache_len = buffer.shape[2]
                max_generation_length = min(max_generation_length, max_cache_len)
                break

        response_tokens = []
        for input_pos in range(min(max_generation_length, prompt_token_len)):
            result = exported_program.module().forward(
                input_ids=prompt_token_ids[:, input_pos : input_pos + 1],
                cache_position=torch.tensor([input_pos], dtype=torch.long),
            )
            response_tokens.append(prompt_token_ids[0][input_pos].item())

        current_token = torch.argmax(result[:, -1, :], dim=-1).item()
        response_tokens.append(current_token)

        while len(response_tokens) < max_generation_length:
            result = exported_program.module().forward(
                input_ids=torch.tensor([[current_token]], dtype=torch.long),
                cache_position=torch.tensor([len(response_tokens)], dtype=torch.long),
            )
            current_token = torch.argmax(result[:, -1, :], dim=-1).item()
            response_tokens.append(current_token)

        return torch.tensor([response_tokens], dtype=torch.long)


def convert_and_export_with_cache(
    model: PreTrainedModel,
    example_input_ids: Optional[torch.Tensor] = None,
    example_cache_position: Optional[torch.Tensor] = None,
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

        if is_torch_greater_or_equal("2.5.0"):
            exported_program = torch.export.export(
                TorchExportableModuleWithStaticCache(model),
                args=(example_input_ids,),
                kwargs={"cache_position": example_cache_position},
                strict=True,
            )
        else:
            # We have to keep this path for BC.
            #
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


class Seq2SeqLMEncoderExportableModule(torch.nn.Module):
    """
    A wrapper module designed to make a Seq2Seq LM encoder exportable with `torch.export`.
    This module ensures that the exported encoder model is compatible with ExecuTorch.
    """

    def __init__(self, encoder_model):
        super().__init__()
        self.encoder = encoder_model

    def forward(self, input_ids):
        return self.encoder(input_ids=input_ids).last_hidden_state


class Seq2SeqLMDecoderExportableModuleWithStaticCache(torch.nn.Module):
    """
    A wrapper module designed to make a Seq2Seq LM decoder exportable with `torch.export`,
    specifically for use with static caching. This module ensures the exported decoder
    is compatible with ExecuTorch.
    """

    def __init__(self, model, max_static_cache_length, batch_size):
        super().__init__()

        # Get the decoder component
        self.decoder = model.get_decoder()
        self.lm_head = model.lm_head
        self.config = model.config

        # Initialize static cache
        self.static_cache = StaticCache(
            config=self.config,
            max_batch_size=batch_size,
            max_cache_len=max_static_cache_length,
            device="cpu",
            dtype=torch.float32,
        )

        # Register cache buffers to make them exportable
        for i in range(len(self.static_cache.key_cache)):
            self.register_buffer(f"key_cache_{i}", self.static_cache.key_cache[i], persistent=False)
            self.register_buffer(f"value_cache_{i}", self.static_cache.value_cache[i], persistent=False)

    def forward(self, decoder_input_ids, encoder_hidden_states, cache_position):
        # Get outputs from decoder
        outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=self.static_cache,
            use_cache=True,
            cache_position=cache_position,
        )

        # Apply language model head
        lm_logits = self.lm_head(outputs[0])

        return lm_logits


class Seq2SeqLMExportableModule(torch.nn.Module):
    def __init__(
        self, model, batch_size=1, max_hidden_seq_length=4096, cache_implementation="static", max_cache_length=1024
    ):
        super().__init__()

        self.full_model = model
        self.encoder = model.get_encoder()
        self.config = model.config
        self.max_hidden_seq_length = max_hidden_seq_length
        self.generation_config = GenerationConfig(
            use_cache=True,
            max_length=max_cache_length,
            cache_implementation=cache_implementation,
            cache_config={
                "batch_size": batch_size,
                "max_cache_len": max_cache_length,
            },
        )
        self.exported_encoder = None
        self.exported_decoder = None

    def _export_encoder(self, encoder_input_ids):
        wrapped_encoder = Seq2SeqLMEncoderExportableModule(self.encoder).to("cpu").eval()

        # Define dynamic sequence length for encoder
        seq_len_dim = torch.export.Dim("encoder_seq_length", max=self.max_hidden_seq_length)

        # Export the encoder
        with torch.no_grad():
            exported_encoder = torch.export.export(
                wrapped_encoder, (encoder_input_ids,), dynamic_shapes={"input_ids": {1: seq_len_dim}}, strict=True
            )

        return exported_encoder

    def _export_decoder(self, decoder_input_ids, encoder_hidden_states, cache_position):
        wrapped_decoder = (
            Seq2SeqLMDecoderExportableModuleWithStaticCache(
                model=self.full_model,
                max_static_cache_length=self.generation_config.cache_config.max_cache_len,
                batch_size=self.generation_config.cache_config.batch_size,
            )
            .to("cpu")
            .eval()
        )

        # Define dynamic dimension for encoder output sequence length
        encoder_seq_len_dim = torch.export.Dim("encoder_hidden_seq_length", max=self.max_hidden_seq_length)

        # Export the decoder
        with torch.no_grad():
            exported_decoder = torch.export.export(
                wrapped_decoder,
                (decoder_input_ids, encoder_hidden_states, cache_position),
                dynamic_shapes={
                    "decoder_input_ids": None,
                    "encoder_hidden_states": {1: encoder_seq_len_dim},
                    "cache_position": None,
                },
                strict=True,
            )

        return exported_decoder

    def export(self, encoder_input_ids=None, decoder_input_ids=None, encoder_hidden_states=None, cache_position=None):
        example_encoder_input_ids = (
            encoder_input_ids if encoder_input_ids is not None else torch.ones((1, 10), dtype=torch.long)
        )
        example_decoder_input_ids = (
            decoder_input_ids if decoder_input_ids is not None else torch.tensor([[0]], dtype=torch.long)
        )  # Start token
        example_cache_position = cache_position if cache_position is not None else torch.tensor([0], dtype=torch.long)
        example_encoder_hidden_states = (
            encoder_hidden_states
            if encoder_hidden_states is not None
            else torch.zeros(
                (self.generation_config.cache_config.batch_size, 10, self.config.d_model), dtype=torch.float32
            )
        )
        self.exported_encoder = self._export_encoder(example_encoder_input_ids)
        self.exported_decoder = self._export_decoder(
            example_decoder_input_ids, example_encoder_hidden_states, example_cache_position
        )

        # Return self to allow chaining
        return self

    def generate(self, prompt_token_ids, max_new_tokens):
        with torch.no_grad():
            # Run encoder
            encoder_output = self.exported_encoder.module()(prompt_token_ids)

            # Initialize with start token (0 for T5)
            decoder_input_ids = torch.tensor([[0]], dtype=torch.long)
            generated_ids = [0]

            # Generate tokens one by one
            for i in range(max_new_tokens - 1):
                # Run decoder for next token prediction
                logits = self.exported_decoder.module()(
                    decoder_input_ids, encoder_output, torch.tensor([i], dtype=torch.long)
                )

                # Get next token
                next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
                generated_ids.append(next_token)

                # Update input for next iteration
                decoder_input_ids = torch.tensor([[next_token]], dtype=torch.long)

                # Check if EOS token
                if next_token == self.config.eos_token_id:
                    break

            return generated_ids
