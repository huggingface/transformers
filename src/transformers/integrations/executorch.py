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

import logging
from typing import Callable, Optional

import torch

from ..cache_utils import DynamicCache, EncoderDecoderCache, HybridCache, StaticCache
from ..generation.configuration_utils import GenerationConfig
from ..masking_utils import (
    ALL_MASK_ATTENTION_FUNCTIONS,
    _ignore_causal_mask_sdpa,
    _is_torch_greater_or_equal_than_2_5,
    prepare_padding_mask,
)
from ..modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ..pytorch_utils import is_torch_greater_or_equal, is_torch_greater_or_equal_than_2_3


class TorchExportableModuleForDecoderOnlyLM(torch.nn.Module):
    """
    A recipe module designed to make a `PreTrainedModel` exportable with `torch.export`,
    specifically for decoder-only LM with cache. This module ensures that the
    exported model is compatible with further lowering and execution in `ExecuTorch`.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        max_batch_size: int = 1,
        max_cache_len: int = 4096,
    ):
        """
        Initializes the exportable module with `HybridCache`.

        Args:
            model (`PreTrainedModel`): The pretrained model to wrap.
            max_batch_size (int): Maximum batch size for the cache.
            max_cache_len (int): Maximum sequence length for the cache.

        Raises:
            ValueError: If the model is configured with a unsupported cache implementation.
        """
        super().__init__()

        if not hasattr(model.config, "use_cache") or model.config.use_cache is False:
            raise ValueError("The model must have caching enabled to be performant.")

        if hasattr(model.config, "layer_types") and getattr(model.config, "sliding_window", None) is not None:
            self.model = TorchExportableModuleWithHybridCache(model, max_batch_size, max_cache_len)
        else:
            # If `layer_types` is not specified explicitly in the config or `sliding_window` is null,
            # there is only 1 type of layers, so export will use `StaticCache` by default.
            logging.info(
                "Using `StaticCache` for export as `layer_types` is not specified or `sliding_window` is `null` in the config."
            )
            self.model = TorchExportableModuleWithStaticCache(model)
        # This is the same as sdpa, but mask creation does not use `vmap` which is not exportable
        ALL_MASK_ATTENTION_FUNCTIONS.register("sdpa_without_vmap", sdpa_mask_without_vmap)
        ALL_ATTENTION_FUNCTIONS.register("sdpa_without_vmap", ALL_ATTENTION_FUNCTIONS["sdpa"])
        self.model.model.config._attn_implementation = "sdpa_without_vmap"

    def forward(
        self,
        input_ids: torch.Tensor,
        cache_position: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the module, which is compatible with the ExecuTorch llm runner.

        Args:
            input_ids (`torch.Tensor`): Tensor representing current input token id to the module.
            cache_position (`torch.Tensor`): Tensor representing current input position in the cache.

        Returns:
            torch.Tensor: Logits output from the model.
        """
        return self.model.forward(input_ids, cache_position)

    def export(
        self,
        input_ids: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        dynamic_shapes: Optional[dict] = None,
        strict: Optional[bool] = None,
    ) -> torch.export.ExportedProgram:
        """
        Export the wrapped module using `torch.export`.

        Args:
            input_ids (`Optional[torch.Tensor]`):
                Tensor representing current input token id to the module. If not provided, a default tensor will be used.
            cache_position (`Optional[torch.Tensor]`):
                Tensor representing current input position in the cache. If not provided, a default tensor will be used.
            dynamic_shapes (`Optional[dict]`):
                Dynamic shapes to use for export if specified.
            strict(`Optional[bool]`):
                Flag to instruct `torch.export` to use `torchdynamo`.
        """
        if hasattr(self.model, "base_model_prefix"):
            base = getattr(self.model, self.model.base_model_prefix, self.model)
            model_device = base.device
        elif hasattr(self.model, "model"):
            model_device = self.model.model.device
        else:
            model_device = "cpu"
            logging.warning(
                "TorchExportableModuleForDecoderOnlyLM.export Can't infer device from the model. Set to CPU by default."
            )

        example_input_ids = (
            input_ids if input_ids is not None else torch.tensor([[1]], dtype=torch.long, device=model_device)
        )
        example_cache_position = (
            cache_position if cache_position is not None else torch.tensor([0], dtype=torch.long, device=model_device)
        )

        exported_program = torch.export.export(
            self.model,
            args=(example_input_ids, example_cache_position),
            kwargs={},
            dynamic_shapes=dynamic_shapes,
            strict=strict if strict is not None else True,
        )
        return exported_program

    @staticmethod
    def generate(
        exported_program: torch.export.ExportedProgram,
        tokenizer,
        prompt: str,
        max_new_tokens: int = 20,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        device: str = "cpu",
    ) -> str:
        """
        Generate a sequence of tokens using an exported program.

        Args:
            exported_program (`torch.export.ExportedProgram`): The exported model being used for generate.
            tokenizer: The tokenizer to use.
            prompt (str): The input prompt.
            max_new_tokens (int): Maximum number of new tokens to generate.
            do_sample (bool): Whether to use sampling or greedy decoding.
            temperature (float): The temperature for sampling.
            top_k (int): The number of highest probability tokens to keep for top-k sampling.
            top_p (float): The cumulative probability for nucleus sampling.
            device (str): The device to use.

        Returns:
            str: The generated text.
        """
        # Get the module from the exported program
        exported_module = exported_program.module()

        # Tokenize the prompt
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        # Initialize with the prompt
        generated_ids = input_ids.clone()

        # Process the prompt tokens first
        curr_position = 0
        for i in range(input_ids.shape[1]):
            # Process one token at a time
            curr_input_ids = input_ids[:, i : i + 1]
            curr_cache_position = torch.tensor([curr_position], dtype=torch.long, device=device)

            # Forward pass
            _ = exported_module(curr_input_ids, curr_cache_position)
            curr_position += 1

        # Generate new tokens
        for _ in range(max_new_tokens):
            # Get the last token as input
            curr_input_ids = generated_ids[:, -1:]
            curr_cache_position = torch.tensor([curr_position], dtype=torch.long, device=device)

            # Forward pass to get next token logits
            outputs = exported_module(curr_input_ids, curr_cache_position)

            # Get the next token ID
            if do_sample:
                # Apply temperature
                if temperature > 0:
                    logits = outputs / temperature
                else:
                    logits = outputs

                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float("-inf")

                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    # Scatter sorted tensors to original indexing
                    indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float("-inf")

                # Sample from the filtered distribution
                probs = torch.softmax(logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_token_id = outputs.argmax(dim=-1, keepdim=True)

            # Ensure next_token_id has the right shape before concatenation
            if next_token_id.dim() > 2:
                next_token_id = next_token_id.squeeze(-1)

            # Append to the generated sequence
            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
            curr_position += 1

            # Stop if we generate an EOS token
            if next_token_id.item() == tokenizer.eos_token_id:
                break

        # Decode the generated text
        return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


class TorchExportableModuleWithStaticCache(torch.nn.Module):
    """
    A recipe module designed to make a `PreTrainedModel` exportable with `torch.export`,
    specifically for decoder-only LM to `StaticCache`. This module ensures that the
    exported model is compatible with further lowering and execution in `ExecuTorch`.

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
            max_batch_size=self.model.generation_config.cache_config.get("batch_size"),
            max_cache_len=self.model.generation_config.cache_config.get("max_cache_len"),
            device=self.model.generation_config.cache_config.get("device"),
            dtype=self.model.dtype,
        )
        for i in range(len(self.static_cache)):
            self.register_buffer(f"key_cache_{i}", self.static_cache.layers[i].keys, persistent=False)
            self.register_buffer(f"value_cache_{i}", self.static_cache.layers[i].values, persistent=False)

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
        position_ids = cache_position.unsqueeze(0)
        past_key_values = self.static_cache

        outs = self.model(
            input_ids=input_ids,
            attention_mask=None,
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=past_key_values,
            use_cache=True,
        )
        return outs.logits

    @staticmethod
    def generate(
        exported_program: torch.export.ExportedProgram,
        prompt_token_ids: torch.Tensor,
        max_new_tokens: int,
    ) -> torch.Tensor:
        """
        Generate a sequence of tokens using an exported program.

        This util function is designed to test exported models by simulating the generation process.
        It processes the input prompt tokens sequentially (no parallel prefill).
        This generate function is not intended to replace the original `generate` method, and the support
        for leveraging the original `generate` is potentially planned!

        Args:
            exported_program (`torch.export.ExportedProgram`): The exported program generated via `torch.export`.
            prompt_token_ids (`torch.Tensor`): Tensor representing the input prompt token IDs.
            max_new_tokens (`int`): Maximum number of new tokens to generate. Note that the total generation
                length is limited by both `max_new_tokens` and the model's cache size.

        Returns:
            torch.Tensor: A tensor containing the generated sequence of token IDs, including the original prompt tokens.
        """
        device = prompt_token_ids.device
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
                cache_position=torch.tensor([input_pos], dtype=torch.long, device=device),
            )
            response_tokens.append(prompt_token_ids[0][input_pos].item())

        current_token = torch.argmax(result[:, -1, :], dim=-1).item()
        response_tokens.append(current_token)

        while len(response_tokens) < max_generation_length:
            result = exported_program.module().forward(
                input_ids=torch.tensor([[current_token]], dtype=torch.long, device=device),
                cache_position=torch.tensor([len(response_tokens)], dtype=torch.long, device=device),
            )
            current_token = torch.argmax(result[:, -1, :], dim=-1).item()
            response_tokens.append(current_token)

        return torch.tensor([response_tokens], dtype=torch.long, device=device)


class TorchExportableModuleWithHybridCache(torch.nn.Module):
    """
    A recipe module designed to make a `PreTrainedModel` exportable with `torch.export`,
    specifically for decoder-only LM to `HybridCache`. This module ensures that the
    exported model is compatible with further lowering and execution in `ExecuTorch`.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        max_batch_size: int = 1,
        max_cache_len: int = 4096,
    ):
        """
        Initializes the exportable module with `HybridCache`.

        Args:
            model (`PreTrainedModel`): The pretrained model to wrap.
            max_batch_size (int): Maximum batch size for the cache.
            max_cache_len (int): Maximum sequence length for the cache.

        Raises:
            AssertionError: If the model doesn't have the expected configuration for HybridCache.
        """
        super().__init__()
        self.model = model

        # Verify the model is configured for HybridCache
        if not self.model.config.use_cache:
            raise AssertionError("Model must have caching enabled")

        # Initialize the HybridCache
        self.cache = HybridCache(
            config=self.model.config,
            max_batch_size=max_batch_size,
            max_cache_len=max_cache_len,
            device=self.model.device,
            dtype=self.model.dtype,
        )

        # Register all key and value cache tensors as buffers
        for i in range(len(self.cache)):
            self.register_buffer(f"key_cache_{i}", self.cache.layers[i].keys, persistent=False)
            self.register_buffer(f"value_cache_{i}", self.cache.layers[i].values, persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        cache_position: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the module, which is compatible with the ExecuTorch llm runner.

        Args:
            input_ids (`torch.Tensor`): Tensor representing current input token id to the module.
            cache_position (`torch.Tensor`): Tensor representing current input position in the cache.

        Returns:
            torch.Tensor: Logits output from the model.
        """
        batch_size = input_ids.shape[0]

        # Generate position_ids from cache_position
        position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)

        # Forward pass with the model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=self.cache,
            use_cache=True,
            cache_position=cache_position,
        )

        # Return only the logits to simplify the export
        return outputs.logits


def convert_and_export_with_cache(
    model: PreTrainedModel,
    example_input_ids: Optional[torch.Tensor] = None,
    example_cache_position: Optional[torch.Tensor] = None,
    dynamic_shapes: Optional[dict] = None,
    strict: Optional[bool] = None,
):
    """
    Convert a `PreTrainedModel` into an exportable module and export it using `torch.export`,
    ensuring the exported model is compatible with `ExecuTorch`.

    Args:
        model (`PreTrainedModel`): The pretrained model to be exported.
        example_input_ids (`Optional[torch.Tensor]`): Example input token id used by `torch.export`.
        example_cache_position (`Optional[torch.Tensor]`): Example current cache position used by `torch.export`.
        dynamic_shapes(`Optional[dict]`): Dynamic shapes used by `torch.export`.
        strict(`Optional[bool]`): Flag to instruct `torch.export` to use `torchdynamo`.

    Returns:
        Exported program (`torch.export.ExportedProgram`): The exported program generated via `torch.export`.
    """
    if not is_torch_greater_or_equal_than_2_3:
        raise ImportError("torch >= 2.3 is required.")

    import torch.export._trace

    # This is the same as sdpa, but mask creation does not use `vmap` which is not exportable
    ALL_MASK_ATTENTION_FUNCTIONS.register("sdpa_without_vmap", sdpa_mask_without_vmap)
    ALL_ATTENTION_FUNCTIONS.register("sdpa_without_vmap", ALL_ATTENTION_FUNCTIONS["sdpa"])
    model.config._attn_implementation = "sdpa_without_vmap"

    with torch.no_grad():
        # TODO: The default inputs only work for text models. We need to add support for vision/audio models.
        example_input_ids = (
            example_input_ids
            if example_input_ids is not None
            else torch.tensor([[1]], dtype=torch.long, device=model.device)
        )
        example_cache_position = (
            example_cache_position
            if example_cache_position is not None
            else torch.tensor([0], dtype=torch.long, device=model.device)
        )

        if is_torch_greater_or_equal("2.6.0"):
            exported_program = torch.export.export(
                TorchExportableModuleWithStaticCache(model),
                args=(example_input_ids, example_cache_position),
                kwargs={},
                dynamic_shapes=dynamic_shapes,
                strict=strict if strict is not None else True,
            )
        else:
            if dynamic_shapes is not None:
                logging.warning(
                    "Dynamic shapes spec will be ignored by convert_and_export_with_cache for torch < 2.6.0."
                )
            if strict is not None:
                logging.warning("The strict flag will be ignored by convert_and_export_with_cache for torch < 2.6.0.")
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

        # Initialize static cache for decoder and DynamicCache for encoder
        self.static_cache = StaticCache(
            config=self.config,
            max_batch_size=batch_size,
            max_cache_len=max_static_cache_length,
            device="cpu",
            dtype=torch.float32,
        )
        self.cache = EncoderDecoderCache(self.static_cache, DynamicCache())

        # Register cache buffers to make them exportable
        for i in range(len(self.static_cache)):
            self.register_buffer(f"key_cache_{i}", self.static_cache.layers[i].keys, persistent=False)
            self.register_buffer(f"value_cache_{i}", self.static_cache.layers[i].values, persistent=False)

    def forward(self, decoder_input_ids, encoder_hidden_states, cache_position):
        # Get outputs from decoder
        outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=self.cache,
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
        wrapped_encoder = Seq2SeqLMEncoderExportableModule(self.encoder).to(self.full_model.device).eval()

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
        device = self.full_model.device
        example_encoder_input_ids = (
            encoder_input_ids
            if encoder_input_ids is not None
            else torch.ones((1, 10), dtype=torch.long, device=device)
        )
        example_decoder_input_ids = (
            decoder_input_ids
            if decoder_input_ids is not None
            else torch.tensor([[0]], dtype=torch.long, device=device)
        )  # Start token
        example_cache_position = (
            cache_position if cache_position is not None else torch.tensor([0], dtype=torch.long, device=device)
        )
        example_encoder_hidden_states = (
            encoder_hidden_states
            if encoder_hidden_states is not None
            else torch.zeros(
                (self.generation_config.cache_config.batch_size, 10, self.config.d_model),
                dtype=torch.float32,
                device=device,
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


def export_with_dynamic_cache(
    model: PreTrainedModel,
    example_input_ids: Optional[torch.Tensor] = None,
    example_attention_mask: Optional[torch.Tensor] = None,
):
    """
    Export a model with DynamicCache using `torch.export`, ensuring the exported model is compatible with `ExecuTorch`.

    Args:
        model (`PreTrainedModel`): The pretrained model to be exported.
        example_input_ids (`Optional[torch.Tensor]`): Example input token id used by `torch.export`.
        example_attention_mask (`Optional[torch.Tensor]`): Example attention mask used by `torch.export`.

    Returns:
        Exported program (`torch.export.ExportedProgram`): The exported program generated via `torch.export`.
    """
    if not is_torch_greater_or_equal_than_2_3:
        raise ImportError("torch >= 2.3 is required.")

    # This is the same as sdpa, but mask creation does not use `vmap` which is not exportable
    ALL_MASK_ATTENTION_FUNCTIONS.register("sdpa_without_vmap", sdpa_mask_without_vmap)
    ALL_ATTENTION_FUNCTIONS.register("sdpa_without_vmap", ALL_ATTENTION_FUNCTIONS["sdpa"])
    model.config._attn_implementation = "sdpa_without_vmap"

    with torch.no_grad():
        exported_program = torch.export.export(
            model,
            (),
            {
                "input_ids": example_input_ids,
                "attention_mask": example_attention_mask,
                "past_key_values": DynamicCache(),
                "use_cache": True,
            },
            strict=False,
        )
        return exported_program


def sdpa_mask_without_vmap(
    batch_size: int,
    cache_position: torch.Tensor,
    kv_length: int,
    kv_offset: int = 0,
    mask_function: Optional[Callable] = None,
    attention_mask: Optional[torch.Tensor] = None,
    local_size: Optional[int] = None,
    allow_is_causal_skip: bool = True,
    allow_torch_fix: bool = True,
    **kwargs,
) -> Optional[torch.Tensor]:
    """
    Create a 4D boolean mask of shape `(batch_size, 1, query_length, kv_length)` where a value of True indicates that
    the element should take part in the attention computation, and False that it should not.

    This is similar to `masking_utils.sdpa_mask` but does not use `vmap` which is incompatible with export.

    Args:
        batch_size (`int`):
            The batch size of the input sequence.
        cache_position (`torch.Tensor`):
            A tensor of shape (query_length,) indicating the current indices of the input sequence elements.
        kv_length (`int`):
            The size that the key and value states will have during the attention computation.
        kv_offset (`int`, optional):
            An optional offset to indicate at which first position the key and values states will refer to.
        mask_function (`Callable`):
            The mask factory function describing the mask pattern.
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length)
        local_size (`int`, optional):
            The size of the local attention, if we do not use full attention. This is used only if `allow_is_causal_skip=True`
            to try to skip mask creation if possible.
        allow_is_causal_skip (`bool`, optional):
            Whether to allow to return `None` for the mask under conditions where we can use the `is_causal` argument in
            `torch.sdpa` instead. Default to `True`.
        allow_torch_fix (`bool`, optional):
            Whether to update the mask in case a query is not attending to any tokens, to solve a bug in torch's older
            versions. We need an arg to skip it when using eager. By default `True`.

    """

    q_length = cache_position.shape[0]
    # Potentially pad the 2D mask, and slice it correctly
    padding_mask = prepare_padding_mask(attention_mask, kv_length, kv_offset)

    #  Under specific conditions, we can avoid materializing the mask, instead relying on the `is_causal` argument
    if allow_is_causal_skip and _ignore_causal_mask_sdpa(padding_mask, q_length, kv_length, local_size):
        return None

    # Similar to `kv_arange = torch.arange(start=kv_offset, end=kv_offset + kv_length, device=cache_position.device)`
    # but without data-dependent slicing (i.e. torch.compile friendly)
    kv_arange = torch.arange(kv_length, device=cache_position.device)
    kv_arange += kv_offset
    reshaped_cache_position = cache_position.view(-1, 1)

    # This is a bit hacky to know what pattern we are using, but all mask creation function actually forward
    # the config through kwargs anyway, so it allows to rely on it
    # Usually, the `mask_function` is the only entry-point to define the pattern - we could do for loops over it,
    # but this is more efficient
    sliding_window = getattr(kwargs["config"], "sliding_window", None)
    chunk_size = getattr(kwargs["config"], "attention_chunk_size", None)

    if sliding_window is not None and chunk_size is not None:
        raise ValueError("Cannot use both `sliding_window` and `attention_chunk_size`")

    # Simplest and most efficient way to obtain a causal mask
    causal_mask = kv_arange <= reshaped_cache_position
    # If using sliding window, add the sliding mask
    if sliding_window is not None:
        sliding_mask_overlay = kv_arange > reshaped_cache_position - sliding_window
        causal_mask *= sliding_mask_overlay
    # If using chunk attention, add the chunked mask
    elif chunk_size is not None:
        chunked_mask_overlay = kv_arange // chunk_size == reshaped_cache_position // chunk_size
        causal_mask *= chunked_mask_overlay

    causal_mask = causal_mask[None, None, :, :].expand(batch_size, -1, -1, -1)
    if padding_mask is not None:
        causal_mask = causal_mask * padding_mask[:, None, None, :]

    # Due to a bug in some older torch version, we need to update the mask in case a query is not attending to any
    # tokens (due to padding). See details in https://github.com/pytorch/pytorch/issues/110213
    if not _is_torch_greater_or_equal_than_2_5 and allow_torch_fix:
        causal_mask |= torch.all(~causal_mask, dim=-1, keepdim=True)
    return causal_mask
