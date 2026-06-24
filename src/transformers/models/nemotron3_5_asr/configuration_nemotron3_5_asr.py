# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..nemotron_asr_streaming.configuration_nemotron_asr_streaming import NemotronAsrStreamingEncoderConfig


@auto_docstring(checkpoint="nvidia/nemotron-3.5-asr-streaming-0.6b")
@strict
class Nemotron3_5AsrConfig(PreTrainedConfig):
    r"""
    vocab_size (`int`, *optional*, defaults to 13088):
        Vocabulary size of the joint network output (including the blank token).
    decoder_hidden_size (`int`, *optional*, defaults to 640):
        Hidden size of the LSTM prediction network (NeMo's `pred_hidden`).
    num_decoder_layers (`int`, *optional*, defaults to 2):
        Number of LSTM layers in the prediction network.
    hidden_act (`str`, *optional*, defaults to `"relu"`):
        Activation in the joint network.
    max_symbols_per_step (`int`, *optional*, defaults to 10):
        Maximum number of non-blank symbols emitted per encoder time step during greedy decoding.
    encoder_config (`Union[dict, NemotronAsrStreamingEncoderConfig]`, *optional*):
        The config object or dictionary of the encoder. Reuses [`NemotronAsrStreamingEncoderConfig`] directly,
        since the encoder is identical to [`NemotronAsrStreaming`]'s.
    blank_token_id (`int`, *optional*, defaults to 13087):
        Blank token id for RNN-T decoding.
    joint_hidden_size (`int`, *optional*, defaults to 640):
        Hidden size of the joint network's encoder/decoder projections (NeMo's `joint_hidden`).
    durations (`list[int]`, *optional*, defaults to `()`):
        Pinned to the empty tuple for RNN-T: no token durations are predicted, so the joint head outputs
        only `vocab_size` logits.
    num_prompts (`int`, *optional*, defaults to 128):
        Number of language-prompt slots. The target language is encoded as a one-hot vector of this
        size, broadcast across the encoder time axis and concatenated with the encoder output before
        the `prompt_kernel` fusion MLP.
    prompt_intermediate_size (`int`, *optional*, defaults to 2048):
        Hidden size of the `prompt_kernel` fusion MLP (`Linear(hidden + num_prompts -> intermediate)
        -> ReLU -> Linear(intermediate -> hidden)`).

    Example:
    ```python
    >>> from transformers import Nemotron3_5AsrForRNNT, Nemotron3_5AsrConfig

    >>> configuration = Nemotron3_5AsrConfig()
    >>> model = Nemotron3_5AsrForRNNT(configuration)
    >>> configuration = model.config
    ```
    """

    model_type = "nemotron3_5_asr"
    # The encoder is identical to NemotronAsrStreaming's, so reuse its config class directly.
    sub_configs = {"encoder_config": NemotronAsrStreamingEncoderConfig}

    vocab_size: int = 13088
    decoder_hidden_size: int = 640
    num_decoder_layers: int = 2
    hidden_act: str = "relu"
    max_symbols_per_step: int = 10
    encoder_config: dict | PreTrainedConfig | None = None
    pad_token_id: int = 0
    blank_token_id: int = 13087
    is_encoder_decoder: bool = True
    joint_hidden_size: int = 640
    durations: list[int] | tuple[int, ...] = ()
    num_prompts: int = 128
    prompt_intermediate_size: int = 2048

    def __post_init__(self, **kwargs):
        if self.decoder_hidden_size != self.joint_hidden_size:
            raise ValueError(
                "Nemotron3_5AsrConfig currently requires decoder_hidden_size == joint_hidden_size "
                f"(got {self.decoder_hidden_size} and {self.joint_hidden_size})."
            )
        # The decoder starts on the blank token at frame 0 (NeMo's blank_as_pad convention).
        kwargs.setdefault("decoder_start_token_id", self.blank_token_id)
        if isinstance(self.encoder_config, dict):
            self.encoder_config = NemotronAsrStreamingEncoderConfig(**self.encoder_config)
        elif self.encoder_config is None:
            self.encoder_config = NemotronAsrStreamingEncoderConfig()
        self.initializer_range = self.encoder_config.initializer_range
        super().__post_init__(**kwargs)


__all__ = ["Nemotron3_5AsrConfig"]
