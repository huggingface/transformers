# Copyright 2026 OpenMOSS and the HuggingFace Inc. team. All rights reserved.
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
""" MossTTSDelay model configuration """


from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging
from ..qwen3 import Qwen3Config


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="OpenMOSS-Team/MOSS-TTS-v1.5")
@strict
class MossTTSDelayConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MossTTSDelayModel`]. It is used to instantiate an
    MossTTSDelay model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the MossTTSDelay [MossTTSDelay-8B](https://huggingface.co/OpenMOSS/mosstts-8b) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        language_config (`Union[Qwen3Config, dict]`, *optional*):
            Configuration for the backbone language model (Qwen3).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        n_vq (`int`, *optional*, defaults to 32):
            Number of additional VQ (Vector Quantization) heads/channels for audio.
            Determines the number of codebooks used in the audio representation.
        pad_token_id (`int`, *optional*, defaults to 151643):
            Padding token id for the text channel.
        im_start_token_id (`int`, *optional*, defaults to 151644):
            Token id used to mark the beginning of a chat message.
        im_end_token_id (`int`, *optional*, defaults to 151645):
            Token id used to mark the end of a chat message.
        audio_vocab_size (`int`, *optional*, defaults to 1024):
            Vocabulary size for the audio tokens (codebooks 1 to N).
        audio_user_slot_token_id (`int`, *optional*, defaults to 151654):
            The specific token ID used as a placeholder/slot for user-side audio inputs in the prompt.
        audio_assistant_gen_slot_token_id (`int`, *optional*, defaults to 151656):
            The specific token ID representing the generation slot for the assistant's audio output.
            Acting as the trigger for the TTS generation process.
        audio_assistant_delay_slot_token_id (`int`, *optional*, defaults to 151662):
            The token ID used in the 'Delay Pattern' paradigm to represent the delayed/offset positions
            between different VQ channels.
        audio_start_token_id (`int`, *optional*, defaults to 151652):
            Special token ID used to denote the start of an audio sequence in the stream.
        audio_end_token_id (`int`, *optional*, defaults to 151653):
            Special token ID used to denote the end of an audio sequence (EOS for audio).
        audio_pad_code (`int`, *optional*, defaults to 1024):
            The padding value used within the audio VQ codebooks. Typically equals `audio_vocab_size`.
        sampling_rate (`int`, *optional*, defaults to 24000):
            Audio sampling rate used by the processor and audio tokenizer.
    """
    model_type = "moss_tts_delay"
    keys_to_ignore_at_inference = ["past_key_values"]
    sub_configs = {"language_config": Qwen3Config}

    def __init__(
        self,
        language_config: Qwen3Config | dict | None = None,
        initializer_range: float = 0.02,
        n_vq: int = 32,
        pad_token_id: int = 151643,
        im_start_token_id: int = 151644,
        im_end_token_id: int = 151645,
        audio_vocab_size: int = 1024,
        audio_user_slot_token_id: int = 151654,
        audio_assistant_gen_slot_token_id: int = 151656,
        audio_assistant_delay_slot_token_id: int = 151662,
        audio_start_token_id: int = 151652,
        audio_end_token_id: int = 151653,
        audio_pad_code: int = 1024,
        sampling_rate: int = 24000,
        **kwargs,
    ):
        if isinstance(language_config, dict):
            self.language_config = Qwen3Config(**language_config)
        elif language_config is None:
            self.language_config = Qwen3Config()
        else:
            self.language_config = language_config

        self.initializer_range = initializer_range
        self.n_vq = n_vq
        self.audio_vocab_size = audio_vocab_size
        self.audio_user_slot_token_id = audio_user_slot_token_id
        self.audio_assistant_gen_slot_token_id = audio_assistant_gen_slot_token_id
        self.audio_assistant_delay_slot_token_id = audio_assistant_delay_slot_token_id
        self.audio_start_token_id = audio_start_token_id
        self.audio_end_token_id = audio_end_token_id
        self.audio_pad_code = audio_pad_code
        self.sampling_rate = sampling_rate

        self.hidden_size = self.language_config.hidden_size
        self.vocab_size = self.language_config.vocab_size
        self.pad_token_id = pad_token_id
        self.im_start_token_id = im_start_token_id
        self.im_end_token_id = im_end_token_id

        super().__init__(**kwargs)

    def to_dict(self):
        output = super().to_dict()
        if hasattr(self.language_config, "to_dict"):
            output["language_config"] = self.language_config.to_dict()
        else:
            output["language_config"] = self.language_config
        return output


__all__ = ["MossTTSDelayConfig"]
