# Copyright 2025 The OpenBMB Team. All rights reserved.
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

import math
import os
import warnings
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from threading import Thread
from typing import Optional, Union, Callable

import numpy as np
from PIL import Image
import soundfile as sf
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as P
from torch.nn.init import _calculate_fan_in_and_fan_out
from torch.nn.utils.parametrizations import weight_norm
from huggingface_hub import hf_hub_download

from ...modeling_utils import PreTrainedModel
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPast,
    BaseModelOutputWithPooling,
    CausalLMOutputWithPast,
)
from ...utils import (
    logging,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
    can_return_tuple,
    auto_docstring,
    ModelOutput,
    TransformersKwargs,
)
from ...utils.import_utils import _is_package_available, is_flash_attn_2_available
from ...cache_utils import Cache, DynamicCache, EncoderDecoderCache, StaticCache
from ...configuration_utils import PretrainedConfig
from ...generation import GenerationMixin
from ...generation.streamers import TextIteratorStreamer
from ...generation.utils import GenerateOutput
from ...generation.logits_process import LogitsProcessor, TopKLogitsWarper, TopPLogitsWarper
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, AttentionMaskConverter
from ...integrations import is_deepspeed_zero3_enabled
from ...modeling_flash_attention_utils import FlashAttentionKwargs, _get_unpad_data
from ...processing_utils import Unpack

from ..bert.tokenization_bert_fast import BertTokenizerFast
from ..siglip.configuration_siglip import SiglipVisionConfig
from ..siglip.modeling_siglip import SiglipEncoderLayer, SiglipEncoder, SiglipMLP, SiglipVisionModelOutput
from ..whisper.configuration_whisper import WhisperConfig
from ..whisper.modeling_whisper import WhisperEncoder, WhisperAttention, WhisperEncoderLayer
from ..qwen2.configuration_qwen2 import Qwen2Config
from ..qwen2.modeling_qwen2 import Qwen2Model, Qwen2PreTrainedModel
from ..llama.configuration_llama import LlamaConfig
from ..llama.modeling_llama import LlamaModel, LlamaDecoderLayer, LlamaPreTrainedModel

from .tts_processing_minicpm_o_2_6 import NumberToTextConverter, sentence_end, VoiceChecker, ChatTTSProcessor

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input

if _is_package_available('vector_quantize_pytorch') and _is_package_available('vocos'):
    from vector_quantize_pytorch import GroupedResidualFSQ
    from vocos import Vocos
    from vocos.pretrained import instantiate_class

_tts_deps = _is_package_available('vector_quantize_pytorch') and _is_package_available('vocos')

logger = logging.get_logger(__name__)

class MiniCPMConditionalTTSTextConfig(LlamaConfig):
    pass


class MiniCPMConditionalTTSConfig(PretrainedConfig):
    model_type = "conditional_chattts"

    # sub_configs = {
    #     "text_config": MiniCPMConditionalTTSTextConfig,
    # }

    def __init__(
        self,
        llm_dim: int = 2560,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        num_attention_heads: int = 12,
        num_hidden_layers: int = 20,
        max_position_embeddings: int = 4096,
        num_audio_tokens: int = 626,
        num_text_tokens: int = 21178,
        num_mel_bins: int = 100,
        num_vq: int = 4,
        use_speaker_embedding: bool = True,
        use_llm_hidden_state: bool = False,
        spk_emb_token_id: int = 21143,
        num_spk_embs: int = 1,
        audio_bos_token_id: int = 21132,
        text_eos_token_id: int = 21133,
        use_text: bool = True,
        streaming: bool = True,
        streaming_text_chunk_size: int = 10,
        streaming_text_reserved_len: int = 300,
        streaming_audio_chunk_size: int = 50,
        attn_implementation: str = "sdpa",
        use_mlp: bool = True,
        aug_loss_weight: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.llm_dim = llm_dim
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.max_position_embeddings = max_position_embeddings
        self.num_audio_tokens = num_audio_tokens
        self.num_text_tokens = num_text_tokens
        self.num_mel_bins = num_mel_bins
        self.num_vq = num_vq
        self.use_speaker_embedding = use_speaker_embedding
        self.use_llm_hidden_state = use_llm_hidden_state
        self.spk_emb_token_id = spk_emb_token_id
        self.num_spk_embs = num_spk_embs
        self.audio_bos_token_id = audio_bos_token_id
        self.text_eos_token_id = text_eos_token_id
        self.use_text = use_text
        self.streaming = streaming
        self.streaming_text_chunk_size = streaming_text_chunk_size
        self.streaming_text_reserved_len = streaming_text_reserved_len
        self.streaming_audio_chunk_size = streaming_audio_chunk_size
        self.attn_implementation = attn_implementation
        self.use_mlp = use_mlp
        self.aug_loss_weight = aug_loss_weight

        self.tts_text_config = MiniCPMConditionalTTSTextConfig(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_attention_heads=self.num_attention_heads,
            num_hidden_layers=self.num_hidden_layers,
            max_position_embeddings=self.max_position_embeddings,
            attn_implementation=self.attn_implementation,
        )


class MiniCPM_o_2_6TextConfig(Qwen2Config):
    model_type = "minicpmo"

class MiniCPMVisionConfig(SiglipVisionConfig):
    pass

class MiniCPMWhisperConfig(WhisperConfig):
    pass 

class MiniCPM_o_2_6Config(PretrainedConfig):

    default_vision_config = {
        "hidden_size": 1152,
        "image_size": 980,
        "intermediate_size": 4304,
        "model_type": "siglip",
        "num_attention_heads": 16,
        "num_hidden_layers": 27,
        "patch_size": 14,
    }

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        audio_config=None,
        tts_config=None,
        use_cache=True,
        query_num=64,
        drop_vision_last_layer=True,
        vision_batch_size=16,
        audio_pool_step=2,
        audio_chunk_length=1.0,
        **kwargs,
    ):
        self.use_cache = use_cache
        self.query_num = query_num
        self.drop_vision_last_layer = drop_vision_last_layer
        self.vision_batch_size = vision_batch_size
        self.audio_pool_step = audio_pool_step
        self.audio_chunk_length = audio_chunk_length

        if text_config is None:
            self.text_config = MiniCPM_o_2_6TextConfig()
        elif isinstance(text_config, dict):
            self.text_config = MiniCPM_o_2_6TextConfig(**text_config)
        elif isinstance(text_config, MiniCPM_o_2_6TextConfig):
            self.text_config = text_config

        if vision_config is None:
            self.vision_config = MiniCPMVisionConfig(
                **self.default_vision_config)
            logger.info("vision_config is None, using default vision config")
        elif isinstance(vision_config, dict):
            self.vision_config = MiniCPMVisionConfig(**vision_config)
        elif isinstance(vision_config, MiniCPMVisionConfig):
            self.vision_config = vision_config

        # same as openai/whisper-medium add use_cache
        if audio_config is None:
            self.audio_config = MiniCPMWhisperConfig()
        elif isinstance(audio_config, dict):
            self.audio_config = MiniCPMWhisperConfig(**audio_config)
        elif isinstance(audio_config, MiniCPMWhisperConfig):
            self.audio_config = audio_config

        if tts_config is None:
            self.tts_config = MiniCPMConditionalTTSConfig()
        elif isinstance(tts_config, dict):
            self.tts_config = MiniCPMConditionalTTSConfig(**tts_config)
        elif isinstance(tts_config, MiniCPMConditionalTTSConfig):
            self.tts_config = tts_config

        # self.patch_size = self.vision_config.patch_size
        super().__init__(**kwargs)


@dataclass
class OmniOutput(ModelOutput):
    text: Optional[Union[str, list[str], Iterator]] = None
    outputs: Optional[Union[GenerateOutput, torch.LongTensor]] = None
    spk_embeds: Optional[torch.FloatTensor] = None
    audio_wav: Optional[np.ndarray] = None
    sampling_rate: Optional[int] = None


@auto_docstring
class MiniCPM_o_2_6PreTrainedModel(Qwen2PreTrainedModel):
    config: MiniCPM_o_2_6Config


class MiniCPM_o_2_6TextModel(Qwen2Model):
    pass


class MiniCPM_o_2_6ForConditionalGeneration(MiniCPM_o_2_6PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config: MiniCPM_o_2_6Config):
        super().__init__(config.text_config)

        text_config = config.text_config
        self.language_model = MiniCPM_o_2_6TextModel(text_config)
        self.vocab_size = text_config.vocab_size
        self.lm_head = nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False)

        self.omni_config = config

        # Initialize weights and apply final processing
        self.post_init()

        self.embed_dim = self.language_model.config.hidden_size
        self.scale_emb = getattr(self.language_model.config, "scale_emb", 1.0)

        # init vision module
        self.vpm = self.init_vision_module()
        self.vision_dim = self.vpm.embed_dim
        self.resampler = self.init_resampler(config.query_num, self.embed_dim, self.vision_dim)

        # init audio module
        self.apm = self.init_audio_module()
        audio_output_dim = int(self.apm.config.encoder_ffn_dim // 4)
        self.audio_avg_pooler = nn.AvgPool1d(self.omni_config.audio_pool_step, stride=self.omni_config.audio_pool_step)
        self.audio_projection_layer = MultiModalProjector(in_dim=audio_output_dim, out_dim=self.embed_dim)
        self.audio_encoder_layer = -1

        # init tts module
        assert _tts_deps, "please make sure vector_quantize_pytorch and vocos are installed."
        self.tts = self.init_tts_module()

        self.force_no_stop = False

        # for stream api
        self.reset_session()

    def reset_session(self):
        self.session_id = None
        self.new_user_msg = True
        self.llm_generated = False
        self.llm_generate_completed = False
        self.llm_past_key_values = None
        self.audio_past_key_values = None  # apm kv cache

    def init_tts(
        self,
        tts_text_tokenizer_path=None,
        vocos_ckpt_path=None,
    ):
        """
        load tts tokenizer and vocos
        1. try load form local 2. try load from huggingface
        """
        if tts_text_tokenizer_path is None:
            tts_text_tokenizer_path = os.path.join(self.omni_config._name_or_path, "assets/chattts_tokenizer")
        if not os.path.exists(tts_text_tokenizer_path):
            # try from hf model_id
            tts_text_tokenizer_path = "openbmb/chattts_tokenizer"

        tts_text_tokenizer = BertTokenizerFast.from_pretrained(tts_text_tokenizer_path)
        self.tts_processor = ChatTTSProcessor(text_tokenizer=tts_text_tokenizer)

        if vocos_ckpt_path is None:
            vocos_ckpt_path = os.path.join(self.omni_config._name_or_path, "assets/Vocos.pt")
        if not os.path.exists(vocos_ckpt_path):
            vocos_ckpt_path = hf_hub_download(repo_id="openbmb/MiniCPM-o-2_6", subfolder="assets", filename="Vocos.pt")

        assert os.path.exists(vocos_ckpt_path)
        self.vocos = self.initialize_vocos(vocos_ckpt_path)

    def initialize_vocos(self, ckpt_path):
        feature_extractor = instantiate_class(
            args=(),
            init={
                "class_path": "vocos.feature_extractors.MelSpectrogramFeatures",
                "init_args": {"sample_rate": 24000, "n_fft": 1024, "hop_length": 256, "n_mels": 100},
            },
        )
        backbone = instantiate_class(
            args=(),
            init={
                "class_path": "vocos.models.VocosBackbone",
                "init_args": {"input_channels": 100, "dim": 512, "intermediate_dim": 1536, "num_layers": 8},
            },
        )
        head = instantiate_class(
            args=(),
            init={"class_path": "vocos.heads.ISTFTHead", "init_args": {"dim": 512, "n_fft": 1024, "hop_length": 256}},
        )
        vocos = Vocos(feature_extractor, backbone, head).to("cuda").eval().to(torch.float32)
        vocos.load_state_dict(torch.load(ckpt_path, weights_only=True, mmap=True))
        return vocos

    def init_vision_module(self):
        if self.omni_config._attn_implementation == "flash_attention_2":
            self.omni_config.vision_config._attn_implementation = "flash_attention_2"
        else:
            self.omni_config.vision_config._attn_implementation = "eager"
        model = MiniCPMVisionTransformer(self.omni_config.vision_config)
        if self.omni_config.drop_vision_last_layer:
            model.encoder.layers = model.encoder.layers[:-1]

        setattr(model, "embed_dim", model.embeddings.embed_dim)
        setattr(model, "patch_size", model.embeddings.patch_size)

        return model

    def init_resampler(self, query_num, embed_dim, vision_dim):
        return Resampler(
            num_queries=query_num,
            embed_dim=embed_dim,
            num_heads=embed_dim // 128,
            kv_dim=vision_dim,
            adaptive=True,
        )

    def init_audio_module(self):
        model = MiniCPMWhisperEncoder(self.omni_config.audio_config)
        return model

    def init_tts_module(self):
        model = ConditionalChatTTS(self.omni_config.tts_config)
        return model

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.language_model = decoder

    def get_decoder(self):
        return self.language_model

    def subsequent_chunk_mask(
        self,
        size: int,
        chunk_size: int,
        num_left_chunks: int = -1,
        device: torch.device = torch.device("cpu"),
        num_lookhead: int = 0,
    ) -> torch.Tensor:
        """Create mask for subsequent steps (size, size) with chunk size,
        this is for streaming encoder

        Args:
            size (int): size of mask
            chunk_size (int): size of chunk
            num_left_chunks (int): number of left chunks
                <0: use full chunk
                >=0: use num_left_chunks
            device (torch.device): "cpu" or "cuda" or torch.Tensor.device

        Returns:
            torch.Tensor: mask

        Examples:
            >>> subsequent_chunk_mask(4, 2)
            [[1, 1, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1]]
        """
        idx = torch.arange(size, device=device)
        chunk_idx = idx // chunk_size

        if num_left_chunks < 0:
            starts = torch.zeros_like(chunk_idx)
        else:
            starts = ((chunk_idx - num_left_chunks).clamp(min=0)) * chunk_size
        ends = ((chunk_idx + 1) * chunk_size + num_lookhead).clamp(max=size)

        jdx = torch.arange(size, device=device).view(1, size).expand(size, size)
        starts_mat = starts.view(size, 1).expand(size, size)
        ends_mat = ends.view(size, 1).expand(size, size)

        mask = (jdx >= starts_mat) & (jdx < ends_mat)
        return mask

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers and the output length of the audio encoder
        """
        input_lengths_after_cnn = (input_lengths - 1) // 2 + 1
        input_lengths_after_pooling = (
            input_lengths_after_cnn - self.omni_config.audio_pool_step
        ) // self.omni_config.audio_pool_step + 1
        input_lengths_after_pooling = input_lengths_after_pooling.to(dtype=torch.int32)

        return input_lengths_after_cnn, input_lengths_after_pooling

    def get_image_features(self, pixel_values_list, tgt_sizes, dtype, device):
        vision_hidden_states = []
        all_pixel_values = []
        img_cnt = []
        for pixel_values in pixel_values_list:
            img_cnt.append(len(pixel_values))
            all_pixel_values.extend([i.flatten(end_dim=1).permute(1, 0) for i in pixel_values])

        tgt_sizes = [tgt_size for tgt_size in tgt_sizes if isinstance(tgt_size, torch.Tensor)]
        tgt_sizes = torch.vstack(tgt_sizes).type(torch.int32)

        max_patches = torch.max(tgt_sizes[:, 0] * tgt_sizes[:, 1])

        all_pixel_values = torch.nn.utils.rnn.pad_sequence(all_pixel_values, batch_first=True, padding_value=0.0)
        B, L, _ = all_pixel_values.shape
        all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(B, 3, -1, L)

        patch_attn_mask = torch.zeros((B, 1, max_patches), dtype=torch.bool, device=device)
        for i in range(B):
            patch_attn_mask[i, 0, : tgt_sizes[i][0] * tgt_sizes[i][1]] = True

        vision_batch_size = self.omni_config.vision_batch_size
        all_pixel_values = all_pixel_values.type(dtype)
        # edge-side model, senstive to gpu memory size in use
        # without batching, it will occur cuda oom
        if B > vision_batch_size:
            hs = []
            for i in range(0, B, vision_batch_size):
                start_idx = i
                end_idx = i + vision_batch_size
                tmp_hs = self.vpm(
                    all_pixel_values[start_idx:end_idx],
                    patch_attention_mask=patch_attn_mask[start_idx:end_idx],
                    tgt_sizes=tgt_sizes[start_idx:end_idx],
                ).last_hidden_state
                hs.append(tmp_hs)
            vision_embedding = torch.cat(hs, dim=0)
        else:
            vision_embedding = self.vpm(
                all_pixel_values, patch_attention_mask=patch_attn_mask, tgt_sizes=tgt_sizes
            ).last_hidden_state

        vision_embedding = self.resampler(vision_embedding, tgt_sizes)

        start = 0
        for pixel_values in pixel_values_list:
            img_cnt = len(pixel_values)
            if img_cnt > 0:
                vision_hidden_states.append(vision_embedding[start : start + img_cnt])
                start += img_cnt
            else:
                vision_hidden_states.append([])

        return vision_hidden_states

    def get_vllm_embedding(self, tgt_sizes=None, pixel_values=None, image_bound=None, input_ids=None, vision_hidden_states=None, **kwargs):
        """
        Compute all visual embeddings, and set into llm embeddings.
        Args:
            - tgt_sizes: image size after patch embedding
            - pixel_values: image features
            - image_bound: position of each picture corresponding to input_ids
            - input_ids: full input_ids, include placeholder
            - vision_hidden_states: if already have vision hidden states, can input directly

        Returns:
            embedding with vision, vision_hidden_states
        """
        dtype = self.language_model.embed_tokens.weight.dtype
        device = self.language_model.embed_tokens.weight.device
        if not vision_hidden_states:
            if pixel_values and len(pixel_values[0]) > 0:
                vision_hidden_states = self.get_image_features(pixel_values, tgt_sizes, dtype, device)
            else:
                vision_hidden_states = [[]] * len(pixel_values)

        vllm_embedding = self.language_model.embed_tokens(input_ids) * self.scale_emb

        new_vllm_embedding = vllm_embedding.clone()

        vision_hidden_states = [
            i.type(vllm_embedding.dtype) if isinstance(i, torch.Tensor) else i for i in vision_hidden_states
        ]

        bs = len(input_ids)
        for i in range(bs):
            cur_vs_hs = vision_hidden_states[i]
            if len(cur_vs_hs) > 0:
                cur_vllm_emb = vllm_embedding[i]
                cur_image_bound = image_bound[i]
                if len(cur_image_bound) > 0:
                    image_indices = torch.stack(
                        [torch.arange(r[0], r[1], dtype=torch.long) for r in cur_image_bound]
                    ).to(vllm_embedding.device)

                    new_vllm_embedding[i] = cur_vllm_emb.scatter(
                        0,
                        image_indices.view(-1, 1).repeat(1, cur_vllm_emb.shape[-1]),
                        cur_vs_hs.view(-1, cur_vs_hs.shape[-1]),
                    )

                elif self.training:
                    new_vllm_embedding[i] += cur_vs_hs[0].mean() * 0

        return new_vllm_embedding, vision_hidden_states

    def get_audio_embedding_streaming(
        self, audio_features: torch.FloatTensor = [], audio_feature_lens_raw: list[list[int]] = []
    ):
        r"""
        Extract audio embeddings in a streaming manner using cached key-value pairs.

        This method processes incoming audio features incrementally and stores/updates `past_key_values`
        for faster inference on subsequent audio frames. It only supports batch_size=1 and is intended
        for streaming scenarios.

        Args:
            audio_features (torch.FloatTensor): Input mel-spectrograms of shape `(batch_size, 80, frames)`.
            audio_feature_lens_raw (List[List[int]]): Lengths of each audio segment for each item in the batch.

        Returns:
            List[List[torch.Tensor]]: audio embeddings
        """
        # exist audio
        if len(audio_features) > 0:
            audio_feature_lens = torch.hstack(audio_feature_lens_raw)
            batch_size, _, max_mel_seq_len = audio_features.shape
            assert batch_size == 1
            max_seq_len = (max_mel_seq_len - 1) // 2 + 1

            if self.audio_past_key_values is not None:
                cache_length = self.audio_past_key_values[0][0].shape[2]
                apm_max_len = self.apm.embed_positions.weight.shape[0]
                if cache_length + max_seq_len >= apm_max_len:
                    logger.warning(
                        f"audio_past_key_values length {cache_length + max_seq_len} exceed {apm_max_len}, reset."
                    )
                    self.audio_past_key_values = None

            audio_outputs = self.apm(audio_features, past_key_values=self.audio_past_key_values, use_cache=True)
            audio_states = audio_outputs.last_hidden_state  # [:, :audio_feat_lengths, :]
            self.audio_past_key_values = audio_outputs.past_key_values

            audio_embeds = self.audio_projection_layer(audio_states)

            audio_embeds = audio_embeds.transpose(1, 2)
            audio_embeds = self.audio_avg_pooler(audio_embeds)
            audio_embeds = audio_embeds.transpose(1, 2)

            _, feature_lens_after_pooling = self._get_feat_extract_output_lengths(audio_feature_lens)

            num_audio_tokens = feature_lens_after_pooling

            final_audio_embeds = []
            idx = 0
            for i in range(len(audio_feature_lens_raw)):
                target_audio_embeds = []
                for _ in range(len(audio_feature_lens_raw[i])):
                    target_audio_embeds.append(audio_embeds[idx, : num_audio_tokens[idx], :])
                    idx += 1
                final_audio_embeds.append(target_audio_embeds)
            return final_audio_embeds
        else:
            return []

    def get_audio_embedding(
        self,
        audio_features: torch.FloatTensor = [],
        audio_feature_lens_raw: list[list[int]] = [],
        chunk_length=-1,
        dummy=True,
    ):
        r"""
        Extract full audio embeddings with optional chunk-based attention.

        This method computes embeddings for all audio frames at once, either using full attention (when
        `chunk_length` is -1) or chunk-based attention (when `chunk_length` is a positive number). It does
        not use key-value caching and is suitable for non-streaming inference.

        Args:
            audio_features (torch.FloatTensor): Input mel-spectrograms of shape `(batch_size, 80, frames)`.
            audio_feature_lens_raw (List[List[int]]): Lengths of each audio segment for each item in the batch.
            chunk_length (int, optional): Determines whether to use full attention (-1) or chunk-based
                attention (>0) during embedding computation.

        Returns:
            List[List[torch.Tensor]]: audio embeddings
        """
        # exist audio
        if len(audio_features) > 0:
            audio_feature_lens = torch.hstack(audio_feature_lens_raw)
            batch_size, _, max_mel_seq_len = audio_features.shape
            max_seq_len = (max_mel_seq_len - 1) // 2 + 1

            # Create a sequence tensor of shape (batch_size, max_seq_len)
            seq_range = (
                torch.arange(0, max_seq_len, dtype=audio_feature_lens.dtype, device=audio_feature_lens.device)
                .unsqueeze(0)
                .expand(batch_size, max_seq_len)
            )
            lengths_expand = audio_feature_lens.unsqueeze(1).expand(batch_size, max_seq_len)
            # Create mask
            padding_mask = seq_range >= lengths_expand  # 1 for padded values

            audio_attention_mask_ = padding_mask.view(batch_size, 1, 1, max_seq_len).expand(
                batch_size, 1, max_seq_len, max_seq_len
            )
            audio_attention_mask = audio_attention_mask_.to(
                dtype=self.apm.conv1.weight.dtype, device=self.apm.conv1.weight.device
            )

            if chunk_length > 0:
                chunk_num_frame = int(chunk_length * 50)
                chunk_mask = self.subsequent_chunk_mask(
                    size=max_seq_len,
                    chunk_size=chunk_num_frame,
                    num_left_chunks=-1,
                    device=audio_attention_mask_.device,
                )
                audio_attention_mask_ = torch.logical_or(audio_attention_mask_, torch.logical_not(chunk_mask))

            audio_attention_mask[audio_attention_mask_] = float("-inf")
            audio_states = self.apm(
                audio_features, output_hidden_states=True, attention_mask=audio_attention_mask
            ).hidden_states[self.audio_encoder_layer]
            audio_embeds = self.audio_projection_layer(audio_states)

            audio_embeds = audio_embeds.transpose(1, 2)
            audio_embeds = self.audio_avg_pooler(audio_embeds)
            audio_embeds = audio_embeds.transpose(1, 2)

            _, feature_lens_after_pooling = self._get_feat_extract_output_lengths(audio_feature_lens)

            num_audio_tokens = feature_lens_after_pooling

            final_audio_embeds = []
            idx = 0
            for i in range(len(audio_feature_lens_raw)):
                target_audio_embeds = []
                for _ in range(len(audio_feature_lens_raw[i])):
                    target_audio_embeds.append(audio_embeds[idx, : num_audio_tokens[idx], :])
                    idx += 1
                final_audio_embeds.append(target_audio_embeds)
            return final_audio_embeds
        else:
            return []

    def get_omni_embedding(self, input_embeddings, chunk_length=-1, stream_input=False, audio_features = [], audio_feature_lens = [], audio_bounds=None, **kwargs):
        """
        Args:
            input_embeddings:
            chunk_length: whisper use full attention or chunk attention
            stream_input: use streaming audio embedding
        Returns:
            final embeddings with audio feature
        """
        if stream_input:
            audio_embeddings = self.get_audio_embedding_streaming(
                audio_features, audio_feature_lens
            )
        else:
            audio_embeddings = self.get_audio_embedding(
                audio_features, audio_feature_lens, chunk_length
            )

        bs = len(input_embeddings)
        if len(audio_features) > 0:
            assert len(audio_embeddings) == len(input_embeddings)

            if self.omni_config.chunk_input:
                for i in range(bs):
                    audio_embs = torch.cat(audio_embeddings[i], dim=0).to(
                        device=input_embeddings.device, dtype=input_embeddings.dtype
                    )
                    audio_start_pos = 0
                    for bound in audio_bounds[i]:
                        audio_len = bound[1] - bound[0]
                        input_embeddings[i, bound[0] : bound[1]] = audio_embs[
                            audio_start_pos : audio_start_pos + audio_len, :
                        ]
                        audio_start_pos += audio_len
            else:
                for i in range(bs):
                    audio_embs = audio_embeddings[i]
                    bounds = audio_bounds[i]
                    for embs, bound in zip(audio_embs, bounds):
                        audio_indices = torch.arange(bound[0], bound[1], dtype=torch.long).to(input_embeddings.device)

                        if embs.shape[0] != len(audio_indices):
                            raise ValueError(
                                f"Shape mismatch: Trying to assign embeddings of shape {embs.shape} "
                                f"to input indices of length {len(audio_indices)}"
                            )
                        input_embeddings[i, audio_indices] = embs.to(input_embeddings.dtype)
        elif self.training:
            for i in range(bs):
                # dummy audio_embeddings
                input_embeddings = input_embeddings + audio_embeddings[0].mean() * 0

        return input_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen2ForCausalLM

        >>> model = Qwen2ForCausalLM.from_pretrained("openbmb/MiniCPM-o-2_6")
        >>> tokenizer = AutoTokenizer.from_pretrained("openbmb/MiniCPM-o-2_6")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.omni_config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.omni_config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _decode_stream(self, inputs_embeds, processor, **kwargs):
        streamer = TextIteratorStreamer(tokenizer=processor.tokenizer)
        generation_kwargs = {
            "inputs_embeds": inputs_embeds,
            "pad_token_id": 0,
            "eos_token_id": processor.terminator_ids,
            "streamer": streamer,
        }
        generation_kwargs.update(kwargs)

        thread = Thread(target=super().generate, kwargs=generation_kwargs)
        thread.start()

        return streamer

    def generate(
        self,
        input_ids=None,
        pixel_values=None,
        tgt_sizes=None,
        audio_features=[],
        audio_feature_lens=None,
        image_bound=None,
        audio_bounds=None,
        spk_bounds=None,
        attention_mask=None,
        processor=None,
        vision_hidden_states=None,
        stream=False,
        generate_audio=False,
        return_spk_embed=False,
        return_dict=False,
        batched=False,
        use_tts_template=False,
        sampling=False,
        max_new_tokens=2048,
        min_new_tokens=0,
        force_no_stop=False,
        **kwargs,
    ):
        assert input_ids is not None
        assert len(input_ids) == len(pixel_values)
        assert sampling or not stream, "if use stream mode, make sure sampling=True"

        kwargs.pop("image_sizes")
        if sampling:
            generation_config = {
                "top_p": 0.8,
                "top_k": 100,
                "temperature": 0.7,
                "do_sample": True,
                "repetition_penalty": 1.05,
            }
        else:
            generation_config = {
                "num_beams": 3,
                "repetition_penalty": 1.2,
            }
        generation_config["max_new_tokens"] = max_new_tokens
        generation_config["min_new_tokens"] = min_new_tokens
        generation_config.update((k, kwargs[k]) for k in generation_config.keys() & kwargs.keys())

        if generate_audio or return_spk_embed:
            return_dict = True

        model_inputs = {
            "input_ids": input_ids,
            "audio_features": audio_features,
            "audio_feature_lens": audio_feature_lens,
            "image_bound": image_bound,
            "audio_bounds": audio_bounds,
            "spk_bounds": spk_bounds,
        }

        if vision_hidden_states is None:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["tgt_sizes"] = tgt_sizes
        else:
            model_inputs["vision_hidden_states"] = vision_hidden_states

        with torch.inference_mode():
            model_inputs["inputs_embeds"], vision_hidden_states = self.get_vllm_embedding(**model_inputs)
            model_inputs["inputs_embeds"] = self.get_omni_embedding(
                input_embeddings=model_inputs["inputs_embeds"],
                chunk_length=self.omni_config.audio_chunk_length,
                **model_inputs,
            )

            if stream:
                result = self._decode_stream(model_inputs["inputs_embeds"], processor, **generation_config)
            else:
                outputs = super().generate(
                    inputs_embeds=model_inputs["inputs_embeds"],
                    pad_token_id=0,
                    eos_token_id=processor.terminator_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                    **generation_config,
                )

        if stream:

            def stream_gen():
                for text in result:
                    for term in processor.terminators:
                        text = text.replace(term, "")
                    yield text

            if return_dict:
                return OmniOutput(text=stream_gen())
            else:
                return stream_gen()

        else:
            spk_embeds = wav_numpy = sr = None

            if not batched and use_tts_template and generate_audio:
                result = processor.decode(outputs.sequences)
                mel_spec = self._generate_mel_spec(
                    model_inputs,
                    outputs,
                    result[0],
                    tts_config={"top_p": 0.7, "top_k": 20, "repetition_penalty": 1.0},
                    force_no_stop=force_no_stop,
                )
                wav_numpy, sr = self.decode_mel_to_audio(mel_spec, kwargs.get("output_audio_path", None))

            if return_spk_embed:
                spk_embeds = self._get_last_spk_embeds(model_inputs, outputs)

            if return_dict:
                return OmniOutput(outputs=outputs, spk_embeds=spk_embeds, audio_wav=wav_numpy, sampling_rate=sr)
            else:
                return outputs

    @torch.inference_mode()
    def streaming_prefill(
        self,
        session_id,
        msgs,
        processor,
        omni_input=True,
        max_slice_nums=None,
        ls_temperature=1.0,
        **kwargs,
    ):
        """
        Streaming video/audio input and output audio stream, Only support batch_size=1
        Args:
            session_id: Note: new connection should use a new session_id
        """
        assert session_id is not None
        if self.session_id is None or session_id != self.session_id:  # new session
            self.is_first = True
        else:
            self.is_first = False

        images = []
        audios = []

        assert len(msgs) == 1
        copy_msgs = deepcopy(msgs)
        msg = copy_msgs[0]

        assert msg["role"] in ["system", "user", "assistant"]

        content = msg["content"]
        cur_msgs = []
        for j, c in enumerate(content):
            if isinstance(c, Image.Image):
                images.append(c)
                cur_msgs.append("(<image>./</image>)")
            elif isinstance(c, np.ndarray):  # audio
                audios.append(c)
                cur_msgs.append("(<audio>./</audio>)")
            elif isinstance(c, str):
                cur_msgs.append(c)
            else:
                logger.error("Invalid content type:", c)

        cur_contents = "".join(cur_msgs) if omni_input else "\n".join(cur_msgs)
        if not self.is_first and self.new_user_msg and msg["role"] == "user":  # new user add im_start
            if self.llm_generated:
                if self.llm_generate_completed:
                    msg["content"] = "<|im_end|>\n<|im_start|>user\n" + cur_contents
                else:  # break llm gen, add tts_eos
                    msg["content"] = "<|tts_eos|><|im_end|>\n<|im_start|>user\n" + cur_contents
            else:
                msg["content"] = "<|im_start|>user\n" + cur_contents
            self.new_user_msg = False
        else:
            msg["content"] = cur_contents

        if msg["role"] in ["system", "assistant"]:
            self.new_user_msg = True
            self.audio_past_key_values = None  # apm kv cache

        if self.is_first:
            # init pask_key_values
            logger.info(f"new session_id: {session_id}, reset kv cache")
            self.reset_session()
            self.session_id = session_id

            prompt = processor.tokenizer.apply_chat_template(
                copy_msgs,
                tokenize=False,
                add_generation_prompt=False,
                chat_template=processor.tts_chat_template,
            )
            add_special_tokens = True  # add bos
        else:
            prompt = copy_msgs[0]["content"]
            add_special_tokens = False

        model_inputs = processor(
            [prompt],
            [images] if images else None,
            [audios] if audios else None,
            max_slice_nums=1 if max_slice_nums is None else max_slice_nums,
            use_image_id=False,
            chunk_input=True,
            return_tensors="pt",
            max_length=None,
            sampling_rate=16000,
            add_special_tokens=add_special_tokens,
        ).to(self.device)

        # 1. prepare input embeddings
        model_inputs["inputs_embeds"], _ = self.get_vllm_embedding(**model_inputs)
        # get audio embedding with audio_past_key_values
        inputs_embeds = self.get_omni_embedding(
            input_embeddings=model_inputs["inputs_embeds"], stream_input=True, **model_inputs
        )

        if self.is_first:
            # clean audio_past_key_values after first prefill
            self.audio_past_key_values = None

        if self.llm_past_key_values is not None:
            cache_length = self.llm_past_key_values[0][0].shape[2]
        else:
            cache_length = 0

        attention_mask = torch.ones((1, cache_length + inputs_embeds.shape[1]), dtype=torch.bool, device=self.device)

        # 2. do prefill and predict listen/speak label
        outputs = self.forward(
            past_key_values=self.llm_past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=None,
            use_cache=True,
            return_dict=True,
        )

        self.llm_past_key_values = outputs["past_key_values"]
        return

    # Copy and modified from transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation
    # if use the same method in `GenerationMixin`, it will cause an error in `_cache_dependant_input_preparation` when use stream, the error message is as follows:
    #   (cache_position[-1] >= input_ids.shape[1])  # Exception 3
    #   IndexError: index -1 is out of bounds for dimension 0 with size 0
    # in there the cache_position=[]
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                # past_length = past_key_values.seen_tokens
                past_length = past_key_values.get_seq_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusivelly passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

        # print("--------- my prepare_inputs_for_generation")
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            # The clone here is for the same reason as for `position_ids`.
            model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
                device = model_inputs["inputs_embeds"].device
            else:
                batch_size, sequence_length = model_inputs["input_ids"].shape
                device = model_inputs["input_ids"].device

            dtype = self.lm_head.weight.dtype
            min_dtype = torch.finfo(dtype).min

            attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_length(),
                dtype=dtype,
                device=device,
                min_dtype=min_dtype,
                cache_position=cache_position,
                batch_size=batch_size,
            )

        model_inputs.update(
            {
                "position_ids": position_ids,
                # "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @torch.inference_mode()
    def streaming_generate(
        self,
        session_id,
        processor,
        max_new_tokens=512,
        min_new_tokens=0,
        sampling=True,
        generate_audio=True,
        enable_regenerate=False,
        force_no_stop=False,
        **kwargs,
    ):
        """
        Streaming video/audio input and output audio stream
        Args:
        """
        tokenizer = processor.tokenizer
        if sampling:
            generation_config = {
                "top_p": 0.8,
                "top_k": 100,
                "temperature": 0.7,
                "do_sample": True,
                "repetition_penalty": 1.05,
            }
        else:
            generation_config = {
                "num_beams": 3,
                "repetition_penalty": 1.2,
            }
        generation_config["min_new_tokens"] = min_new_tokens
        generation_config.update((k, kwargs[k]) for k in generation_config.keys() & kwargs.keys())

        # do generate
        # reset buffer
        self.new_user_msg = True
        self.llm_generated = True
        self.llm_generate_completed = False
        self.audio_past_key_values = None  # apm kv cache

        generate_prompt = "<|im_end|>\n<|im_start|>assistant\n<|spk_bos|><|spk|><|spk_eos|><|tts_bos|>"
        input_ids = tokenizer(generate_prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].cuda()

        spk_start_idx = torch.where(input_ids[0] == tokenizer.spk_start_id)[0]
        spk_end_idx = torch.where(input_ids[0] == tokenizer.spk_end_id)[0]
        spk_bounds = [
            torch.hstack([(spk_start_idx + 1).unsqueeze(-1), spk_end_idx.unsqueeze(-1)])
        ]  # List[Tensor], (1,2)

        cache_length = past_length = self.llm_past_key_values[0][0].shape[2]
        attention_mask = torch.ones((1, cache_length + input_ids.shape[1]), dtype=torch.bool, device=self.device)

        generation_config["max_new_tokens"] = max_new_tokens
        streamer = self.llm_generate_chunk(
            input_ids, attention_mask, tokenizer, processor.terminator_ids, generation_config, processor
        )

        if generate_audio:
            result = self._generate_mel_spec_audio_streaming(
                spk_bounds,
                streamer,
                output_chunk_size=25,
                enable_regenerate=enable_regenerate,
                tts_config={"top_p": 0.7, "top_k": 20, "repetition_penalty": 1.0},
                force_no_stop=force_no_stop,
            )
            return result
        else:
            return streamer

    def llm_generate_chunk(self, input_ids, attention_mask, tokenizer, terminators, generation_config, processor):
        def check_uncompleted_token(ids):
            cur_text = tokenizer.decode(ids)
            end = len(ids)
            while cur_text[-1] == "�":
                end -= 1
                if end == 0:
                    break
                cur_text = tokenizer.decode(ids[:end])
            return end

        max_new_tokens = int(generation_config.pop("max_new_tokens", 2048))
        new_len = 0
        first_chunk = True
        eos = False
        left_ids = None

        while True:
            outputs = super().generate(
                input_ids=input_ids,
                past_key_values=self.llm_past_key_values.to_legacy_cache()
                if isinstance(self.llm_past_key_values, DynamicCache)
                else self.llm_past_key_values,
                attention_mask=attention_mask,
                use_cache=True,
                max_new_tokens=3,  # reduce first token delay
                pad_token_id=0,
                output_hidden_states=True if first_chunk else False,
                return_dict_in_generate=True,
                eos_token_id=terminators,
                **generation_config,
            )
            if outputs.sequences[0, -1] in terminators:
                eos = True
            input_len = input_ids.shape[1]
            cur_ids = outputs.sequences[:, input_len:]
            new_len += cur_ids.shape[1]

            if left_ids is not None and left_ids.shape[1] > 0:
                cur_ids = torch.cat([left_ids, cur_ids], dim=1)
            end = check_uncompleted_token(cur_ids[0])
            left_ids = cur_ids[:, end:]
            cur_ids = cur_ids[:, :end]
            text = processor.decode(cur_ids)[0] if end > 0 else ""

            self.llm_past_key_values = outputs.past_key_values
            input_ids = outputs.sequences[:, -1:]
            cache_length = past_length = self.llm_past_key_values[0][0].shape[2]
            attention_mask = torch.ones((1, cache_length + input_ids.shape[1]), dtype=torch.bool, device=self.device)

            res = {"text": text}
            if first_chunk:
                res["hidden_states"] = outputs.hidden_states
                first_chunk = False
            yield res

            if eos:
                self.llm_generate_completed = True
                break
            if new_len >= max_new_tokens:
                logger.debug(f"LLM generation {new_len} exceeds max_new_tokens({max_new_tokens}), break.")
                break

    def prepare_tts_text(self, text):
        tts_tokens = self.tts_processor.text_tokenizer.encode(text, add_special_tokens=False)
        tts_tokens_len = len(tts_tokens)
        if tts_tokens_len < self.tts.streaming_text_reserved_len:
            num_pad_tokens = self.tts.streaming_text_reserved_len - tts_tokens_len

            pad_str = "[Etts]" + "[PAD]" * (num_pad_tokens - 1)
        else:
            tts_tokens = tts_tokens[0 : self.tts.streaming_text_reserved_len]
            tts_tokens_len = len(tts_tokens)
            text = self.tts_processor.text_tokenizer.decode(tts_tokens, add_special_tokens=False)
            pad_str = ""
        spk_emb_placeholder_tts = "[spk_emb]" * self.tts.num_spk_embs

        new_text_tts = f"[Stts]{spk_emb_placeholder_tts}{text}{pad_str}[Ptts]"
        return new_text_tts, tts_tokens_len

    def get_tts_text_start_token_ids(self):
        text = "[Stts]" + "[spk_emb]" * self.tts.num_spk_embs
        tts_input_ids = self.tts_processor.text_tokenizer(text, return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ].cuda()
        return tts_input_ids

    def _build_streaming_mask(self, tts_tokens_len):
        tts_sequence_full_length = (
            1 + self.tts.num_spk_embs * self.tts.use_speaker_embedding + self.tts.streaming_text_reserved_len + 1
        )
        streaming_attention_mask = torch.zeros(tts_sequence_full_length, dtype=torch.int8)
        streaming_attention_mask[0 : 1 + 1 + tts_tokens_len + 1] = 1
        streaming_attention_mask[-1] = 1
        return streaming_attention_mask

    def _get_last_spk_embeds(self, inputs, outputs):
        last_hidden_states = [hs[-1] for hs in outputs.hidden_states]

        # batch = 1
        last_hidden_states = torch.vstack([i[0] for i in last_hidden_states])

        # last spk
        spk_bound = inputs["spk_bounds"][0][-1]

        spk_embeds = last_hidden_states[spk_bound[0] : spk_bound[1]]
        return spk_embeds

    def _generate_mel_spec(
        self,
        inputs,
        outputs,
        text,
        force_no_stop=False,
        output_chunk_size=25,
        tts_max_new_tokens=2048,
        tts_config: dict = {"top_p": 0.7, "top_k": 20, "repetition_penalty": 1.0},
    ):
        spk_embeds = self._get_last_spk_embeds(inputs, outputs)

        text = text.split("<|tts_bos|>")[-1]
        gen_text = text.split("<|tts_eos|>")[0]
        tts_text, tts_token_lens = self.prepare_tts_text(gen_text)
        tts_inputs = self.tts_processor.text_tokenizer.encode(tts_text, add_special_tokens=False)
        tts_input_ids = torch.Tensor(tts_inputs).unsqueeze(0).to("cuda", dtype=torch.long)
        streaming_tts_text_mask = self._build_streaming_mask(tts_token_lens).to(device=self.tts.device)

        logits_warpers, logits_processors = gen_logits(
            num_code=626,
            top_P=tts_config["top_p"],
            top_K=tts_config["top_k"],
            repetition_penalty=tts_config["repetition_penalty"],
        )

        condition_length = (
            1 + self.tts.use_speaker_embedding * self.tts.num_spk_embs + self.tts.streaming_text_reserved_len + 1
        )

        dtype = self.tts.emb_text.weight.dtype
        emb = torch.zeros(1, condition_length, self.tts.num_vq, dtype=dtype, device=self.tts.device)
        past_key_values = [
            (
                torch.zeros(
                    1,
                    self.tts.config.num_attention_heads,
                    condition_length - 1,
                    self.tts.config.hidden_size // self.tts.config.num_attention_heads,
                    dtype=emb.dtype,
                    device=self.tts.device,
                ),
                torch.zeros(
                    1,
                    self.tts.config.num_attention_heads,
                    condition_length - 1,
                    self.tts.config.hidden_size // self.tts.config.num_attention_heads,
                    dtype=emb.dtype,
                    device=self.tts.device,
                ),
            )
            for _ in range(self.tts.config.num_hidden_layers)
        ]

        audio_input_ids = torch.zeros(1, condition_length, self.tts.num_vq, dtype=torch.long, device=self.tts.device)

        eos_lab = False
        for chunk_idx in range(math.ceil(emb.shape[1] / self.tts.streaming_text_chunk_size)):
            if chunk_idx == 0:
                begin = chunk_idx * self.tts.streaming_text_chunk_size + 0
                end = (
                    (chunk_idx + 1) * self.tts.streaming_text_chunk_size
                    + 1
                    + self.tts.use_speaker_embedding * self.tts.num_spk_embs
                )
            else:
                begin = (
                    chunk_idx * self.tts.streaming_text_chunk_size
                    + 1
                    + self.tts.use_speaker_embedding * self.tts.num_spk_embs
                )
                end = min(
                    (chunk_idx + 1) * self.tts.streaming_text_chunk_size
                    + 1
                    + self.tts.use_speaker_embedding * self.tts.num_spk_embs,
                    condition_length - 1,
                )

            if end - begin > 0:
                text_input_ids = tts_input_ids[:, begin:end]
                position_ids = torch.arange(begin, end, dtype=torch.long, device=self.tts.device).unsqueeze(0)

                if begin == 0:
                    past_key_values = self.tts.prefill_text(
                        input_ids=text_input_ids,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        lm_spk_emb_last_hidden_states=spk_embeds,
                    )
                else:
                    past_key_values = self.tts.prefill_text(
                        input_ids=text_input_ids, position_ids=position_ids, past_key_values=past_key_values
                    )

            outputs = self.tts.generate(
                input_ids=audio_input_ids,
                past_key_values=past_key_values,
                streaming_tts_text_mask=streaming_tts_text_mask,
                max_new_token=output_chunk_size,
                force_no_stop=force_no_stop,
                temperature=torch.tensor([0.1, 0.3, 0.1, 0.3], dtype=torch.float, device=self.tts.device),
                eos_token=torch.tensor([625], dtype=torch.long, device=self.tts.device),
                logits_warpers=logits_warpers,
                logits_processors=logits_processors,
            )
            audio_input_ids = outputs.audio_input_ids
            past_key_values = outputs.past_key_values

            if outputs.finished:
                logger.debug("Generation finished.")
                eos_lab = True
                break

        if not eos_lab:
            logger.debug("eos_lab False, Generation continue.")
            while True:
                outputs = self.tts.generate(
                    input_ids=audio_input_ids,
                    past_key_values=past_key_values,
                    streaming_tts_text_mask=streaming_tts_text_mask,
                    max_new_token=output_chunk_size,
                    force_no_stop=force_no_stop,
                    temperature=torch.tensor([0.1, 0.3, 0.1, 0.3], dtype=torch.float, device=self.tts.device),
                    eos_token=torch.tensor([625], dtype=torch.long, device=self.tts.device),
                    logits_warpers=logits_warpers,
                    logits_processors=logits_processors,
                )

                audio_input_ids = outputs.audio_input_ids
                past_key_values = outputs.past_key_values

                if outputs.finished:
                    logger.debug("Generation finished.")
                    break
                if outputs.new_ids.shape[1] > tts_max_new_tokens:
                    logger.debug(f"Generation length > {tts_max_new_tokens}, stopped.")
                    break

        mel_spec = self.tts.decode_to_mel_specs(outputs.new_ids)
        return mel_spec

    def _linear_overlap_add2_wav(self, frames: list[torch.Tensor], overlap: int):
        """
        Merge two audio waveforms with smooth in streaming audio generation.
        Borrowed some codes from `https://github.com/huggingface/transformers/blob/main/src/transformers/models/encodec/modeling_encodec.py`
        """
        assert len(frames) == 2
        device = frames[0].device
        dtype = frames[0].dtype
        # shape = frames[0].shape[:-1]

        frame0_length = frames[0].shape[-1]
        frame1_length = frames[1].shape[-1]
        total_size = frame0_length + frame1_length - overlap
        weight_len = max(frame0_length, frame1_length) + overlap
        t = torch.linspace(0, 1, weight_len + 2, device=device, dtype=dtype)[1:-1]
        weight = 0.5 - (t - 0.5).abs()

        sum_weight = torch.zeros(total_size, device=device, dtype=dtype)
        out = torch.zeros(total_size, device=device, dtype=dtype)
        offset: int = 0

        out[offset : offset + frame0_length] += weight[-frame0_length:] * frames[0]
        sum_weight[offset : offset + frame0_length] += weight[-frame0_length:]
        offset += frame0_length - overlap
        out[offset : offset + frame1_length] += weight[:frame1_length] * frames[1]
        sum_weight[offset : offset + frame1_length] += weight[:frame1_length]

        assert sum_weight.min() > 0
        out = out / sum_weight
        return out[:frame0_length], out[frame0_length:]

    def _generate_mel_spec_audio_streaming(
        self,
        spk_bounds,
        streamer,
        output_chunk_size=25,
        spk_embeds=None,
        prev_seg_text_ids=None,
        prev_seg_text_left="",
        prev_seg_audio_ids=None,
        enable_regenerate=False,
        force_no_stop=False,
        tts_config: dict = {"top_p": 0.7, "top_k": 20, "repetition_penalty": 1.0},
    ):
        # get spk_embedding
        gen_text = ""
        tts_text = ""
        new_segment_gen = False
        if spk_embeds is None:
            spk_bound = spk_bounds[0][-1]
            r = next(streamer)
            txt = r["text"]
            gen_text += txt.split("<|tts_eos|>")[0]
            tts_text, tts_token_lens = self.prepare_tts_text(gen_text)
            last_hidden_states = r["hidden_states"][0][-1][0]  # output: (input_seq_len, dim)
            spk_embeds = last_hidden_states[spk_bound[0] : spk_bound[1]]

        # init past_key_values
        logits_warpers, logits_processors = gen_logits(
            num_code=626,
            top_P=tts_config["top_p"],
            top_K=tts_config["top_k"],
            repetition_penalty=tts_config["repetition_penalty"],
        )
        condition_length = (
            1 + self.tts.use_speaker_embedding * self.tts.num_spk_embs + self.tts.streaming_text_reserved_len + 1
        )
        tts_start_token_len = 1 + self.tts.use_speaker_embedding * self.tts.num_spk_embs
        dtype = self.tts.emb_text.weight.dtype
        past_key_values = [
            (
                torch.zeros(
                    1,
                    self.tts.config.num_attention_heads,
                    condition_length - 1,
                    self.tts.config.hidden_size // self.tts.config.num_attention_heads,
                    dtype=dtype,
                    device=self.tts.device,
                ),
                torch.zeros(
                    1,
                    self.tts.config.num_attention_heads,
                    condition_length - 1,
                    self.tts.config.hidden_size // self.tts.config.num_attention_heads,
                    dtype=dtype,
                    device=self.tts.device,
                ),
            )
            for _ in range(self.tts.config.num_hidden_layers)
        ]
        audio_input_ids = torch.zeros(1, condition_length, self.tts.num_vq, dtype=torch.long, device=self.tts.device)

        # prefill prev segment for smooth
        chunk_idx = 0
        new_ids_len = 0
        prev_text_len = 0
        if prev_seg_text_ids is not None and prev_seg_audio_ids is not None:
            tts_token_lens = prev_seg_text_ids.shape[1]
            # assert tts_token_lens % self.tts.streaming_text_chunk_size == 0
            streaming_tts_text_mask = self._build_streaming_mask(tts_token_lens).to(device=self.tts.device)
            position_ids = torch.arange(
                0, tts_token_lens + tts_start_token_len, dtype=torch.long, device=self.tts.device
            ).unsqueeze(0)

            text_input_ids = self.get_tts_text_start_token_ids()
            text_input_ids = torch.cat([text_input_ids, prev_seg_text_ids], dim=1)
            past_key_values = self.tts.prefill_text(
                input_ids=text_input_ids,
                position_ids=position_ids,
                past_key_values=past_key_values,
                lm_spk_emb_last_hidden_states=spk_embeds,
            )
            past_key_values = self.tts.prefill_audio_ids(
                input_ids=prev_seg_audio_ids[:, :-1, :],
                # not prefill last id, which will be input_id of next generation
                past_key_values=past_key_values,
                streaming_tts_text_mask=streaming_tts_text_mask,
            )

            # update init
            chunk_idx += int(tts_token_lens / self.tts.streaming_text_chunk_size)
            audio_input_ids = torch.cat([audio_input_ids, prev_seg_audio_ids], dim=1)
            text = self.tts_processor.text_tokenizer.decode(prev_seg_text_ids[0].tolist(), add_special_tokens=False)

            gen_text += text
            gen_text += prev_seg_text_left
            prev_text_len = len(gen_text)  # takecare the position
            new_ids_len += prev_seg_audio_ids.shape[1]

        prev_wav = None
        eos_lab = False
        stop = False
        shift_len = 180
        voice_checker = VoiceChecker()
        number_converter = NumberToTextConverter()
        lang = None
        gen_text_raw = gen_text
        for t, r in enumerate(streamer):
            t += 1
            txt = r["text"]
            txt = txt.split("<|tts_eos|>")[0]
            gen_text_raw += txt
            if t == 1 and txt == "" and prev_seg_text_ids is not None:
                logger.warning("New segment is empty, generation finished.")
                return
            if t <= 2:  # do just one time, more token greater certainty
                lang = number_converter.detect_language(gen_text_raw)
            gen_text += number_converter.replace_numbers_with_text(txt, lang).replace("*", "")  # markdown **

            # TODO speed up
            tts_text, tts_token_lens = self.prepare_tts_text(gen_text)

            if tts_token_lens >= self.tts.streaming_text_reserved_len - shift_len:
                end_c = sentence_end(txt)
                if end_c:
                    end_c_idx = gen_text.rfind(end_c)
                    assert end_c_idx != -1
                    text_left = gen_text[end_c_idx + 1 :]
                    gen_text = gen_text[: end_c_idx + 1]
                    tts_text, tts_token_lens = self.prepare_tts_text(gen_text)
                    new_segment_gen = True
                    logger.debug(
                        f"tts_text tokens {tts_token_lens} exceed {self.tts.streaming_text_reserved_len - shift_len}, starting a new segment generation"
                    )
                    break

            if tts_token_lens >= (chunk_idx + 1) * self.tts.streaming_text_chunk_size:
                # do prefill and generate
                if chunk_idx == 0:
                    begin = 0
                    end = (chunk_idx + 1) * self.tts.streaming_text_chunk_size + tts_start_token_len
                else:
                    begin = chunk_idx * self.tts.streaming_text_chunk_size + tts_start_token_len
                    end = min(
                        (chunk_idx + 1) * self.tts.streaming_text_chunk_size + tts_start_token_len,
                        condition_length - 1,
                    )

                tts_input_ids = self.tts_processor.text_tokenizer(
                    tts_text, return_tensors="pt", add_special_tokens=False
                )["input_ids"].cuda()
                text_input_ids = tts_input_ids[:, begin:end]
                streaming_tts_text_mask = self._build_streaming_mask(tts_token_lens).to(device=self.tts.device)
                position_ids = torch.arange(begin, end, dtype=torch.long, device=self.tts.device).unsqueeze(0)

                past_key_values = self.tts.prefill_text(
                    input_ids=text_input_ids,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    lm_spk_emb_last_hidden_states=spk_embeds if chunk_idx == 0 else None,
                )
                outputs = self.tts.generate(
                    input_ids=audio_input_ids,
                    past_key_values=past_key_values,
                    streaming_tts_text_mask=streaming_tts_text_mask,
                    max_new_token=output_chunk_size,
                    force_no_stop=force_no_stop,
                    temperature=torch.tensor([0.1, 0.3, 0.1, 0.3], dtype=torch.float, device=self.tts.device),
                    eos_token=torch.tensor([625], dtype=torch.long, device=self.tts.device),
                    logits_warpers=logits_warpers,
                    logits_processors=logits_processors,
                )
                audio_input_ids = (
                    outputs.audio_input_ids
                )  # [1,seq_len,4] seq_len=tts.streaming_text_reserved_len + 3 + len(new_ids)
                past_key_values = outputs.past_key_values
                chunk_idx += 1

                mel_spec = self.tts.decode_to_mel_specs(outputs.new_ids[:, max(new_ids_len - 4, 0) :, :])
                new_ids_len = outputs.new_ids.shape[1]  # [1, seq_len, 4]

                wav_np, sr = self.decode_mel_to_audio(mel_spec)  # [1,100,50] -> [50*256]

                if enable_regenerate:
                    if prev_wav is not None:
                        check_wav_np = wav_np[2048:].cpu().numpy()  # 2*4*256(hop)
                        check_mel = mel_spec[0, :, 8:].cpu().numpy()  # 2*4
                    else:
                        check_wav_np = wav_np.cpu().numpy()
                        check_mel = mel_spec[0].cpu().numpy()
                if enable_regenerate and voice_checker.is_bad(check_wav_np, check_mel, chunk_size=2560):
                    voice_checker.reset()
                    # regenerate
                    N = output_chunk_size if prev_wav is None else output_chunk_size * 2
                    past_kv = []
                    for i in range(len(past_key_values)):
                        past_kv.append(
                            (
                                past_key_values[i][0][:, :, :-N, :],  # .clone(),
                                past_key_values[i][1][:, :, :-N, :],  # .clone(),
                            )
                        )
                    outputs = self.tts.generate(
                        input_ids=audio_input_ids[:, :-N, :],
                        past_key_values=past_kv,
                        streaming_tts_text_mask=streaming_tts_text_mask,
                        max_new_token=N,
                        force_no_stop=force_no_stop,
                        temperature=torch.tensor([0.1, 0.3, 0.1, 0.3], dtype=torch.float, device=self.tts.device),
                        eos_token=torch.tensor([625], dtype=torch.long, device=self.tts.device),
                        logits_warpers=logits_warpers,
                        logits_processors=logits_processors,
                    )
                    audio_input_ids = outputs.audio_input_ids
                    past_key_values = outputs.past_key_values

                    new_ids_len -= N
                    mel_spec = self.tts.decode_to_mel_specs(outputs.new_ids[:, new_ids_len:, :])
                    new_ids_len = outputs.new_ids.shape[1]  # [1, seq_len, 4]
                    wav_np, sr = self.decode_mel_to_audio(mel_spec)

                    if prev_wav is not None:
                        wav_y = wav_np[: len(prev_wav)]
                        prev_wav = wav_np[len(prev_wav) :]
                        cur_text = gen_text_raw[prev_text_len:]
                        prev_text_len = len(gen_text_raw)
                        yield OmniOutput(text=cur_text, audio_wav=wav_y, sampling_rate=sr)

                    else:
                        prev_wav = wav_np
                else:
                    # smooth wav
                    if prev_wav is not None:
                        wav_np, prev_wav = self._linear_overlap_add2_wav(
                            [prev_wav, wav_np], overlap=512 * 4
                        )  # tts_hop256*2
                        cur_text = gen_text_raw[prev_text_len:]
                        prev_text_len = len(gen_text_raw)
                        yield OmniOutput(text=cur_text, audio_wav=wav_np, sampling_rate=sr)

                    else:
                        prev_wav = wav_np

                if outputs.finished:
                    logger.debug("Generation finished.")
                    eos_lab = True
                    break

        if not eos_lab and tts_text:
            logger.debug("eos_lab False, Generation continue.")

            if chunk_idx == 0:
                begin = 0
            else:
                begin = chunk_idx * self.tts.streaming_text_chunk_size + tts_start_token_len
            end = tts_token_lens + tts_start_token_len + 1  # 1 for [Etts]
            if end > begin:
                tts_input_ids = self.tts_processor.text_tokenizer(
                    tts_text, return_tensors="pt", add_special_tokens=False
                )["input_ids"].cuda()
                text_input_ids = tts_input_ids[:, begin:end]
                streaming_tts_text_mask = self._build_streaming_mask(tts_token_lens).to(device=self.tts.device)
                position_ids = torch.arange(begin, end, dtype=torch.long, device=self.tts.device).unsqueeze(0)

                past_key_values = self.tts.prefill_text(
                    input_ids=text_input_ids,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    lm_spk_emb_last_hidden_states=spk_embeds if chunk_idx == 0 else None,
                )

            while True:
                # temp = [0.1, 0.3, 0.1, 0.3] if chunk_idx < 21 else [0.1] * self.tts.num_vq
                outputs = self.tts.generate(
                    input_ids=audio_input_ids,
                    past_key_values=past_key_values,
                    streaming_tts_text_mask=streaming_tts_text_mask,
                    max_new_token=output_chunk_size,
                    force_no_stop=force_no_stop,
                    # temperature=torch.tensor([0.1] * self.tts.num_vq, dtype=torch.float, device=self.tts.device),
                    temperature=torch.tensor([0.1, 0.3, 0.1, 0.3], dtype=torch.float, device=self.tts.device),
                    eos_token=torch.tensor([625], dtype=torch.long, device=self.tts.device),
                    logits_warpers=logits_warpers,
                    logits_processors=logits_processors,
                )
                audio_input_ids = outputs.audio_input_ids
                past_key_values = outputs.past_key_values
                chunk_idx += 1

                mel_spec = self.tts.decode_to_mel_specs(outputs.new_ids[:, max(new_ids_len - 4, 0) :, :])
                new_ids_len = outputs.new_ids.shape[1]  # [1, seq_len, 4]

                wav_np, sr = self.decode_mel_to_audio(mel_spec)

                if enable_regenerate:
                    if prev_wav is not None:
                        check_wav_np = wav_np[2048:].cpu().numpy()  # 2*4*256(hop)
                        check_mel = mel_spec[0, :, 8:].cpu().numpy()  # 2*4
                    else:
                        check_wav_np = wav_np.cpu().numpy()
                        check_mel = mel_spec[0].cpu().numpy()
                if enable_regenerate and voice_checker.is_bad(check_wav_np, check_mel, chunk_size=2560):
                    voice_checker.reset()
                    # regenerate
                    N = output_chunk_size if prev_wav is None else output_chunk_size * 2
                    past_kv = []
                    for i in range(len(past_key_values)):
                        past_kv.append(
                            (
                                past_key_values[i][0][:, :, :-N, :],  # .clone(),
                                past_key_values[i][1][:, :, :-N, :],  # .clone(),
                            )
                        )
                    outputs = self.tts.generate(
                        input_ids=audio_input_ids[:, :-N, :],
                        past_key_values=past_kv,
                        streaming_tts_text_mask=streaming_tts_text_mask,
                        max_new_token=N,
                        force_no_stop=force_no_stop,
                        temperature=torch.tensor([0.1, 0.3, 0.1, 0.3], dtype=torch.float, device=self.tts.device),
                        eos_token=torch.tensor([625], dtype=torch.long, device=self.tts.device),
                        logits_warpers=logits_warpers,
                        logits_processors=logits_processors,
                    )
                    audio_input_ids = outputs.audio_input_ids
                    past_key_values = outputs.past_key_values

                    new_ids_len -= N
                    mel_spec = self.tts.decode_to_mel_specs(outputs.new_ids[:, new_ids_len:, :])
                    new_ids_len = outputs.new_ids.shape[1]  # [1, seq_len, 4]
                    wav_np, sr = self.decode_mel_to_audio(mel_spec)

                    if prev_wav is not None:
                        wav_y = wav_np[: len(prev_wav)]
                        prev_wav = wav_np[len(prev_wav) :]
                        cur_text = gen_text_raw[prev_text_len:]
                        prev_text_len = len(gen_text_raw)
                        yield OmniOutput(text=cur_text, audio_wav=wav_y, sampling_rate=sr)
                    else:
                        prev_wav = wav_np
                else:
                    # smooth wav
                    if prev_wav is not None:
                        wav_np, prev_wav = self._linear_overlap_add2_wav(
                            [prev_wav, wav_np], overlap=512 * 4
                        )  # tts_hop256*2
                        cur_text = gen_text_raw[prev_text_len:]
                        prev_text_len = len(gen_text_raw)
                        yield OmniOutput(text=cur_text, audio_wav=wav_np, sampling_rate=sr)
                    else:
                        prev_wav = wav_np

                if outputs.finished:
                    logger.debug("Generation finished.")
                    break
                if outputs.new_ids.shape[1] > 2048:
                    stop = True
                    logger.debug("Generation length > 2048, stopped.")
                    break

        if prev_wav is not None:
            cur_text = gen_text_raw[prev_text_len:]
            yield OmniOutput(
                text=cur_text, audio_wav=prev_wav, sampling_rate=sr
            )  # yield last chunk wav without smooth

        if new_segment_gen and not stop:
            logger.debug(
                f"tts_text tokens {tts_token_lens} exceed {self.tts.streaming_text_reserved_len - shift_len}, start a new segment generation"
            )
            tid_len = 5  # self.tts.streaming_text_chunk_size
            prev_seg_text_ids = tts_input_ids[:, end - 1 - tid_len : end - 1]  # exclude last Etts
            aid_len = 50  # int(tid_len * new_ids_len / tts_token_lens)
            prev_seg_audio_ids = outputs.new_ids[:, -aid_len:, :]

            result = self._generate_mel_spec_audio_streaming(
                spk_bounds,
                streamer,
                output_chunk_size,
                spk_embeds,
                prev_seg_text_ids,
                text_left,
                prev_seg_audio_ids,
                enable_regenerate=enable_regenerate,
                tts_config={"top_p": 0.7, "top_k": 20, "repetition_penalty": 1.0},
                force_no_stop=force_no_stop,
            )
            for res in result:
                yield res

    def decode_mel_to_audio(self, mel_spec, output_path=""):
        with torch.inference_mode():
            wav_numpy = self.vocos.decode(mel_spec.float()).cpu().squeeze()
            sr = 24000
        if output_path:
            sf.write(output_path, wav_numpy.numpy(), samplerate=sr)
            logger.info(f"Audio saved to {output_path}")
        return wav_numpy, sr


def get_cache_usable_length(past_key_value: Cache, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
    """Given the sequence length of the new inputs, returns the usable length of the cache."""
    # Cache without size limit -> all cache is usable
    # Cache with size limit -> if the length cache plus the length of the new inputs is larger the maximum cache
    #   length, we will need to evict part of the cache (and thus not all cache is usable)
    max_length = past_key_value.get_max_cache_shape()
    previous_seq_length = past_key_value.get_seq_length(layer_idx)
    if max_length is not None and previous_seq_length + new_seq_length > max_length:
        return max_length - new_seq_length
    return previous_seq_length


def whisper_eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: Optional[float] = None,
    dropout: float = 0.0,
    head_mask: Optional[torch.Tensor] = None,
    **kwargs,
):
    if scaling is None:
        scaling = query.size(-1) ** -0.5

    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None and attention_mask.ndim == 4:
        attn_weights = attn_weights + attention_mask[:, :, :, : key.shape[-2]]

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    if head_mask is not None:
        attn_weights = attn_weights * head_mask.view(1, -1, 1, 1)

    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

# Copied from transformers.models.whisper.modeling_whisper.WhisperAttention and support past_key_value
class MiniCPMWhisperAttention(WhisperAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        cache_position: Optional[torch.Tensor] = None,
        # TODO: we need a refactor so that the different attention modules can get their specific kwargs
        # ATM, we have mixed things encoder, decoder, and encoder-decoder attn
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        # determine input shapes
        bsz, tgt_len = hidden_states.shape[:-1]
        q_input_shape = (bsz, tgt_len, -1, self.head_dim)

        # Scaling is susceptible to floating point arithmetics' inprecisions
        # which can lead to different results (this is dependent from model
        # to model, e.g. whisper is one such case). We therefore keep the
        # original order of scaling to follow the original implementation
        # and enforce no scaling (1.0) in the attention call below.
        query_states = self.q_proj(hidden_states) * self.scaling
        query_states = query_states.view(*q_input_shape)
        query_states = query_states.transpose(1, 2).contiguous()

        if past_key_values is not None:
            is_updated = past_key_values.is_updated.get(self.layer_idx)
            if is_cross_attention:
                # after the first generated id, we can subsequently re-use all key/value_states from cache
                past_key_values.is_updated[self.layer_idx] = True
                past_key_values = past_key_values.cross_attention_cache
            else:
                past_key_values = past_key_values.self_attention_cache

        # use key_value_states if cross attention
        current_states = key_value_states if key_value_states is not None else hidden_states
        if is_cross_attention and past_key_values and is_updated:
            # reuse k,v, cross_attentions
            key_states = past_key_values.key_cache[self.layer_idx]
            value_states = past_key_values.value_cache[self.layer_idx]
        else:
            key_states = self.k_proj(current_states).view(bsz, -1, self.num_heads, self.head_dim)
            value_states = self.v_proj(current_states).view(bsz, -1, self.num_heads, self.head_dim)
            key_states = key_states.transpose(1, 2).contiguous()
            value_states = value_states.transpose(1, 2).contiguous()
            if past_key_values is not None:
                # save all key/value_states to cache to be re-used for fast auto-regressive generation
                cache_position = cache_position if not is_cross_attention else None
                key_states, value_states = past_key_values.update(
                    key_states, value_states, self.layer_idx, {"cache_position": cache_position}
                )

        attention_interface: Callable = whisper_eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.dropout,
            scaling=1.0,
            output_attentions=output_attentions,
            head_mask=layer_head_mask,
            **kwargs,
        )

        attn_output = attn_output.reshape(bsz, tgt_len, -1).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights, past_key_values


# Copied from transformers.models.whisper.modeling_whisper.WhisperEncoderLayer and add use_cache for streaming inference
class MiniCPMWhisperEncoderLayer(WhisperEncoderLayer):
    def __init__(self, config: MiniCPMWhisperConfig, layer_idx: int = None):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = MiniCPMWhisperAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            config=config,
            layer_idx=layer_idx,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
        past_key_values: Optional[EncoderDecoderCache] = None,
        use_cache: Optional[bool] = False,
    ) -> torch.Tensor:
        r"""
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, seq_len, embed_dim)`):
                Hidden states to be fed into the encoder layer.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, 1, tgt_len, src_len)`):
                Attention mask where padding elements are indicated by large negative values.
            layer_head_mask (`torch.FloatTensor` of shape `(encoder_attention_heads,)`):
                Mask to nullify selected heads of the attention modules.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attention weights.
            past_key_values (`EncoderDecoderCache`, *optional*):
                Past key-value pairs used for incremental decoding.
            use_cache (`bool`, *optional*):
                Whether or not to return updated `past_key_values` for caching.

        Returns:
            A tuple of shape `(hidden_states, optional(attn_weights), optional(past_key_values))`.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights, past_key_values = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            past_key_values=past_key_values,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        if use_cache:
            outputs += (past_key_values,)

        return outputs


# Copied from from transformers.models.whisper.modeling_whisper.WhisperEncoder and add use_cache for streaming inference
class MiniCPMWhisperEncoder(WhisperEncoder):
    def __init__(self, config: MiniCPMWhisperConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [MiniCPMWhisperEncoderLayer(config, layer_idx=i) for i in range(config.encoder_layers)]
        )

    def forward(
        self,
        input_features,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        past_key_values: Optional[EncoderDecoderCache] = None,
        use_cache: Optional[bool] = None,
    ):
        r"""
        Forward pass of the Whisper encoder.

        Args:
            input_features (`torch.FloatTensor` of shape `(batch_size, feature_size, sequence_length)`):
                Float values of log-mel features extracted from the raw audio waveform. Typically generated
                by a feature extractor (e.g., `WhisperFeatureExtractor`) that processes `.flac` or `.wav`
                files into padded 2D mel spectrogram frames. These features are projected via convolution layers
                (`conv1` and `conv2`) and then transformed into embeddings for the encoder.

            attention_mask (`torch.Tensor`, *optional*):
                Not used by Whisper for masking `input_features`, but included for API compatibility with
                other models. If provided, it is simply ignored within the model. By default, Whisper
                effectively ignores silence in the input log-mel spectrogram.

            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected attention heads. The elements should be either 1 or 0, where:
                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked** (i.e., the attention head is dropped).

            output_attentions (`bool`, *optional*):
                Whether or not to return the attention tensors of all encoder layers. If set to `True`, the
                returned tuple (or `BaseModelOutputWithPast`) will contain an additional element with
                attention weights for each encoder layer.

            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. If set to `True`, the returned
                tuple (or `BaseModelOutputWithPast`) will contain a tuple of hidden states, including the
                initial embedding output as well as the outputs of each layer.

            return_dict (`bool`, *optional*):
                Whether or not to return a `BaseModelOutputWithPast` (a subclass of `ModelOutput`) instead
                of a plain tuple. If set to `True`, the output will be a `BaseModelOutputWithPast` object,
                otherwise it will be a tuple.

            past_key_values (`EncoderDecoderCache`, *optional*):
                When using caching for faster inference, this is an object that stores the key-value pairs
                for attention states. If provided, the model will append new states to the existing cache
                and return the updated cache. This speeds up sequential decoding or chunked inference.

                - If `past_key_values` is `None`, no past states are used or returned.
                - If `past_key_values` is not `None` and `use_cache=True`, the model will use the provided
                cache and return the updated cache (as `next_encoder_cache`).

            use_cache (`bool`, *optional*):
                Whether or not the model should use caching (`past_key_values`) to speed up processing
                during inference. When set to `True`, the model will:
                - Inspect and use `past_key_values` if provided.
                - Return updated `past_key_values` (under the name `next_encoder_cache` in
                    `BaseModelOutputWithPast`).

        Returns:
            `BaseModelOutputWithPast` or `tuple` (depending on `return_dict`):
                If `return_dict=True`, a `BaseModelOutputWithPast` is returned, which contains:
                - **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                The output of the final encoder layer.
                - **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned if `output_hidden_states=True`):
                Hidden states of the model at each layer (including the initial projection).
                - **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned if `output_attentions=True`):
                Attention weights from each encoder layer.
                - **past_key_values** (an object of type `EncoderDecoderCache` or `None`, *optional*):
                Updated cache of key-value pairs if `use_cache=True`.

                If `return_dict=False`, a tuple is returned, where the format is:
                `(last_hidden_state, hidden_states, attentions)`, with `hidden_states` and `attentions`
                only present if their respective `output_*` arguments are set to `True`.

        Example:
            >>> from transformers import AutoFeatureExtractor, MiniCPMWhisperConfig, WhisperForConditionalGeneration
            >>> import torch

            >>> # Load a feature extractor and a Whisper model
            >>> feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-tiny.en")
            >>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

            >>> # Assume you have audio (list of floats or numpy array) loaded from a file
            >>> # Then extract the mel features:
            >>> input_features = feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features

            >>> # Forward pass
            >>> outputs = model.encoder(
            ...     input_features=input_features,
            ...     output_hidden_states=True,
            ...     output_attentions=True,
            ...     use_cache=True
            ... )

            >>> # Retrieve the last hidden state
            >>> last_hidden_state = outputs.last_hidden_state
            >>> print(last_hidden_state.shape)
            torch.Size([batch_size, seq_length, hidden_size])

            >>> # Retrieve the intermediate hidden states if output_hidden_states=True
            >>> all_encoder_hidden_states = outputs.hidden_states

            >>> # Retrieve attention weights if output_attentions=True
            >>> all_encoder_attentions = outputs.attentions

            >>> # Retrieve updated past key values if use_cache=True
            >>> encoder_cache = outputs.past_key_values
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Ignore copy
        input_features = input_features.to(dtype=self.conv1.weight.dtype, device=self.conv1.weight.device)

        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)

        embed_pos = self.embed_positions.weight
        past_key_values_length = 0
        if use_cache:
            if past_key_values is None:
                past_key_values = EncoderDecoderCache(DynamicCache(), DynamicCache())
            elif isinstance(past_key_values, list):
                past_key_values = EncoderDecoderCache(DynamicCache.from_legacy_cache(past_key_values), DynamicCache())
            elif isinstance(past_key_values, DynamicCache):
                past_key_values = EncoderDecoderCache(past_key_values, DynamicCache())
            else:
                pass
            past_key_values_length = get_cache_usable_length(
                past_key_values.self_attention_cache, inputs_embeds.shape[1]
            )
            if inputs_embeds.shape[1] + past_key_values_length > embed_pos.shape[0]:
                logger.warning("seems the audio is longer than 30s. repeating the last part of the audio")
                embed_pos_front = embed_pos[past_key_values_length:, :]
                embed_pos = torch.cat(
                    (
                        embed_pos_front,
                        torch.repeat_interleave(
                            embed_pos[-1, :].unsqueeze(0),
                            inputs_embeds.shape[1] - embed_pos.shape[0] + past_key_values_length,
                            dim=0,
                        ),
                    )
                )
            else:
                embed_pos = embed_pos[past_key_values_length : inputs_embeds.shape[1] + past_key_values_length, :]
        else:
            embed_pos = embed_pos[: inputs_embeds.shape[1], :]

        hidden_states = inputs_embeds + embed_pos
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (len(self.layers)), (
                f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
            )

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            # Ignore copy
            if to_drop:
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        encoder_layer.__call__,
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                        output_attentions,
                        past_key_values,
                        use_cache,
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                        past_key_values=past_key_values,
                        use_cache=use_cache,
                    )

                hidden_states = layer_outputs[0]

            if use_cache:
                next_encoder_cache = layer_outputs[2 if output_attentions else 1]
            else:
                next_encoder_cache = None

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
            past_key_values=next_encoder_cache,
        )


# Borrowed from `https://github.com/2noise/ChatTTS/blob/main/ChatTTS/model/dvae.py`
class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        kernel: int,
        dilation: int,
        layer_scale_init_value: float = 1e-6,
    ):
        # ConvNeXt Block copied from Vocos.
        super().__init__()
        self.dwconv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel,
            padding=dilation * (kernel // 2),
            dilation=dilation,
            groups=dim,
        )

        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.coef = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x: torch.Tensor, cond=None) -> torch.Tensor:
        residual = x

        y = self.dwconv(x)
        y.transpose_(1, 2)  # (B, C, T) -> (B, T, C)
        x = self.norm(y)
        del y
        y = self.pwconv1(x)
        del x
        x = self.act(y)
        del y
        y = self.pwconv2(x)
        del x
        if self.coef is not None:
            y *= self.coef
        y.transpose_(1, 2)  # (B, T, C) -> (B, C, T)

        x = y + residual
        del y

        return x


# Borrowed from `https://github.com/2noise/ChatTTS/blob/main/ChatTTS/model/dvae.py`
class GFSQ(nn.Module):
    def __init__(
        self,
        dim: int,
        levels: list[int],
        G: int,
        R: int,
        eps=1e-5,
        transpose=True,
    ):
        super(GFSQ, self).__init__()
        self.quantizer = GroupedResidualFSQ(
            dim=dim,
            levels=list(levels),
            num_quantizers=R,
            groups=G,
        )
        self.n_ind = math.prod(levels)
        self.eps = eps
        self.transpose = transpose
        self.G = G
        self.R = R

    def _embed(self, x: torch.Tensor):
        if self.transpose:
            x = x.transpose(1, 2)
        x = x.view(x.size(0), x.size(1), self.G, self.R).permute(2, 0, 1, 3)
        feat = self.quantizer.get_output_from_indices(x)
        return feat.transpose_(1, 2) if self.transpose else feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.transpose:
            x.transpose_(1, 2)
        _, ind = self.quantizer(x)
        ind = ind.permute(1, 2, 0, 3).contiguous()
        ind = ind.view(ind.size(0), ind.size(1), -1)
        return ind.transpose_(1, 2) if self.transpose else ind


# Borrowed from `https://github.com/2noise/ChatTTS/blob/main/ChatTTS/model/dvae.py`
class DVAEDecoder(nn.Module):
    def __init__(
        self,
        idim: int,
        odim: int,
        n_layer=12,
        bn_dim=64,
        hidden=256,
        kernel=7,
        dilation=2,
        up=False,
    ):
        super().__init__()
        self.up = up
        self.conv_in = nn.Sequential(
            nn.Conv1d(idim, bn_dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv1d(bn_dim, hidden, 3, 1, 1),
        )
        self.decoder_block = nn.ModuleList(
            [
                ConvNeXtBlock(
                    hidden,
                    hidden * 4,
                    kernel,
                    dilation,
                )
                for _ in range(n_layer)
            ]
        )
        self.conv_out = nn.Conv1d(hidden, odim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor, conditioning=None) -> torch.Tensor:
        # B, C, T
        y = self.conv_in(x)
        del x
        for f in self.decoder_block:
            y = f(y, conditioning)

        x = self.conv_out(y)
        del y
        return x


# Borrowed from `https://github.com/2noise/ChatTTS/blob/main/ChatTTS/model/dvae.py`
class DVAE(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

        coef = torch.rand(100)
        self.coef = nn.Parameter(coef.unsqueeze(0).unsqueeze_(2))

        self.downsample_conv = nn.Sequential(
            nn.Conv1d(100, 512, 3, 1, 1),
            nn.GELU(),
            nn.Conv1d(512, 512, 4, 2, 1),
            nn.GELU(),
        )

        # These hardcodings match our model and cannot be modified.
        self.encoder = DVAEDecoder(
            idim=512,
            odim=1024,
            hidden=256,
            n_layer=12,
            bn_dim=128,
        )

        self.decoder = DVAEDecoder(
            idim=512,
            odim=512,
            hidden=256,
            n_layer=12,
            bn_dim=128,
        )

        self.out_conv = nn.Conv1d(512, 100, 3, 1, 1, bias=False)

        self.vq_layer = GFSQ(
            dim=1024,
            levels=(5, 5, 5, 5),
            G=2,
            R=2,
        )

    @torch.inference_mode()
    def encode(self, mel: torch.Tensor) -> torch.Tensor:
        """Encode mel spectrogram to VQ indices."""
        mel = mel.clone()
        x = self.downsample_conv(torch.div(mel, self.coef.view(100, 1).expand(mel.shape), out=mel))
        x = x.unsqueeze(0)
        x = self.encoder(x)
        ind = self.vq_layer(x)
        del x
        return ind

    def decode(self, inp: torch.Tensor) -> torch.Tensor:
        """Decode VQ inputs (either indices or embeddings) back to spectrogram."""
        vq_feats = self.vq_layer._embed(inp)

        # reshape embeddings
        vq_feats = (
            vq_feats.view(
                (vq_feats.size(0), 2, vq_feats.size(1) // 2, vq_feats.size(2)),
            )
            .permute(0, 2, 3, 1)
            .flatten(2)
        )

        dec_out = self.decoder(x=vq_feats)
        dec_out = self.out_conv(dec_out)
        del vq_feats
        # re-apply coef
        return torch.mul(dec_out, self.coef, out=dec_out)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Standard forward = encode + decode → full reconstruction."""
        indices = self.encode(inp)
        recon = self.decode(indices)
        return recon


def apply_spk_emb(
    input_ids: torch.Tensor = None,
    spk_emb: torch.Tensor = None,
    input_embeds: torch.Tensor = None,
    spk_emb_token_id: int = 0,
    num_spk_embs: int = 1,
):
    """
    Replace consecutive `num_spk_embs` speaker embedding placeholders in input_embeds with pre-prepared speaker embeddings. This is an in-place replacement, no new tensor is created, so no value is returned.

    Args:
        input_ids (torch.Tensor): Input ID tensor, shape [batch_size, seq_len_max]
        spk_emb (torch.Tensor): Speaker embedding tensor, shape [batch_size, num_spk_emb, hidden_dim]
        input_embeds (torch.Tensor): Input embedding tensor, shape [batch_size, seq_len_max, hidden_dim]
        spk_emb_token_id (int): ID of the speaker embedding token
        num_spk_embs (int): Number of speaker embeddings

    Returns:
        None
    """

    batch_size = input_ids.shape[0]

    for idx in range(batch_size):
        input_ids_ = input_ids[idx]  # [seq_len_max]
        spk_emb_ = spk_emb[idx]  # [num_spk_emb]
        mask_ = input_ids_ == spk_emb_token_id  # [batch_size, seq_len_max]
        nonzero_position_idx = mask_.nonzero(as_tuple=False)  # [num_spk_emb, 1]
        assert nonzero_position_idx.shape[0] == num_spk_embs
        begin_idx = nonzero_position_idx.min()
        end_idx = nonzero_position_idx.max()
        input_embeds[idx, begin_idx : end_idx + 1, :] = spk_emb_

    return


def make_streaming_chunk_mask_generation(
    inputs_embeds: torch.Tensor,
    past_seen_tokens: int,
    streaming_tts_text_mask: torch.Tensor,
    streaming_reserved_length: int = 300,
    streaming_audio_chunk_size: int = 50,
    streaming_text_chunk_size: int = 10,
    num_spk_emb: int = 1,
    use_spk_emb: bool = True,
) -> torch.Tensor:
    """
    In streaming audio generation, determine which `text` positions the TTS model can attend to when generating each chunk of `audio` tokens.

    This function creates a mask that allows the model to attend to a specific chunk of text
    tokens when generating each chunk of audio tokens, enabling streaming TTS generation.

    Args:
        inputs_embeds (torch.Tensor): Input embeddings tensor.
        past_seen_tokens (int): Number of tokens already seen by the model.
        streaming_tts_text_mask (torch.Tensor): Mask for the text tokens.
        streaming_reserved_length (int, optional): Number of reserved tokens for streaming. Defaults to 300.
        streaming_chunk_length (int, optional): Length of each streaming chunk. Defaults to 50.
        streaming_text_chunk_size (int, optional): Size of each text chunk. Defaults to 7.

    Returns:
        torch.Tensor: Causal mask for streaming TTS generation, shape is [batch_size=1, 1, seq_len=1, past_seen_tokens+1]

    Raises:
        AssertionError: If the batch size is not 1 (only supports batch size of 1 for inference).
    """
    assert inputs_embeds.shape[0] == 1

    dtype = inputs_embeds.dtype
    device = inputs_embeds.device
    min_dtype = torch.finfo(dtype).min

    # Add `1` to the past seen tokens to account for new `tokens` during `generate`
    causal_mask = torch.full((1, past_seen_tokens + inputs_embeds.shape[1]), fill_value=0, dtype=dtype, device=device)

    # Calculate the start of invisible text tokens
    invisible_text_tokens_start = (
        min(
            math.ceil((past_seen_tokens - streaming_reserved_length) / streaming_audio_chunk_size)
            * streaming_text_chunk_size,
            streaming_reserved_length,
        )
        + 1
        + num_spk_emb * use_spk_emb
    )  # Add 1 for [Stts] and N for [spk_emb] tokens if `use_spk_emb` is True

    invisible_text_tokens_end = (
        streaming_reserved_length + 1 + num_spk_emb * use_spk_emb + 1
    )  # Add 1 for [Ptts] (aka `audio_bos_token_id`)

    # Set invisible text tokens to min_dtype (effectively -inf)
    causal_mask[0, invisible_text_tokens_start:invisible_text_tokens_end] = min_dtype

    # Mask padding positions in the text mask
    causal_mask[0, 0 : 1 + num_spk_emb * use_spk_emb + streaming_reserved_length + 1].masked_fill_(
        streaming_tts_text_mask == 0, min_dtype
    )

    # Add extra dimensions for batch and heads
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

    return causal_mask


# Borrowed from `https://github.com/2noise/ChatTTS/blob/main/ChatTTS/model/processors.py`
class CustomRepetitionPenaltyLogitsProcessorRepeat:
    def __init__(self, penalty: float, max_input_ids: int, past_window: int):
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(f"`penalty` has to be a strictly positive float, but is {penalty}")

        self.penalty = penalty
        self.max_input_ids = max_input_ids
        self.past_window = past_window

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.size(1) > self.past_window:
            input_ids = input_ids.narrow(1, -self.past_window, self.past_window)
        freq = F.one_hot(input_ids, scores.size(1)).sum(1)
        if freq.size(0) > self.max_input_ids:
            freq.narrow(0, self.max_input_ids, freq.size(0) - self.max_input_ids).zero_()
        alpha = torch.pow(self.penalty, freq)
        scores = scores.contiguous()
        inp = scores.multiply(alpha)
        oth = scores.divide(alpha)
        con = scores < 0
        out = torch.where(con, inp, oth)
        del inp, oth, scores, con, alpha
        return out


@dataclass
class ConditionalChatTTSGenerationOutput(ModelOutput):
    """
    Output class for ConditionalChatTTS generation.

    Args:
        new_ids (torch.LongTensor): Newly generated audio code sequence, shape (batch_size, sequence_length, num_vq).
        audio_input_ids (torch.LongTensor): Updated input IDs including condition and generated audio codes, shape (batch_size, full_sequence_length, num_vq).
        past_key_values (tuple[tuple[torch.FloatTensor]]): tuple containing pre-computed keys and values used for attention mechanism. Each element has shape (batch_size, num_heads, sequence_length, embed_size_per_head).
        finished (bool): Boolean indicating whether generation is complete.

    """

    new_ids: torch.LongTensor = None
    audio_input_ids: torch.LongTensor = None
    past_key_values: Optional[tuple[tuple[torch.FloatTensor]]] = None
    finished: bool = None


class MultiModalProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear1 = nn.Linear(in_features=in_dim, out_features=out_dim, bias=True)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=out_dim, out_features=out_dim, bias=True)

    def forward(self, audio_features):
        hidden_states = self.relu(self.linear1(audio_features))
        hidden_states = self.linear2(hidden_states)
        return hidden_states


def _prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: torch.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
    min_dtype: float,
    cache_position: torch.Tensor,
    batch_size: int,
):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

    Args:
        attention_mask (`torch.Tensor`):
            A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
        sequence_length (`int`):
            The sequence length being processed.
        target_length (`int`):
            The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
        dtype (`torch.dtype`):
            The dtype to use for the 4D attention mask.
        device (`torch.device`):
            The device to plcae the 4D attention mask on.
        min_dtype (`float`):
            The minimum value representable with the dtype `dtype`.
        cache_position (`torch.Tensor`):
            Indices depicting the position of the input sequence tokens in the sequence.
        batch_size (`torch.Tensor`):
            Batch size.
    """
    if attention_mask is not None and attention_mask.dim() == 4:
        # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
        causal_mask = attention_mask
    else:
        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )

    return causal_mask


class MiniCPMConditionalTTSTextDecoderLayer(LlamaDecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


@auto_docstring
class MiniCPMConditionalTTSTextPreTrainedModel(LlamaPreTrainedModel):
    config_class = MiniCPMConditionalTTSTextConfig

@auto_docstring
class MiniCPMConditionalTTSTextModel(LlamaModel):
    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # in transformers>=4.53.1, this is `create_causal_mask`, but it will be wrong in our case
        # so copy `_update_causal_mask` from `LlamaModel` which transformers=4.44.2
        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
        # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
        # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
        # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114

        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            min_dtype=min_dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask


class ConditionalChatTTS(PreTrainedModel):
    """A conditional text-to-speech model that can generate speech from text with speaker conditioning.

    This model extends PreTrainedModel to provide text-to-speech capabilities with:
    - LLM hidden state conditioning
    - Streaming generation

    The model uses a transformer architecture with LLM hidden states and can operate in both
    streaming and non-streaming modes for flexible deployment.

    The model process sequence in the following format:
    | text bos token | LLM embedding projected to tts embedding space | text tokens (fixed length, reserved for future tokens) | audio bos token | audio tokens (audio token length is not fixed)| audio eos token |

    The format is designed to support LLM-conditioned streaming audio generation.

    Usage:
    To support streaming generation, two global variables should be maintained outside of the model.
        1. `audio_input_ids`: stores *discrete* audio codes. It is a tensor with shape [1, sequence length+1, num_vq].
        2. `past_key_values`: stores the KV cache for both text tokens and audio codes. It is a list of tuples, each tuple contains two tensors with shape [1, num_attention_heads, sequence length, hidden_size // num_attention_heads]

    where `num_vq` is the number of audio codebooks, in default setting, it is `4`.

    1. Create an empty `past_key_values` with
    ```python
    initial_kv_cache_length = 1 + model.num_spk_embs + model.streaming_text_reserved_len # where `1` denotes the `bos` token
    dtype = model.emb_text.weight.dtype
    device = model.emb_text.weight.device
    past_key_values = [
        (
            torch.zeros(1, model.config.num_attention_heads, initial_kv_cache_length, model.config.hidden_size // model.config.num_attention_heads, dtype=dtype, device=device),
            torch.zeros(1, model.config.num_attention_heads, initial_kv_cache_length, model.config.hidden_size // model.config.num_attention_heads, dtype=dtype, device=device)
        )
        for _ in range(model.config.num_hidden_layers)
    ]

    2. At the same time, create an empty `audio_input_ids` with shape [1, sequence length, num_vq], `num_vq` denotes multiple layer audio codebooks. But here we also include text tokens in the sequence, but they will be zeros, and will not be used, just a placeholder.

    ```python
    initial_audio_input_ids_length = 1 + model.num_spk_embs + model.streaming_text_reserved_len + 1
    # [bos token, speaker embeddings, text tokens, audio bos token]
    audio_input_ids = torch.zeros(batch_size=1, initial_audio_input_ids_length, model.num_vq)
    ```

    2. Prefill some text tokens to TTS model (for example, 10 tokens) using `prefill_text` method.

    ```python
    outputs = llm.generate(**kwargs)
    llm_tokens = some_function_to_extract_llm_tokens(outputs)
    lm_spk_emb_last_hidden_states = some_function_to_extract_lm_spk_emb_last_hidden_states(outputs)
    tts_text_input_ids = tts_tokenizer.encode(llm_tokenizer.decode(llm_tokens))
    # here assume we are prefilling text token 0 to text token 9 (included), totally 10 tokens.
    begin = 0
    end = 9+1
    position_ids = torch.arange(begin, end, dtype=torch.long, device=device)

    past_key_values = model.prefill_text(
        input_ids=tts_text_input_ids,
        position_ids=position_ids,
        past_key_values=past_key_values,
        lm_spk_emb_last_hidden_states=lm_spk_emb_last_hidden_states,
    )
    ```

    3. Make a `streaming_tts_text_mask` to denote which position contains valid text tokens, similar to `attention_mask` in standard causal attention.

    ```python
    streaming_tts_text_mask = torch.zeros(model.streaming_reserved_length)
    streaming_tts_text_mask[0:end] = 1 # denotes these post
    ```

    3. Generate audio codes using `generate` method.

    ```python
    outputs = model.generate(
        input_ids=audio_input_ids,
        past_key_values=past_key_values,
        streaming_tts_text_mask=streaming_tts_text_mask,
        max_new_token=50,
    )

    # update past_key_values and input_ids
    past_key_values = outputs.past_key_values
    audio_input_ids = outputs.input_ids
    ```

    The `past_key_values` is extended by `max_new_token=50`, and `audio_input_ids` is also extended by `max_new_token=50` after `generate` calling.

    4. Notice that after prefilling `10` text tokens, the model can generate up to `50` audio tokens, if you want to generate more audio tokens, you need to prefill next `10` text tokens. And it is okay to only generate `25` audio tokens for faster initial response.

    5. Repeat steps `2,3,4` as needed in your streaming audio generation cases, but ensure usage complies with the following guidelines discussed above.
    """

    config_class = MiniCPMConditionalTTSConfig
    _no_split_modules = []

    def __init__(self, config: MiniCPMConditionalTTSConfig):
        super().__init__(config)

        self.use_speaker_embedding = config.use_speaker_embedding
        self.use_llm_hidden_state = config.use_llm_hidden_state
        self.num_spk_embs = config.num_spk_embs
        self.spk_emb_token_id = config.spk_emb_token_id

        self.use_text = config.use_text
        self.streaming = config.streaming
        self.streaming_text_chunk_size = config.streaming_text_chunk_size
        self.streaming_audio_chunk_size = config.streaming_audio_chunk_size
        self.streaming_text_reserved_len = config.streaming_text_reserved_len
        self.audio_bos_token_id = config.audio_bos_token_id
        self.num_mel_bins = config.num_mel_bins
        self.num_vq = config.num_vq
        self.num_audio_tokens = config.num_audio_tokens

        if self.config.use_mlp:
            self.projector = MultiModalProjector(config.llm_dim, config.hidden_size)
        else:
            self.projector = nn.Linear(config.llm_dim, config.hidden_size, bias=False)
        self.emb_code = nn.ModuleList(
            [nn.Embedding(config.num_audio_tokens, config.hidden_size) for _ in range(config.num_vq)]
        )
        self.emb_text = nn.Embedding(config.num_text_tokens, config.hidden_size)
        self.head_code = nn.ModuleList(
            [
                weight_norm(
                    nn.Linear(config.hidden_size, config.num_audio_tokens, bias=False),
                    name="weight",
                )
                for _ in range(config.num_vq)
            ]
        )
        dvae = DVAE()
        self.dvae = dvae

        model = MiniCPMConditionalTTSTextModel(config.tts_text_config)
        self.model = model

    @torch.inference_mode()
    def merge_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        lm_spk_emb_last_hidden_states: Optional[torch.Tensor] = None,
    ):
        """Merge `input_ids` and `lm_spk_emb_last_hidden_states` to `inputs_embeds`.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            lm_spk_emb_last_hidden_states (Optional[torch.Tensor], optional): Last hidden states of speaker embeddings from the language model. Defaults to None.

        Raises:
            NotImplementedError: If speaker embedding is not used and language model hidden states are not implemented.

        Returns:
            torch.Tensor: Prepared input embeddings for the model.
        """
        assert input_ids.shape[0] == 1

        # Embed input_ids to input_embeds
        inputs_embeds = self.emb_text(input_ids)

        # Inject speaker embedding to input_embeds if it exists
        if self.use_speaker_embedding:
            spk_emb_mask = input_ids == self.spk_emb_token_id
            if spk_emb_mask.any():
                assert lm_spk_emb_last_hidden_states is not None
                # Project spk emb to tts hidden size first, [batch_size, num_spk_emb, llm_dim] -> [batch_size, num_spk_emb, self.hidden_size]
                lm_spk_emb_last_hidden_states = lm_spk_emb_last_hidden_states.to(self.projector.linear1.weight.dtype)
                projected_spk_emb = self.projector(lm_spk_emb_last_hidden_states)
                projected_spk_emb = F.normalize(projected_spk_emb, p=2, dim=-1)
                apply_spk_emb(
                    input_ids=input_ids,
                    spk_emb=projected_spk_emb,
                    input_embeds=inputs_embeds,
                    spk_emb_token_id=self.spk_emb_token_id,
                    num_spk_embs=self.num_spk_embs,
                )
        else:
            raise NotImplementedError

        return inputs_embeds

    @torch.inference_mode()
    def prefill_text(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.LongTensor,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]],
        lm_spk_emb_last_hidden_states: Optional[torch.Tensor] = None,
    ):
        """Prefill a chunk of new text tokens in streaming setting.
        Specifically speaking, update `past_key_values` using new text tokens, then the model will read the new text tokens.

        Args:
            input_ids (Tensor): Tensor of shape [batch_size, seq_len]
            position_ids (LongTensor): Tensor of shape [batch_size, seq_len]
            past_key_values (List[tuple[Tensor]]): KV Cache of all layers, each layer is a tuple (Tensor, Tensor) denoting keys and values. Each tensor is of seq_len = `self.streaming_text_reserved_len`. `past_key_values` will be updated.
            lm_spk_emb_last_hidden_states (Tensor, optional): Tensor of shape [batch_size, num_spk_emb, llm_dim]. Defaults to None.
            lm_last_hidden_states (Tensor, optional): _description_. Defaults to None.

        Note that all `batch_size` should be `1`.
        """
        assert input_ids.shape[0] == 1
        assert past_key_values is not None

        # Merge text and LLM embeddings
        inputs_embeds = self.merge_inputs_embeds(
            input_ids=input_ids,
            lm_spk_emb_last_hidden_states=lm_spk_emb_last_hidden_states,
        )

        # Clone KV Cache
        past_key_values_for_prefill = []
        for i in range(len(past_key_values)):
            past_key_values_for_prefill.append(
                (
                    past_key_values[i][0][:, :, : position_ids[:, 0], :].clone(),
                    past_key_values[i][1][:, :, : position_ids[:, 0], :].clone(),
                )
            )

        # Model forward
        outputs_prefill: BaseModelOutputWithPast = self.model(
            attention_mask=None,  # because for text, it is standard causal attention mask, do nothing
            position_ids=position_ids,  # position_ids denotes the position of new text tokens in the sequence
            past_key_values=DynamicCache.from_legacy_cache(
                past_key_values_for_prefill
            ),  # `past_key_values` will be updated by the model
            inputs_embeds=inputs_embeds,  # contains text and language model embedding
            use_cache=True,
            output_attentions=False,
            cache_position=position_ids,  # which new positions will use this cache, basically the same as position_ids
        )

        # Get model updated KV Cache
        past_key_values_for_prefill_updated = outputs_prefill.past_key_values.to_legacy_cache()

        # Update generated KV Cache to input `past_key_values`
        for layer_idx in range(len(past_key_values)):
            # Update keys
            past_key_values[layer_idx][0][:, :, position_ids[:, 0] : position_ids[:, -1] + 1, :] = (
                past_key_values_for_prefill_updated[layer_idx][0][
                    :, :, position_ids[:, 0] : position_ids[:, -1] + 1
                ].clone()
            )
            # Update values
            past_key_values[layer_idx][1][:, :, position_ids[:, 0] : position_ids[:, -1] + 1, :] = (
                past_key_values_for_prefill_updated[layer_idx][1][
                    :, :, position_ids[:, 0] : position_ids[:, -1] + 1
                ].clone()
            )

        # TODO: del past_key_values_for_prefill_updated recursively
        # TODO: del outputs_prefill recursively

        return past_key_values

    @torch.inference_mode()
    def prefill_audio_ids(
        self,
        input_ids: torch.Tensor,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]],
        streaming_tts_text_mask=None,
        add_audio_bos: bool = True,
    ):
        """Prefill a chunk of audio ids to the model. Used in sliding-window long audio generation.
        Specifically, prefill many audio ids (typically from last window) to the model in the new window.

        Args:
            input_ids (torch.Tensor): (1, seq_len, num_vq) Audio input token ids.
            past_key_values (List[tuple[torch.Tensor, torch.Tensor]]): Past key values for attention mechanism.
        """
        assert input_ids.shape[0] == 1
        assert past_key_values is not None

        code_emb = [self.emb_code[i](input_ids[:, :, i]) for i in range(self.num_vq)]
        inputs_embeds = torch.stack(code_emb, 3).sum(3)  # [1,seq_len,768]
        input_len = input_ids.shape[1]

        if add_audio_bos:
            narrowed_input_ids = torch.tensor([[self.audio_bos_token_id]], dtype=torch.long, device=self.device)
            bos_inputs_embeds = self.emb_text(narrowed_input_ids)
            inputs_embeds = torch.cat([bos_inputs_embeds, inputs_embeds], dim=1)
            input_len += 1

        past_key_values_length = past_key_values[0][0].shape[2]
        position_ids = torch.arange(
            past_key_values_length, past_key_values_length + input_len, dtype=torch.long, device=self.device
        ).unsqueeze(0)

        cache_position = position_ids.clone()
        causal_mask = make_streaming_chunk_mask_generation(
            inputs_embeds=inputs_embeds,
            past_seen_tokens=past_key_values[0][0].shape[2],
            streaming_tts_text_mask=streaming_tts_text_mask,
            streaming_reserved_length=self.streaming_text_reserved_len,
            streaming_text_chunk_size=self.streaming_text_chunk_size,
        )  # [1, 1, 1, past_key_values_length + input_len]

        # Model forward
        outputs: BaseModelOutputWithPast = self.model(
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=DynamicCache.from_legacy_cache(past_key_values),
            inputs_embeds=inputs_embeds,
            use_cache=True,
            output_attentions=False,
            cache_position=cache_position,
        )
        past_key_values = outputs.past_key_values.to_legacy_cache()
        return past_key_values

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]],
        temperature: torch.Tensor,
        eos_token: Union[int, torch.Tensor],
        streaming_tts_text_mask=None,
        force_no_stop=False,
        min_new_token=10,
        max_new_token=50,
        logits_warpers: list[LogitsProcessor] = [],
        logits_processors: list[CustomRepetitionPenaltyLogitsProcessorRepeat] = [],
        show_tqdm=False,
    ):
        """Generate audio codes in streaming setting or non-streaming setting.
        Specifically speaking, generate audio codes when not all text tokens are prefilled.

        Always pass a valid `past_key_values` to the method. The method does not do `prefill` by itself. It relies on `prefill_text` method to provide valid `past_key_values`. Please refer to docstring of this class for more details.

        In this method, we borrowed a lot of codes from `https://github.com/2noise/ChatTTS/blob/main/ChatTTS/model/gpt.py`.

        Args:
            input_ids (torch.Tensor): Input token ids.
            past_key_values (List[tuple[torch.Tensor, torch.Tensor]]): Past key values for attention mechanism.
            temperature (torch.Tensor): Temperature for sampling.
            eos_token (Union[int, torch.Tensor]): End of sequence token.
            streaming_tts_text_mask (Optional[torch.Tensor], optional): Mask for streaming TTS text. Defaults to None.
            max_new_token (int, optional): Maximum number of new tokens to generate. Defaults to 50.
            logits_warpers (List[LogitsWarper], optional): List of logits warpers. Defaults to [].
            logits_processors (List[CustomRepetitionPenaltyLogitsProcessorRepeat], optional): List of logits processors. Defaults to [].
            show_tqdm (bool, optional): Whether to show progress bar. Defaults to True.

        Returns:
            GenerationOutputs: Generation outputs.
        """

        # We only support batch size `1` for now
        assert input_ids.shape[0] == 1
        assert past_key_values is not None

        # fix: this should not be `input_ids.shape[1]`
        # start_idx = input_ids.shape[1]
        start_idx = 1 + self.num_spk_embs * self.use_speaker_embedding + self.streaming_text_reserved_len + 1

        finish = torch.zeros(input_ids.shape[0], device=input_ids.device).bool()

        temperature = temperature.unsqueeze(0).expand(input_ids.shape[0], -1).contiguous().view(-1, 1)

        progress = input_ids.shape[1]

        # Pre-allocate input_ids, shape is [batch_size=1, max_possible_seq_len, self.num_vqs]
        input_ids_buf = torch.zeros(
            input_ids.shape[0],  # batch_size
            progress + max_new_token,  # max_possible_seq_len = input_ids.shape[1] + max_new_token
            input_ids.shape[2],  # self.num_vqs
            dtype=input_ids.dtype,
            device=input_ids.device,
        )

        # Copy existing `input_ids` to `input_ids_buf`
        input_ids_buf.narrow(1, 0, progress).copy_(input_ids)

        del input_ids
        input_ids = input_ids_buf.narrow(1, 0, progress)

        pbar: Optional[tqdm] = None
        if show_tqdm:
            pbar = tqdm(
                total=max_new_token,
                desc="code",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}(max) [{elapsed}, {rate_fmt}{postfix}]",
            )

        condition_length = 1 + self.num_spk_embs * self.use_speaker_embedding + self.streaming_text_reserved_len + 1

        for i in range(max_new_token):
            # Prepare generation inputs
            audio_bos = False

            # If this is the first audio token, the case is SPECIAL
            if progress == condition_length:
                audio_bos = True

            assert progress == (
                past_key_values[0][0].shape[2] + 1
            )  # If you are using according to the guidelines, this should be passed.

            if audio_bos:
                # Generate the first token, activate the model with `self.audio_bos_token_id`, the model will predict a new audio token. This is a special case because without the `audio bos token`, it is impossible to generate the first audio token in our streaming setting.
                narrowed_input_ids = torch.tensor([[self.audio_bos_token_id]], dtype=torch.long, device=self.device)
                inputs_embeds = self.emb_text(narrowed_input_ids)
                del narrowed_input_ids
            else:
                # Generate the following audio tokens, it is applicable to all other cases, including second and the following calling of `generate`.
                narrowed_input_ids = input_ids.narrow(dim=1, start=input_ids.shape[1] - 1, length=1)
                code_emb = [self.emb_code[i](narrowed_input_ids[:, :, i]) for i in range(self.num_vq)]
                inputs_embeds = torch.stack(code_emb, 3).sum(3)

            position_ids = torch.tensor(
                [past_key_values[0][0].shape[2]], dtype=torch.long, device=self.device
            ).unsqueeze(0)

            cache_position = position_ids.clone()

            # Make causal mask
            causal_mask = make_streaming_chunk_mask_generation(
                inputs_embeds=inputs_embeds,
                past_seen_tokens=past_key_values[0][0].shape[2],
                streaming_tts_text_mask=streaming_tts_text_mask,
                streaming_reserved_length=self.streaming_text_reserved_len,
                streaming_text_chunk_size=self.streaming_text_chunk_size,
            )

            # Model forward
            outputs: BaseModelOutputWithPast = self.model(
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=DynamicCache.from_legacy_cache(past_key_values),
                inputs_embeds=inputs_embeds,
                use_cache=True,
                output_attentions=False,
                cache_position=cache_position,
            )

            del position_ids
            del inputs_embeds
            del cache_position
            del causal_mask

            hidden_states = outputs.last_hidden_state
            past_key_values = outputs.past_key_values.to_legacy_cache()

            with P.cached():
                logits = torch.empty(
                    hidden_states.size(0),
                    hidden_states.size(1),
                    self.num_audio_tokens,
                    self.num_vq,
                    dtype=torch.float,
                    device=self.device,
                )
                for num_vq_iter in range(self.num_vq):
                    x: torch.Tensor = self.head_code[num_vq_iter](hidden_states)
                    logits[..., num_vq_iter] = x
                    del x

            del hidden_states

            # logits = logits[:, -1].float()
            logits = logits.narrow(1, -1, 1).squeeze_(1).float()

            # logits = rearrange(logits, "b c n -> (b n) c")
            logits = logits.permute(0, 2, 1)
            logits = logits.reshape(-1, logits.size(2))
            # logits_token = rearrange(input_ids[:, start_idx:], "b c n -> (b n) c")
            input_ids_sliced = input_ids.narrow(
                1,
                start_idx,
                input_ids.size(1) - start_idx,
            ).permute(0, 2, 1)
            logits_token = input_ids_sliced.reshape(
                input_ids_sliced.size(0) * input_ids_sliced.size(1),
                -1,
            ).to(self.device)
            del input_ids_sliced

            logits /= temperature

            if not audio_bos:
                for logitsProcessors in logits_processors:
                    logits = logitsProcessors(logits_token, logits)
            if not audio_bos:
                for logitsWarpers in logits_warpers:
                    logits = logitsWarpers(logits_token, logits)

            del logits_token

            if i < min_new_token:
                logits[:, eos_token] = -torch.inf

            if force_no_stop:
                logits[:, eos_token] = -torch.inf

            scores = F.softmax(logits, dim=-1)

            del logits
            idx_next = torch.multinomial(scores, num_samples=1)  # .to(finish.device)

            del scores

            # idx_next = rearrange(idx_next, "(b n) 1 -> b n", n=self.num_vq)
            idx_next = idx_next.view(-1, self.num_vq)
            finish_or = idx_next.eq(eos_token).any(1)
            finish.logical_or_(finish_or)

            del finish_or
            # Store new `token` into `input_ids_buf`
            input_ids_buf.narrow(1, progress, 1).copy_(idx_next.unsqueeze_(1))

            if i == 0 and finish.any():
                # raise Exception
                break

            del idx_next
            progress += 1
            input_ids = input_ids_buf.narrow(1, 0, progress)

            if finish.all():
                break

            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        if not finish.all():
            if show_tqdm:
                logger.info(f"incomplete result. hit max_new_token: {max_new_token}")

        del input_ids_buf

        if finish.all():
            # the last may contains eos token
            genrated_input_ids = input_ids[:, condition_length:-1, :]
        else:
            # there is no eos token
            genrated_input_ids = input_ids[:, condition_length:, :]

        return ConditionalChatTTSGenerationOutput(
            new_ids=genrated_input_ids,
            audio_input_ids=input_ids,  # for update purpose
            past_key_values=past_key_values,  # for update purpose
            finished=finish.all(),
        )

    @torch.inference_mode()
    def decode_to_mel_specs(
        self,
        result_list: list[torch.Tensor],
    ):
        """Decode discrete audio codes to mel spectrograms.

        Borrowed from `https://github.com/2noise/ChatTTS/blob/main/ChatTTS/core.py`

        Args:
            result_list (List[torch.Tensor]): Audio codes output from `generate`.

        Returns:
            torch.Tensor: Mel spectrograms.
        """

        decoder = self.dvae
        max_x_len = -1
        if len(result_list) == 0:
            return np.array([], dtype=np.float32)
        for result in result_list:
            if result.size(0) > max_x_len:
                max_x_len = result.size(0)
        batch_result = torch.zeros(
            (len(result_list), result_list[0].size(1), max_x_len),
            dtype=result_list[0].dtype,
            device=result_list[0].device,
        )
        for i in range(len(result_list)):
            src = result_list[i]
            batch_result[i].narrow(1, 0, src.size(0)).copy_(src.permute(1, 0))
            del src

        mel_specs = decoder.decode(batch_result)
        del batch_result
        return mel_specs


# Borrowed from `https://github.com/2noise/ChatTTS/blob/main/ChatTTS/model/processors.py`
def gen_logits(
    num_code: int,
    top_P=0.7,
    top_K=20,
    repetition_penalty=1.0,
):
    logits_warpers = []
    if top_P is not None:
        logits_warpers.append(TopPLogitsWarper(top_P, min_tokens_to_keep=3))
    if top_K is not None:
        logits_warpers.append(TopKLogitsWarper(top_K, min_tokens_to_keep=3))

    logits_processors = []
    if repetition_penalty is not None and repetition_penalty != 1:
        logits_processors.append(CustomRepetitionPenaltyLogitsProcessorRepeat(repetition_penalty, num_code, 16))

    return logits_warpers, logits_processors


def get_2d_sincos_pos_embed(embed_dim, image_size):
    """
    image_size: image_size or (image_height, image_width)
    return:
    pos_embed: [image_height, image_width, embed_dim]
    """
    if isinstance(image_size, int):
        grid_h_size, grid_w_size = image_size, image_size
    else:
        grid_h_size, grid_w_size = image_size[0], image_size[1]

    grid_h = np.arange(grid_h_size, dtype=np.float32)
    grid_w = np.arange(grid_w_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid_new(embed_dim // 2, grid[0])  # (H, W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid_new(embed_dim // 2, grid[1])  # (H, W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=-1)  # (H, W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid_new(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (H, W)
    out: (H, W, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    out = np.einsum("hw,d->hwd", pos, omega)  # (H, W, D/2), outer product

    emb_sin = np.sin(out)  # (H, W, D/2)
    emb_cos = np.cos(out)  # (H, W, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=-1)  # (H, W, D)
    return emb


class Resampler(nn.Module):
    """
    A 2D perceiver-resampler network with one cross attention layers by
       given learnable queries and 2d sincos pos_emb
    Outputs:
        A tensor with the shape of (batch_size, num_queries, embed_dim)
    """

    def __init__(
        self,
        num_queries,
        embed_dim,
        num_heads,
        kv_dim=None,
        adaptive=False,
        max_size=(70, 70),
    ):
        super().__init__()
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.adaptive = adaptive
        self.max_size = max_size

        self.query = nn.Parameter(torch.zeros(self.num_queries, embed_dim))

        if kv_dim is not None and kv_dim != embed_dim:
            self.kv_proj = nn.Linear(kv_dim, embed_dim, bias=False)
        else:
            self.kv_proj = nn.Identity()

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.attn = MultiheadAttention(embed_dim, num_heads)
        self.ln_q = norm_layer(embed_dim)
        self.ln_kv = norm_layer(embed_dim)

        self.ln_post = norm_layer(embed_dim)
        self.proj = nn.Parameter((embed_dim**-0.5) * torch.randn(embed_dim, embed_dim))

        self._set_2d_pos_cache(self.max_size)

    def _set_2d_pos_cache(self, max_size, device="cpu"):
        if is_deepspeed_zero3_enabled():
            device = "cuda"
        pos_embed = torch.from_numpy(get_2d_sincos_pos_embed(self.embed_dim, max_size)).float().to(device)
        self.register_buffer("pos_embed", pos_embed, persistent=False)

    def _adjust_pos_cache(self, tgt_sizes, device):
        max_h = torch.max(tgt_sizes[:, 0])
        max_w = torch.max(tgt_sizes[:, 1])
        if max_h > self.max_size[0] or max_w > self.max_size[1]:
            self.max_size = [max(max_h, self.max_size[0]), max(max_w, self.max_size[1])]
            self._set_2d_pos_cache(self.max_size, device)

    def forward(self, x, tgt_sizes=None):
        assert x.shape[0] == tgt_sizes.shape[0]
        bs = x.shape[0]

        device = x.device
        dtype = x.dtype

        patch_len = tgt_sizes[:, 0] * tgt_sizes[:, 1]

        self._adjust_pos_cache(tgt_sizes, device=device)

        max_patch_len = torch.max(patch_len)
        key_padding_mask = torch.zeros((bs, max_patch_len), dtype=torch.bool, device=device)

        pos_embed = []
        for i in range(bs):
            tgt_h, tgt_w = tgt_sizes[i]
            pos_embed.append(self.pos_embed[:tgt_h, :tgt_w, :].reshape((tgt_h * tgt_w, -1)).to(dtype))  # patches * D
            key_padding_mask[i, patch_len[i] :] = True

        pos_embed = torch.nn.utils.rnn.pad_sequence(pos_embed, batch_first=True, padding_value=0.0).permute(
            1, 0, 2
        )  # BLD => L * B * D

        x = self.kv_proj(x)  # B * L * D
        x = self.ln_kv(x).permute(1, 0, 2)  # L * B * D

        q = self.ln_q(self.query)  # Q * D

        out = self.attn(
            self._repeat(q, bs),  # Q * B * D
            x + pos_embed,  # L * B * D +  L * B * D
            x,
            key_padding_mask=key_padding_mask,
        )[0]
        #  out: Q * B * D
        x = out.permute(1, 0, 2)  # B * Q * D

        x = self.ln_post(x)
        x = x @ self.proj
        return x

    def _repeat(self, query, N: int):
        return query.unsqueeze(1).repeat(1, N, 1)


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        batch_first: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        factory_kwargs = {"device": device, "dtype": dtype}

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias, **factory_kwargs)
        self.k_proj = nn.Linear(self.kdim, self.embed_dim, bias=bias, **factory_kwargs)
        self.v_proj = nn.Linear(self.vdim, self.embed_dim, bias=bias, **factory_kwargs)
        self.o_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias, **factory_kwargs)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        tgt_len, bsz, _ = query.shape
        src_len, _, _ = key.shape

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        q = q.reshape(bsz, self.num_heads, tgt_len, self.head_dim)
        k = k.reshape(bsz, self.num_heads, src_len, self.head_dim)
        v = v.reshape(bsz, self.num_heads, src_len, self.head_dim)

        if key_padding_mask is not None:
            if key_padding_mask.dim() != 2 or key_padding_mask.shape != (bsz, src_len):
                raise ValueError(
                    f"Expected key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
                )
            # F.sdpa expects mask where True means "don't attend"
            # key_padding_mask is (bsz, src_len), needs to be broadcastable to (bsz, num_heads, tgt_len, src_len)
            # So we reshape to (bsz, 1, 1, src_len)
            if attn_mask is None:
                attn_mask = key_padding_mask.view(bsz, 1, 1, src_len)
            else:  # Float or bool mask
                # Assuming attn_mask is already broadcastable or has shape (tgt_len, src_len)
                if attn_mask.dim() == 2:
                    attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # to (1, 1, tgt_len, src_len)
                elif attn_mask.dim() == 3:  # (bsz, tgt_len, src_len)
                    attn_mask = attn_mask.unsqueeze(1)  # to (bsz, 1, tgt_len, src_len)

                # Combine masks. Assuming both are boolean, True means ignore.
                attn_mask = attn_mask.expand(-1, self.num_heads, -1, -1) | key_padding_mask.view(bsz, 1, 1, src_len)

        # F.scaled_dot_product_attention does not return attention weights.
        # If weights are needed, we have to compute attention manually.
        if need_weights:
            # Manual calculation to get attention weights
            scaling_factor = self.head_dim**-0.5
            attn_output_weights = torch.matmul(q * scaling_factor, k.transpose(-2, -1))

            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    attn_output_weights.masked_fill_(attn_mask, float("-inf"))
                else:
                    attn_output_weights += attn_mask

            if is_causal:
                # is_causal is a built-in option in F.sdpa, but here we apply it manually
                # This is a simplified causal mask, assuming query and key have same length
                causal_mask = torch.triu(torch.ones(tgt_len, src_len, device=q.device, dtype=torch.bool), diagonal=1)
                attn_output_weights.masked_fill_(causal_mask, float("-inf"))

            attn_output_weights = F.softmax(attn_output_weights, dim=-1)

            if self.training and self.dropout > 0.0:
                attn_output_weights = F.dropout(attn_output_weights, p=self.dropout)

            attn_output = torch.matmul(attn_output_weights, v)

            if average_attn_weights:
                attn_output_weights = attn_output_weights.sum(dim=1) / self.num_heads

        else:
            # Use the optimized F.scaled_dot_product_attention path
            attn_output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0, is_causal=is_causal
            )
            attn_output_weights = None

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, tgt_len, self.embed_dim)
        attn_output = self.o_proj(attn_output)

        if self.batch_first:
            # Reshape back to (bsz, tgt_len, embed_dim)
            return attn_output, attn_output_weights
        else:
            # Reshape back to (tgt_len, bsz, embed_dim)
            return attn_output.transpose(0, 1), attn_output_weights


_CHECKPOINT_FOR_DOC = "google/siglip-base-patch16-224"

SIGLIP_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/siglip-base-patch16-224",
    # See all SigLIP models at https://huggingface.co/models?filter=siglip
]


def _trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    # Values are generated by using a truncated uniform distribution and
    # then using the inverse CDF for the normal distribution.
    # Get upper and lower cdf values
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    # Uniformly fill tensor with values from [l, u], then translate to
    # [2l-1, 2u-1].
    tensor.uniform_(2 * l - 1, 2 * u - 1)

    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal
    if tensor.dtype in [torch.float16, torch.bfloat16]:
        # The `erfinv_` op is not (yet?) defined in float16+cpu, bfloat16+gpu
        og_dtype = tensor.dtype
        tensor = tensor.to(torch.float32)
        tensor.erfinv_()
        tensor = tensor.to(og_dtype)
    else:
        tensor.erfinv_()

    # Transform to proper mean, std
    tensor.mul_(std * math.sqrt(2.0))
    tensor.add_(mean)

    # Clamp to ensure it's in the proper range
    if tensor.dtype == torch.float16:
        # The `clamp_` op is not (yet?) defined in float16+cpu
        tensor = tensor.to(torch.float32)
        tensor.clamp_(min=a, max=b)
        tensor = tensor.to(torch.float16)
    else:
        tensor.clamp_(min=a, max=b)


def trunc_normal_tf_(
    tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0
) -> torch.Tensor:
    """Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \\leq \text{mean} \\leq b`.
    NOTE: this 'tf' variant behaves closer to Tensorflow / JAX impl where the
    bounds [a, b] are applied when sampling the normal distribution with mean=0, std=1.0
    and the result is subsquently scaled and shifted by the mean and std args.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    """
    with torch.no_grad():
        _trunc_normal_(tensor, 0, 1.0, a, b)
        tensor.mul_(std).add_(mean)


def variance_scaling_(tensor, scale=1.0, mode="fan_in", distribution="normal"):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == "fan_in":
        denom = fan_in
    elif mode == "fan_out":
        denom = fan_out
    elif mode == "fan_avg":
        denom = (fan_in + fan_out) / 2

    variance = scale / denom

    if distribution == "truncated_normal":
        # constant is stddev of standard normal truncated to (-2, 2)
        trunc_normal_tf_(tensor, std=math.sqrt(variance) / 0.87962566103423978)
    elif distribution == "normal":
        with torch.no_grad():
            tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        with torch.no_grad():
            tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode="fan_in", distribution="truncated_normal")


def default_flax_embed_init(tensor):
    variance_scaling_(tensor, mode="fan_in", distribution="normal")


@dataclass
class MiniCPMVisionModelOutput(SiglipVisionModelOutput):
    """
    Base class for vision model's outputs that also contains image embeddings of the pooling of the last hidden states.
    Args:
        image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The image embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    pass


class MiniCPMVisionEmbedding(nn.Module):
    def __init__(self, config: MiniCPMVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches_per_side = self.image_size // self.patch_size
        self.num_patches = self.num_patches_per_side**2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        patch_attention_mask: torch.BoolTensor,
        tgt_sizes: Optional[torch.IntTensor] = None,
    ) -> torch.Tensor:
        batch_size = pixel_values.size(0)

        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        max_im_h, max_im_w = pixel_values.size(2), pixel_values.size(3)
        max_nb_patches_h, max_nb_patches_w = max_im_h // self.patch_size, max_im_w // self.patch_size
        boundaries = torch.arange(1 / self.num_patches_per_side, 1.0, 1 / self.num_patches_per_side)
        position_ids = torch.full(
            size=(
                batch_size,
                max_nb_patches_h * max_nb_patches_w,
            ),
            fill_value=0,
        )

        for batch_idx, p_attn_mask in enumerate(patch_attention_mask):
            if tgt_sizes is not None:
                nb_patches_h = tgt_sizes[batch_idx][0]
                nb_patches_w = tgt_sizes[batch_idx][1]
            else:
                nb_patches_h = p_attn_mask[:, 0].sum()
                nb_patches_w = p_attn_mask[0].sum()

            fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / nb_patches_h)
            fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / nb_patches_w)

            bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
            bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)

            pos_ids = (bucket_coords_h[:, None] * self.num_patches_per_side + bucket_coords_w).flatten()
            position_ids[batch_idx][p_attn_mask.view(-1).cpu()] = pos_ids

        position_ids = position_ids.to(self.position_embedding.weight.device)

        embeddings = embeddings + self.position_embedding(position_ids)
        return embeddings


class MiniCPMVisionAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: MiniCPMVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        batch_size, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        k_v_seq_len = key_states.shape[-2]
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale

        if attn_weights.size() != (batch_size, self.num_heads, q_len, k_v_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, q_len, k_v_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (batch_size, 1, q_len, k_v_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(batch_size, 1, q_len, k_v_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class MiniCPMVisionFlashAttention2(MiniCPMVisionAttention):
    """
    Llama flash attention module. This module inherits from `LlamaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False  # Hack to make sure we don't use a causal mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # if past_key_value is not None:
        #     cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        #     key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                "The input hidden states seems to be silently casted in float32, this might be related to the fact"
                " you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
        )

        attn_output = attn_output.reshape(bsz, q_len, self.embed_dim).contiguous()
        attn_output = self.out_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights

    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.
        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """

        # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
        causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


class MiniCPMVisionMLP(SiglipMLP):
    pass


class MiniCPMVisionEncoderLayer(SiglipEncoderLayer):
    def __init__(self, config: MiniCPMVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.self_attn = (
            MiniCPMVisionAttention(config) if not self._use_flash_attention_2 else MiniCPMVisionFlashAttention2(config)
        )
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = MiniCPMVisionMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)


class MiniCPMVisionPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = MiniCPMVisionConfig
    base_model_prefix = "siglip"
    supports_gradient_checkpointing = True

    def _initialize_weights(self, module):
        """Initialize the weights"""

        if isinstance(module, MiniCPMVisionEmbedding):
            width = self.config.hidden_size
            nn.init.normal_(module.position_embedding.weight, std=1 / np.sqrt(width))
        elif isinstance(module, nn.Embedding):
            default_flax_embed_init(module.weight)
        elif isinstance(module, MiniCPMVisionAttention):
            nn.init.normal_(module.q_proj.weight)
            nn.init.normal_(module.k_proj.weight)
            nn.init.normal_(module.v_proj.weight)
            nn.init.normal_(module.out_proj.weight)
            nn.init.zeros_(module.q_proj.bias)
            nn.init.zeros_(module.k_proj.bias)
            nn.init.zeros_(module.v_proj.bias)
            nn.init.zeros_(module.out_proj.bias)
        elif isinstance(module, MiniCPMVisionMLP):
            nn.init.normal_(module.fc1.weight)
            nn.init.normal_(module.fc2.weight)
            nn.init.normal_(module.fc1.bias, std=1e-6)
            nn.init.normal_(module.fc2.bias, std=1e-6)
        elif isinstance(module, (nn.Linear, nn.Conv2d)):
            lecun_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


SIGLIP_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)
    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.
    Parameters:
        config ([`MiniCPMVisionConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


SIGLIP_VISION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`CLIPImageProcessor.__call__`] for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class MiniCPMVisionEncoder(SiglipEncoder):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`SiglipEncoderLayer`].
    Args:
        config: MiniCPMVisionConfig
    """

    def __init__(self, config: MiniCPMVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([MiniCPMVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    # Ignore copy
    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    output_attentions,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


@add_start_docstrings(
    """The vision model from SigLIP without any head or projection on top.""", SIGLIP_START_DOCSTRING
)
class MiniCPMVisionTransformer(MiniCPMVisionPreTrainedModel):
    config_class = MiniCPMVisionConfig
    main_input_name = "pixel_values"
    _supports_flash_attn_2 = True
    _no_split_modules = []

    def __init__(self, config: MiniCPMVisionConfig):
        super().__init__(config)
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = MiniCPMVisionEmbedding(config)
        self.encoder = MiniCPMVisionEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.embeddings.patch_embedding

    @add_start_docstrings_to_model_forward(SIGLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=MiniCPMVisionConfig)
    def forward(
        self,
        pixel_values,
        patch_attention_mask: Optional[torch.BoolTensor] = None,
        tgt_sizes: Optional[torch.IntTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = pixel_values.size(0)
        if patch_attention_mask is None:
            patch_attention_mask = torch.ones(
                size=(
                    batch_size,
                    pixel_values.size(2) // self.config.patch_size,
                    pixel_values.size(3) // self.config.patch_size,
                ),
                dtype=torch.bool,
                device=pixel_values.device,
            )

        hidden_states = self.embeddings(
            pixel_values=pixel_values, patch_attention_mask=patch_attention_mask, tgt_sizes=tgt_sizes
        )

        patch_attention_mask = patch_attention_mask.view(batch_size, -1)
        # The call to `_upad_input` in `_flash_attention_forward` is expensive
        # So when the `patch_attention_mask` is full of 1s (i.e. attending to the whole sequence),
        # avoiding passing the attention_mask, which is equivalent to attending to the full sequence
        if not torch.any(~patch_attention_mask):
            attention_mask = None
        else:
            attention_mask = (
                _prepare_4d_attention_mask(patch_attention_mask, hidden_states.dtype)
                if not self._use_flash_attention_2
                else patch_attention_mask
            )

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)

        if not return_dict:
            return (last_hidden_state, None) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=None,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


__all__ = ["MiniCPM_o_2_6ForConditionalGeneration", "MiniCPM_o_2_6TextModel", "MiniCPM_o_2_6PreTrainedModel", "MiniCPM_o_2_6Config"]
