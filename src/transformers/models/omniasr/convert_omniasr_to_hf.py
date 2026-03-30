# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
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

"""
Setup
```
pip install omnilingual-asr
pip install sentencepiece
pip install -e .

# - macOS (Apple Silicon)
brew install libsndfile

# - DGX
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install \
  torch==2.8.0+cu128 \
  torchvision==0.23.0+cu128 \
  torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install fairseq2 \
  --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/pt2.8.0/cu128
python -m pip install --upgrade huggingface_hub
```

See here for available models: https://github.com/facebookresearch/omnilingual-asr?tab=readme-ov-file#model-architectures

Example conversion:
```python
# -- CTC-variant v2
python src/transformers/models/omniasr/convert_omniasr_to_hf.py \
    --model_card omniASR_CTC_300M_v2 \
    --repo_id bezzam/omniasr-ctc-300m-v2

# -- LLM-variant v2
python src/transformers/models/omniasr/convert_omniasr_to_hf.py \
--model_card omniASR_LLM_300M_v2 \
--repo_id bezzam/omniasr-llm-300m-v2


## release v1
python src/transformers/models/omniasr/convert_omniasr_to_hf.py \
    --model_card omniASR_CTC_300M \
    --repo_id bezzam/omniasr-ctc-300m

python src/transformers/models/omniasr/convert_omniasr_to_hf.py \
    --model_card omniASR_W2V_300M \
    --repo_id bezzam/omniasr-w2v-300m
```

Original model checkpoints are saved under:  ~/.cache/fairseq2/assets/
"""

import argparse
import json
import os
import torch
import urllib.request

from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
from omnilingual_asr.models.wav2vec2_llama.config import Wav2Vec2LlamaConfig, ModelType, Wav2Vec2LlamaStreamingConfig
from fairseq2.models.llama import LLaMAConfig
from fairseq2.models.wav2vec2.asr.config import Wav2Vec2AsrConfig
from fairseq2.models.wav2vec2.config import Wav2Vec2Config
from fairseq2.models.hub import load_model
from fairseq2.runtime.dependency import get_dependency_resolver
from fairseq2.runtime.config_registry import get_config
from fairseq2.models.transformer.norm_order import TransformerNormOrder

from transformers import (
    LlamaConfig,
    OmniASRCTCConfig,
    OmniASRLLMConfig,
    OmniASREncoderConfig,
    OmniASRForCTC,
    OmniASRForConditionalGeneration,
    OmniASRModel,
    Wav2Vec2CTCTokenizer,
    ParakeetTokenizerFast,
    SeamlessM4TTokenizer,
    Wav2Vec2FeatureExtractor,
    OmniASRFeatureExtractor,
    LasrTokenizer,
    logging,
)
from transformers.models.omniasr.processing_omniasr import OmniASRProcessor
from transformers.tokenization_utils_sentencepiece import SentencePieceExtractor


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


# TODO change to state dict mapping like in newer models
encoder_convert_list = [
    # OmniASRFeatureEncoder
    ("encoder_frontend.feature_extractor.layers", "encoder.feature_extractor.conv_layers"),
    ("encoder_frontend.post_extract_layer_norm", "encoder.feature_extractor.layer_norm"),
    # OmniASRFeatureProjection
    ("encoder_frontend.model_dim_proj", "encoder.feature_projection.projection"),
    # OmniASREncoder
    ("encoder_frontend.pos_encoder.conv", "encoder.encoder.embed_positions.conv"),
    ("encoder.layers", "encoder.encoder.layers"),
    ("self_attn.output_proj", "self_attn.out_proj"),
    ("self_attn_layer_norm", "layer_norm"),
    ("ffn_layer_norm", "final_layer_norm"),
    ("inner_proj", "intermediate_dense"),
    ("output_proj", "output_dense"),
    ("encoder.layer_norm", "encoder.encoder.layer_norm"),
]


ctc_convert_list = [
    ("final_proj", "ctc_head"),
]

llm_convert_list = [
    ("final_proj", "language_model.lm_head"),
    ("encoder_proj", "multi_modal_projector"),
    # LLaMA decoder - order matters! More specific patterns first
    ("llama_decoder.layers", "language_model.model.layers"),
    ("self_attn.output_proj", "self_attn.o_proj"),
    ("ffn.gate_proj", "mlp.gate_proj"),
    ("ffn.inner_proj", "mlp.up_proj"),
    ("ffn.output_proj", "mlp.down_proj"),
    ("self_attn_layer_norm", "input_layernorm"),
    ("ffn_layer_norm", "post_attention_layernorm"),
    ("llama_decoder.layer_norm", "language_model.model.norm"),
    ("text_frontend", "language_model.model.embed_tokens"),
]


"""
LLM model also has:
(encoder_proj): Linear(input_dim=1024, output_dim=4096, bias=True)                                                                           
(text_frontend): StandardEmbedding(num_embeddings=10289, embed_dim=4096)
(llama_decoder): StandardTransformerLMDecoder(
(lang_embeddings): StandardEmbedding(num_embeddings=1694, embed_dim=4096)  
"""


def _convert_model(
    original_model,
    hf_model,
    encoder_convert_list,
    decoder_convert_list=None,
    verbose=False
):
    """
    ValueError: 1 extra keys found: {'lang_embeddings.weight'}
    """

    state_dict = original_model.state_dict()
    print("Number of keys in original model :", len(state_dict))
    print("Number of keys in HF model       : ", len(hf_model.state_dict()))

    # Convert encoder keys
    for k, v in list(state_dict.items()):
        new_key = k
        for old_layer_name, new_layer_name in encoder_convert_list:
            if "encoder." in k or "encoder_frontend." in k:
                if old_layer_name in new_key:
                    if verbose:
                        print("Converting key:", new_key, " to ", new_key.replace(old_layer_name, new_layer_name))
                    new_key = new_key.replace(old_layer_name, new_layer_name)
        state_dict[new_key] = state_dict.pop(k)

    # Convert decoder keys
    if decoder_convert_list is not None:
        for k, v in list(state_dict.items()):
            new_key = k
            for old_layer_name, new_layer_name in decoder_convert_list:
                if "encoder." in k or "encoder_frontend." in k:
                    continue
                if old_layer_name in new_key:
                    if verbose:
                        print("Converting key:", new_key, " to ", new_key.replace(old_layer_name, new_layer_name))
                    new_key = new_key.replace(old_layer_name, new_layer_name)
            state_dict[new_key] = state_dict.pop(k)
        
        # Rearrange Q/K projection weights for RoPE compatibility (interleaved -> half-split)
        # Based on convert_pe_audio_video_to_hf.py and convert_perception_lm_weights_to_hf.py
        num_heads = 8
        num_key_value_heads = 8
        head_dim = 512
        for k in list(state_dict.keys()):
            # Only permute decoder Q/K weights, not encoder weights
            if "language_model.model.layers" in k and ".self_attn.q_proj.weight" in k:
                weight = state_dict[k]
                state_dict[k] = weight.view(num_heads, head_dim // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)
                if verbose:
                    print(f"Permuted {k} for RoPE: {weight.shape} -> {state_dict[k].shape}")
            elif "language_model.model.layers" in k and ".self_attn.k_proj.weight" in k:
                weight = state_dict[k]
                dim1, dim2 = weight.shape
                state_dict[k] = weight.view(num_key_value_heads, head_dim // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)
                if verbose:
                    print(f"Permuted {k} for RoPE: {weight.shape} -> {state_dict[k].shape}")

    # Pad lm_head weight if needed: the original final_proj has vocab_size outputs
    # but LlamaForCausalLM creates lm_head with vocab_size + num_special_tokens.
    # Zero-pad the extra rows so the special tokens don't produce meaningful logits.
    lm_head_key = None
    for k in state_dict:
        if "lm_head.weight" in k:
            lm_head_key = k
            break
    if lm_head_key is not None and lm_head_key in hf_model.state_dict():
        src_shape = state_dict[lm_head_key].shape
        tgt_shape = hf_model.state_dict()[lm_head_key].shape
        if src_shape[0] < tgt_shape[0]:
            pad_rows = tgt_shape[0] - src_shape[0]
            padding = torch.zeros(pad_rows, src_shape[1], dtype=state_dict[lm_head_key].dtype, device=state_dict[lm_head_key].device)
            state_dict[lm_head_key] = torch.cat([state_dict[lm_head_key], padding], dim=0)
            logger.info(f"Padded {lm_head_key} from {list(src_shape)} to {list(state_dict[lm_head_key].shape)}")

    # Check for missing or extra keys
    extra_keys = set(state_dict.keys()) - set(hf_model.state_dict().keys())
    extra_keys = set({k for k in extra_keys if "num_updates" not in k})  # filter unnecessary param
    if len(extra_keys) != 0:
        raise ValueError(f"{len(extra_keys)} extra keys found: {extra_keys}")
    missing_keys = set(hf_model.state_dict().keys()) - set(state_dict.keys())
    if len(missing_keys) != 0:
        raise ValueError(f"{len(missing_keys)} missing keys found: {missing_keys}")
    hf_model.load_state_dict(state_dict, strict=True)
    n_params = param_count(hf_model)

    logger.info(f"model loaded: {round(n_params / 1e6, 1)}M params")

    hf_model.eval()
    del state_dict

    return hf_model


def param_count(model):
    return sum(p[1].numel() for p in model.named_parameters())


@torch.no_grad()
def convert_omniasr_checkpoint(model_card, repo_id=None, bfloat16=False):

    if not torch.cuda.is_available():
        logger.warning("CUDA is not available, conversion will be done on CPU but it is STRONGLY recommended to use GPU for proper removal of weight norm.")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    if not bfloat16:
        dtype = torch.float32
    else:
        dtype = torch.bfloat16

    # 1) Load original model
    assert model_card is not None, "Must specify original model name in omnilingual-asr"
    if "W2V" not in model_card:
        pipeline = ASRInferencePipeline(model_card=model_card, device=device, dtype=dtype)
        original_model = pipeline.model
        original_tokenizer = pipeline.tokenizer
    else:
        original_model = load_model(model_card, device=device, dtype=dtype)
        original_tokenizer = None

    resolver = get_dependency_resolver()
    if "CTC" in model_card or "LLM" in model_card:
        # https://github.com/facebookresearch/omnilingual-asr/blob/9b95719b482d755c8dc9ec1aff7b477f4dd89d6c/src/omnilingual_asr/models/wav2vec2_asr/config.py#L13
        if "300" in model_card:
            encoder_config_name = "large_lv60k"
        elif "1b" in model_card:
            encoder_config_name = "1b"
        elif "3b" in model_card:
            encoder_config_name = "3b"
        elif "7b" in model_card:
            encoder_config_name = "7b"
        else:
            raise ValueError(f"Unsupported size, got {model_card}")

        original_config = get_config(resolver, Wav2Vec2AsrConfig, "base_10h")
        original_config.encoder_config = get_config(resolver, Wav2Vec2Config, encoder_config_name).encoder_config

        original_config.encoder_config.dropout_p = 0.0
        original_config.encoder_config.attn_dropout_p = 0.0
        original_config.encoder_config.ffn_inner_dropout_p = 0.1
        original_config.encoder_config.layer_drop_p = 0.1

        original_config.use_masking = False
        original_config.max_temporal_mask_prob = 0.0
        original_config.max_spatial_mask_prob = 0.0
        original_config.target_vocab_size = original_tokenizer.vocab_info.size

        if "LLM" in model_card:
            # load additional configuration for LLM, beam search, streaming
            
            # v2: https://github.com/facebookresearch/omnilingual-asr/blob/81f51e224ce9e74b02cc2a3eaf21b2d91d743455/src/omnilingual_asr/models/wav2vec2_llama/config.py#L257
            # v1: https://github.com/facebookresearch/omnilingual-asr/blob/81f51e224ce9e74b02cc2a3eaf21b2d91d743455/src/omnilingual_asr/models/wav2vec2_llama/config.py#L229
            # v2 and v1 are same except for vocab size which we can get programmatically
            llama_config = LLaMAConfig(
                model_dim=4096,
                max_seq_len=8192,
                vocab_size=original_tokenizer.vocab_info.size,
                pad_idx=1,
                num_layers=12,
                num_attn_heads=8,
                num_key_value_heads=8,
                ffn_inner_dim=4096,
                rope_theta=10_000.0,
                dropout_p=0.1,
            )
            original_config_llm = Wav2Vec2LlamaConfig(
                wav2vec2_asr_config=original_config, llama_config=llama_config
            )
            original_config_llm.lang_embeddings_p = 0.5
            original_config_llm.n_special_tokens = 1
            original_config_llm.model_type = ModelType.LLM_ASR_LID

            if "unlimited" in model_card.lower():
                original_config_llm.n_special_tokens = 3
                original_config_llm.lang_embeddings_p = 0.8
                original_config_llm.streaming_config = Wav2Vec2LlamaStreamingConfig(
                    is_streaming=True,
                    text_tokenizer="omniASR_tokenizer_written_v2",
                )
            elif "zs" in model_card.lower():
                original_config_llm.llama_config.max_seq_len = 16384
                original_config_llm.encoder_stacking = 3
                original_config_llm.n_special_tokens = 6
                original_config_llm.model_type = ModelType.ZERO_SHOT
                original_config_llm.n_context_examples = 10
                original_config_llm.lang_embeddings_p = 0.0

                # TODO remove this? already set correctly?
                vocab_size = 9812
                original_config_llm.llama_config.vocab_size = vocab_size
                original_config_llm.wav2vec2_asr_config.target_vocab_size = vocab_size
    
    elif "W2V" in model_card:
        # https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/models/wav2vec2_ssl/config.py
        original_config = get_config(resolver, Wav2Vec2Config, "large_lv60k")
        original_config.encoder_config.attn_dropout_p = 0.0
        if "300" in model_card:
            pass
        elif "1b" in model_card:
            original_config.encoder_config.model_dim = 1280
            original_config.encoder_config.num_encoder_layers = 48
            original_config.encoder_config.ffn_inner_dim = 5120
            original_config.encoder_config.dropout_p = 0.0
            original_config.quantized_dim = 1024
            original_config.final_dim = 1024
            original_config.encoder_config.first_pass_dropout_p = 0.1
        elif "3b" in model_card:
            original_config.encoder_config.model_dim = 2048
            original_config.encoder_config.num_encoder_layers = 60
            original_config.encoder_config.ffn_inner_dim = 8192
            original_config.encoder_config.dropout_p = 0.0
            original_config.quantized_dim = 1024
            original_config.final_dim = 1024
            original_config.encoder_config.first_pass_dropout_p = 0.1
        elif "7b" in model_card:
            original_config.encoder_config.model_dim = 2048
            original_config.encoder_config.num_encoder_layers = 128
            original_config.encoder_config.ffn_inner_dim = 8192
            original_config.encoder_config.dropout_p = 0.0
            original_config.quantized_dim = 1024
            original_config.final_dim = 1024
            original_config.encoder_config.first_pass_dropout_p = 0.1
            original_config.encoder_config.num_encoder_attn_heads = 16
        else:
            raise ValueError(f"Unsupported size, got {model_card}")

        # NOTE added but not done in original like with CTC
        original_config.use_masking = False
        original_config.max_temporal_mask_prob = 0.0
        original_config.max_spatial_mask_prob = 0.0

    else:
        raise ValueError(f"Unsupported model type, got {model_card}")

    # 2) Initialize Transformers model
    conv_dim, conv_kernel, conv_stride = zip(*original_config.encoder_config.feature_extractor_layer_descs)
    layer_norm_pre = original_config.encoder_config.norm_order == TransformerNormOrder.PRE
    feat_extract_norm = "layer" if original_config.encoder_config.feature_extractor_layer_norm_convs else "group"
    encoder_config = OmniASREncoderConfig(
        # TODO clean up unused and make Transformers compatible
        max_seq_len=original_config.encoder_config.max_seq_len, 
        feature_dim=original_config.encoder_config.feature_dim, 
        use_fbank=original_config.encoder_config.use_fbank, 
        first_pass_dropout_p=original_config.encoder_config.first_pass_dropout_p, 
        layer_norm_features=original_config.encoder_config.layer_norm_features,
        feature_grad_scale=original_config.encoder_config.feature_grad_scale, 
        num_fbank_channels=original_config.encoder_config.num_fbank_channels, 
        fbank_stride=original_config.encoder_config.fbank_stride, 
        sample_fbank_every_k=original_config.encoder_config.sample_fbank_every_k, 
        pos_encoder_depth=original_config.encoder_config.pos_encoder_depth, 
        use_conformer=original_config.encoder_config.use_conformer, 
        # dropout_p=original_config.encoder_config.dropout_p, 
        depthwise_conv_kernel_size=original_config.encoder_config.depthwise_conv_kernel_size,
        # NOTE (ebezzam): below are modified to Transformer convention
        hidden_size=original_config.encoder_config.model_dim,
        conv_dim=conv_dim,
        conv_kernel=conv_kernel,
        conv_stride=conv_stride,
        conv_bias=original_config.encoder_config.feature_extractor_bias,
        feat_extract_norm=feat_extract_norm,
        layer_norm_pre=layer_norm_pre,
        attention_dropout=original_config.encoder_config.attn_dropout_p,
        num_hidden_layers=original_config.encoder_config.num_encoder_layers, 
        num_attention_heads=original_config.encoder_config.num_encoder_attn_heads, 
        num_conv_pos_embeddings=original_config.encoder_config.pos_conv_kernel_size,
        num_conv_pos_embedding_groups=original_config.encoder_config.num_pos_conv_groups,
        hidden_dropout=original_config.encoder_config.ffn_inner_dropout_p,
        activation_dropout=original_config.encoder_config.ffn_inner_dropout_p,
        intermediate_size=original_config.encoder_config.ffn_inner_dim,
        position_embeddings_type=original_config.encoder_config.pos_encoder_type, 
        final_dropout=original_config.final_dropout_p,
        layerdrop=original_config.encoder_config.layer_drop_p,
        # NOTE (ebezzam) do we keep specaugment params?
        apply_spec_augment=original_config.use_masking,
        mask_time_length=original_config.temporal_mask_span_len,
        mask_time_prob=original_config.max_temporal_mask_prob,
        mask_time_min_masks=original_config.min_num_temporal_mask_spans,
        mask_feature_length=original_config.spatial_mask_span_len,
        mask_feature_prob=original_config.max_spatial_mask_prob,
        mask_feature_min_masks=original_config.min_num_spatial_mask_spans,
    )

    if "CTC" in model_card:
        config = OmniASRCTCConfig(
            encoder_config=encoder_config,
            vocab_size=original_config.target_vocab_size,
            pad_token_id=pipeline.tokenizer.vocab_info.pad_idx,
            bos_token_id=pipeline.tokenizer.vocab_info.bos_idx,
            eos_token_id=pipeline.tokenizer.vocab_info.eos_idx,
            unk_token_id=pipeline.tokenizer.vocab_info.unk_idx,
        )
        hf_model = OmniASRForCTC(config)
    elif "LLM" in model_card:

        """
        (ffn): GLUFeedForwardNetwork(                                                                                       
          inner_dim_scale=0.666667, inner_dim_to_multiple=256                                                               
          (gate_proj): Linear(input_dim=4096, output_dim=2816, bias=False, init_fn=init_projection)                         
          (gate_activation): SiLU()  
          (inner_proj): Linear(input_dim=4096, output_dim=2816, bias=False, init_fn=init_projection)                        
          (inner_dropout): Dropout(p=0.1, inplace=False)                                                                    
          (output_proj): Linear(input_dim=2816, output_dim=4096, bias=False, init_fn=init_projection)                       
        ) 
        """
        # Compute Llama config
        # -- Compute intermediate_size according to original: https://github.com/facebookresearch/fairseq2/blob/main/src/fairseq2/models/transformer/ffn.py#L274-L283
        intermediate_size = original_config_llm.llama_config.ffn_inner_dim
        inner_dim_scale = original_config_llm.llama_config.ffn_inner_dim_scale
        inner_dim_to_multiple = original_config_llm.llama_config.ffn_inner_dim_multiple_of
        if inner_dim_scale != 1.0:
            intermediate_size = int(intermediate_size * inner_dim_scale)
        if inner_dim_to_multiple != 1:
            intermediate_size = inner_dim_to_multiple * (
                (intermediate_size + inner_dim_to_multiple - 1) // inner_dim_to_multiple
            )
        llama_config = LlamaConfig(
            vocab_size=original_config_llm.llama_config.vocab_size + original_config_llm.n_special_tokens,
            hidden_size=original_config_llm.llama_config.model_dim,
            intermediate_size=intermediate_size,
            max_position_embeddings=original_config_llm.llama_config.max_seq_len,
            num_hidden_layers=original_config_llm.llama_config.num_layers,
            num_attention_heads=original_config_llm.llama_config.num_attn_heads,
            num_key_value_heads=original_config_llm.llama_config.num_key_value_heads,
            tie_word_embeddings=original_config_llm.llama_config.tied_embeddings,
            rope_theta=10000.0,
            rms_norm_eps=1e-5,
        )

        # TODO: adding special tokens?
        # see https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/models/wav2vec2_llama/factory.py#L212-L218
        # https://github.com/facebookresearch/omnilingual-asr/blob/81f51e224ce9e74b02cc2a3eaf21b2d91d743455/src/omnilingual_asr/models/wav2vec2_llama/config.py#L28

        config = OmniASRLLMConfig(
            encoder_config=encoder_config,
            text_config=llama_config,
            encoder_stacking=original_config_llm.encoder_stacking,
            # num_lang_embeddings=len(original_model.lang_embeddings.weight),   # TODO remove
            bos_token_id=original_config_llm.bos_idx,
            pad_token_id=original_config_llm.pad_idx,
            eos_token_id=original_config_llm.eos_idx,
            unk_token_id=original_config_llm.unk_idx,
            # TODO better handlnig?
            num_special_tokens=original_config_llm.n_special_tokens,
            language_mapping=original_model.lang_mapping,
            language_embedding_probability=original_config_llm.lang_embeddings_p,
            # LID marker is the first reserved special token, i.e. the base
            # vocabulary size before adding extra syntax tokens.
            language_token_id=original_config_llm.llama_config.vocab_size,
        )
        hf_model = OmniASRForConditionalGeneration(config)
    elif "W2V" in model_card:
        # TODO not working
        config = OmniASREncoderConfig(
            **encoder_config.to_dict()
        )
        hf_model = OmniASRModel(config)
    else:
        raise ValueError(f"Unsupported model type, got {model_card}")
    hf_model.to(device).to(dtype)

    # 3) Convert weights
    hf_model.apply_weight_norm()
    print(f"Total parameters (original): {param_count(original_model)}")
    print(f"Total parameters (HF)      : {param_count(hf_model)}")

    decoder_convert_list = None
    # TODO check w2v2 only model
    if "CTC" in model_card:
        decoder_convert_list = ctc_convert_list
    elif "LLM" in model_card:
        decoder_convert_list = llm_convert_list
    hf_model = _convert_model(original_model, hf_model, encoder_convert_list, decoder_convert_list)
    hf_model.remove_weight_norm()

    # 4) Prepare processor (feature extraction and tokenizer)
    feature_extractor = OmniASRFeatureExtractor(
    # feature_extractor = Wav2Vec2FeatureExtractor(
        # TODO check vals
        feature_size=1,
        sampling_rate=16000,
        padding_value=0,
        do_normalize=True,
        # TODO always layer or group?
        return_attention_mask=(encoder_config.feat_extract_norm == "layer"),
    )

    # -- create Transformers-compatible tokenizer
    # Release v1: https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/cards/models/rc_models_v1.yaml
    # Release v2: https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/cards/models/rc_models_v2.yaml
    if "v2" in model_card:
        tokenizer_url = "https://dl.fbaipublicfiles.com/mms/omniASR_tokenizer_written_v2.model"
    elif model_card in ["omniASR_LLM_7B"]:
        tokenizer_url = "https://dl.fbaipublicfiles.com/mms/omniASR_tokenizer_v7.model"
    else:
        tokenizer_url = "https://dl.fbaipublicfiles.com/mms/omniASR_tokenizer.model"
    download_dir = os.getcwd()
    tokenizer_path = os.path.join(download_dir, os.path.basename(tokenizer_url))
    urllib.request.urlretrieve(tokenizer_url, tokenizer_path)
    vocab_ids, vocab_scores, merges = SentencePieceExtractor(tokenizer_path).extract()
    # TODO do we also need to overwrite the pad token to be ID 0? (before <s>)
    vocab_scores[0] = ("<pad>", vocab_scores[0][1])
    # TODO create own tokenizer like LasrTokenizer but with correct special tokens?
    tokenizer = LasrTokenizer(vocab=vocab_scores)
    tokenizer.add_eos_token = False

    # # -- create Transformers-compatible tokenizer
    # vocab_path = "vocab.json"
    # vocab_dict = {}
    # for idx in range(original_tokenizer.vocab_info.size):
    #     token = original_tokenizer._model.index_to_token(idx)
    #     vocab_dict[token] = idx
    # with open(vocab_path, "w", encoding="utf-8") as f:
    #     json.dump(vocab_dict, f, ensure_ascii=False, indent=2)
    # # NOTE: For CTC models, pad_token should be the CTC blank token.
    # # In the original fairseq2 model, token 0 (<s> - BOS) is used as the CTC blank.
    # # Wav2Vec2CTCTokenizer uses pad_token_id as the blank token for CTC decoding.
    # tokenizer = Wav2Vec2CTCTokenizer(
    #     vocab_file=vocab_path,
    #     unk_token=original_tokenizer._model.index_to_token(original_tokenizer.vocab_info.unk_idx),
    #     # pad_token=original_tokenizer._model.index_to_token(original_tokenizer.vocab_info.pad_idx),
    #     pad_token=original_tokenizer._model.index_to_token(original_tokenizer.vocab_info.bos_idx),  # Use BOS as CTC blank
    #     bos_token=original_tokenizer._model.index_to_token(original_tokenizer.vocab_info.bos_idx),
    #     eos_token=original_tokenizer._model.index_to_token(original_tokenizer.vocab_info.eos_idx),
    #     word_delimiter_token="|",
    #     do_lower_case=False,    # TODO: set to True?
    # )

    # vocab_file = "/raid/eric/.cache/fairseq2/assets/e7be1a6acb8f76fdbca19dce/omniASR_tokenizer_written_v2.model"
    # # NOTE or directly use TokenizersBackend?
    # from ...tokenization_utils_tokenizers import TokenizersBackend
    # tokenizer = SeamlessM4TTokenizer(vocab_file=vocab_file) # leads to empty transcript

    # -- create processor
    language_mapping = None
    if "LLM" in model_card:
        language_mapping = original_model.lang_mapping
    processor = OmniASRProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer, language_mapping=language_mapping)

    # 5) Upload to hub
    if repo_id:
        logger.info("Pushing model to the Hub ...")
        hf_model.push_to_hub(repo_id)
        processor.push_to_hub(repo_id)

    # 6) Cleanup
    # if os.path.exists(vocab_path):
    #     os.remove(vocab_path)
    if os.path.exists(tokenizer_path):
        os.remove(tokenizer_path)

    # TODO try loading model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_card", default=None, type=str, help="Name of original model in omnilingual-asr")
    parser.add_argument("--repo_id", default=None, type=str, help="The repository ID for pushing the model to the Hub")
    parser.add_argument("--bfloat16", action="store_true", help="Whether to do bfloat16, otherwise default is float32")
    # Original defaults to bfloat16: https://github.com/facebookresearch/omnilingual-asr/blob/81f51e224ce9e74b02cc2a3eaf21b2d91d743455/src/omnilingual_asr/models/inference/pipeline.py#L157
    args = parser.parse_args()

    convert_omniasr_checkpoint(
        args.model_card,
        args.repo_id,
        args.bfloat16
    )
