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

import argparse
import json
import os
import torch

from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
from fairseq2.models.wav2vec2.asr.config import Wav2Vec2AsrConfig
from fairseq2.models.wav2vec2.config import Wav2Vec2Config
from fairseq2.data.tokenizers.hub import load_tokenizer
from fairseq2.models.hub import load_model
from fairseq2.runtime.dependency import get_dependency_resolver
from fairseq2.runtime.config_registry import ConfigRegistrar, get_config
from fairseq2.runtime.dependency import DependencyContainer, DependencyResolver
from fairseq2.models.transformer.norm_order import TransformerNormOrder

from transformers import (
    OmniASRCTCConfig,
    OmniASREncoderConfig,
    OmniASRForCTC,
    OmniASRModel,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    logging,
)


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

MAPPING = {
    "post_extract_proj": "feature_projection.projection",
    "encoder.pos_conv.0": "encoder.pos_conv_embed.conv",
    "self_attn.k_proj": "encoder.layers.*.attention.k_proj",
    "self_attn.v_proj": "encoder.layers.*.attention.v_proj",
    "self_attn.q_proj": "encoder.layers.*.attention.q_proj",
    "self_attn.out_proj": "encoder.layers.*.attention.out_proj",
    "self_attn_layer_norm": "encoder.layers.*.layer_norm",
    "fc1": "encoder.layers.*.feed_forward.intermediate_dense",
    "fc2": "encoder.layers.*.feed_forward.output_dense",
    "final_layer_norm": "encoder.layers.*.final_layer_norm",
    "encoder.layer_norm": "encoder.layer_norm",
    "adapter_layer": "encoder.layers.*.adapter_layer",
    "w2v_model.layer_norm": "feature_projection.layer_norm",
    "quantizer.weight_proj": "quantizer.weight_proj",
    "quantizer.vars": "quantizer.codevectors",
    "project_q": "project_q",
    "final_proj": "project_hid",
    "w2v_encoder.proj": "lm_head",
    "mask_emb": "masked_spec_embed",
    "pooling_layer.linear": "projector",
    "pooling_layer.projection": "classifier",
}
TOP_LEVEL_KEYS = [
    "lm_head",
    "quantizer.weight_proj",
    "quantizer.codevectors",
    "project_q",
    "project_hid",
    "projector",
    "classifier",
]


def read_txt_into_dict(filename):
    result = {}
    with open(filename, "r") as file:
        for line_number, line in enumerate(file):
            line = line.strip()
            if line:
                words = line.split()
                key = line_number
                value = words[0]
                result[key] = value
    return result


def set_recursively(key, value, full_name, weight_type, hf_pointer):
    for attribute in key.split("."):
        hf_pointer = getattr(hf_pointer, attribute)

    hf_param_name = None
    for param_key in PARAM_MAPPING:
        if full_name.endswith(param_key):
            hf_param_name = PARAM_MAPPING[full_name.split(".")[-1]]
            weight_type = "param"

    # fairseq uses nn.utils.weight_norm() while transformers switches to nn.utils.parametrizations.weight_norm()
    # the mapping between two versions:
    # https://github.com/pytorch/pytorch/blob/56935684c3dfad7841c83c719eeebecb560fe466/torch/nn/utils/parametrizations.py#L389-L395

    if weight_type is not None and weight_type != "param":
        if weight_type == "weight_g" and not hasattr(hf_pointer, "weight_g"):
            hf_shape = hf_pointer.parametrizations.weight.original0.shape
        elif weight_type == "weight_v" and not hasattr(hf_pointer, "weight_v"):
            hf_shape = hf_pointer.parametrizations.weight.original1.shape
        else:
            hf_shape = getattr(hf_pointer, weight_type).shape
    elif weight_type is not None and weight_type == "param":
        shape_pointer = hf_pointer
        for attribute in hf_param_name.split("."):
            shape_pointer = getattr(shape_pointer, attribute)
        hf_shape = shape_pointer.shape

        # let's reduce dimension
        value = value[0]
    else:
        hf_shape = hf_pointer.shape

    if hf_shape != value.shape:
        raise ValueError(
            f"Shape of hf {key + '.' + weight_type if weight_type is not None else ''} is {hf_shape}, but should be"
            f" {value.shape} for {full_name}"
        )

    if weight_type == "weight":
        hf_pointer.weight.data = value
    elif weight_type == "weight_g":
        if hasattr(hf_pointer, "weight_g"):
            hf_pointer.weight_g.data = value
        else:
            hf_pointer.parametrizations.weight.original0.data = value
    elif weight_type == "weight_v":
        if hasattr(hf_pointer, "weight_v"):
            hf_pointer.weight_v.data = value
        else:
            hf_pointer.parametrizations.weight.original1.data = value
    elif weight_type == "bias":
        hf_pointer.bias.data = value
    elif weight_type == "param":
        for attribute in hf_param_name.split("."):
            hf_pointer = getattr(hf_pointer, attribute)
        hf_pointer.data = value
    else:
        hf_pointer.data = value

    logger.info(f"{key + '.' + weight_type if weight_type is not None else ''} was initialized from {full_name}.")


def rename_dict(key, value, full_name, weight_type, hf_dict):
    hf_param_name = None
    for param_key in PARAM_MAPPING:
        if full_name.endswith(param_key):
            hf_param_name = PARAM_MAPPING[full_name.split(".")[-1]]
            weight_type = "param"

    if weight_type is not None and weight_type != "param":
        full_key = ".".join([key, weight_type])
    elif weight_type is not None and weight_type == "param":
        full_key = ".".join([key, hf_param_name])
    else:
        full_key = key

    hf_dict[full_key] = value if "lm_head" in full_key else value[0]


PARAM_MAPPING = {
    "W_a": "linear_1.weight",
    "W_b": "linear_2.weight",
    "b_a": "linear_1.bias",
    "b_b": "linear_2.bias",
    "ln_W": "norm.weight",
    "ln_b": "norm.bias",
}


def load_wav2vec2_layer(name, value, hf_model=None, hf_dict=None):
    is_used = False
    for key, mapped_key in MAPPING.items():
        mapped_key = "wav2vec2." + mapped_key if mapped_key not in TOP_LEVEL_KEYS else mapped_key
        if key in name or key.split("w2v_model.")[-1] == name.split(".")[0]:
            is_used = True
            if "*" in mapped_key:
                layer_index = name.split(key)[0].split(".")[-2]
                mapped_key = mapped_key.replace("*", layer_index)
            if "weight_g" in name:
                weight_type = "weight_g"
            elif "weight_v" in name:
                weight_type = "weight_v"
            elif "bias" in name:
                weight_type = "bias"
            elif "weight" in name:
                # TODO: don't match quantizer.weight_proj
                weight_type = "weight"
            else:
                weight_type = None
            if hf_dict is not None:
                rename_dict(mapped_key, value, name, weight_type, hf_dict)
            else:
                set_recursively(mapped_key, value, name, weight_type, hf_model)
            return is_used
    return is_used


def recursively_load_weights(fairseq_model, hf_model, is_headless):
    unused_weights = []
    fairseq_dict = fairseq_model.state_dict()

    feature_extractor = hf_model.wav2vec2.feature_extractor

    for name, value in fairseq_dict.items():
        is_used = False
        if "conv_layers" in name:
            load_conv_layer(
                name,
                value,
                feature_extractor,
                unused_weights,
                hf_model.config.feat_extract_norm == "group",
            )
            is_used = True
        else:
            is_used = load_wav2vec2_layer(name, value, hf_model)
        if not is_used:
            unused_weights.append(name)

    logger.warning(f"Unused weights: {unused_weights}")


def load_conv_layer(full_name, value, feature_extractor, unused_weights, use_group_norm):
    name = full_name.split("conv_layers.")[-1]
    items = name.split(".")
    layer_id = int(items[0])
    type_id = int(items[1])

    if type_id == 0:
        if "bias" in name:
            if value.shape != feature_extractor.conv_layers[layer_id].conv.bias.data.shape:
                raise ValueError(
                    f"{full_name} has size {value.shape}, but"
                    f" {feature_extractor.conv_layers[layer_id].conv.bias.data.shape} was found."
                )
            feature_extractor.conv_layers[layer_id].conv.bias.data = value
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
        elif "weight" in name:
            if value.shape != feature_extractor.conv_layers[layer_id].conv.weight.data.shape:
                raise ValueError(
                    f"{full_name} has size {value.shape}, but"
                    f" {feature_extractor.conv_layers[layer_id].conv.weight.data.shape} was found."
                )
            feature_extractor.conv_layers[layer_id].conv.weight.data = value
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
    elif (type_id == 2 and not use_group_norm) or (type_id == 2 and layer_id == 0 and use_group_norm):
        if "bias" in name:
            if value.shape != feature_extractor.conv_layers[layer_id].layer_norm.bias.data.shape:
                raise ValueError(
                    f"{full_name} has size {value.shape}, but"
                    f" {feature_extractor.conv_layers[layer_id].layer_norm.bias.data.shape} was found."
                )
            feature_extractor.conv_layers[layer_id].layer_norm.bias.data = value
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
        elif "weight" in name:
            if value.shape != feature_extractor.conv_layers[layer_id].layer_norm.weight.data.shape:
                raise ValueError(
                    f"{full_name} has size {value.shape}, but"
                    f" {feature_extractor.conv_layers[layer_id].layer_norm.weight.data.shape} was found."
                )
            feature_extractor.conv_layers[layer_id].layer_norm.weight.data = value
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
    else:
        unused_weights.append(full_name)


@torch.no_grad()
def convert_omniasr_checkpoint(
    model_card, output_dir, repo_id=None
):

    # 1) Load original model
    pipeline = ASRInferencePipeline(model_card=model_card)
    original_model = pipeline.model

    resolver = get_dependency_resolver()
    if "CTC" in model_card:
        # https://github.com/facebookresearch/omnilingual-asr/blob/9b95719b482d755c8dc9ec1aff7b477f4dd89d6c/src/omnilingual_asr/models/wav2vec2_asr/config.py#L13
        if "v2" in model_card:
            vocab_size = 10288
        else:
            vocab_size = 9812
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
        original_config.target_vocab_size = vocab_size

    else:
        raise ValueError(f"Only CTC models are supported for now, got {model_card}")
        # Base model: https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/models/wav2vec2_ssl/config.py
        # LLM decoder: https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/models/wav2vec2_llama/config.py

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
        pos_encoder_type=original_config.encoder_config.pos_encoder_type, 
        pos_encoder_depth=original_config.encoder_config.pos_encoder_depth, 
        use_conformer=original_config.encoder_config.use_conformer, 
        dropout_p=original_config.encoder_config.dropout_p, 
        layer_drop_p=original_config.encoder_config.layer_drop_p,
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
    )
    config = OmniASRCTCConfig(
        vocab_size=original_config.target_vocab_size,
        final_dropout=original_config.final_dropout_p,
        apply_spec_augment=original_config.use_masking,
        mask_time_length=original_config.temporal_mask_span_len,
        mask_time_prob=original_config.max_temporal_mask_prob,
        mask_time_min_masks=original_config.min_num_temporal_mask_spans,
        mask_feature_length=original_config.spatial_mask_span_len,
        mask_feature_prob=original_config.max_spatial_mask_prob,
        mask_feature_min_masks=original_config.min_num_spatial_mask_spans,
        encoder_config=encoder_config,
    )
    model = OmniASRForCTC(config)

    # count number of parameters
    total_params = sum(p.numel() for p in original_model.parameters())
    print(f"Total parameters (original): {total_params}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters (HF): {total_params}")
    

    
    raise ValueError



    if is_finetuned:
        if dict_path:
            target_dict = Dictionary.load(dict_path)

            # important change bos & pad token id since CTC symbol is <pad> and
            # not <s> as in fairseq
            original_config.bos_token_id = target_dict.pad_index
            original_config.pad_token_id = target_dict.bos_index
            original_config.eos_token_id = target_dict.eos_index
            original_config.vocab_size = len(target_dict.symbols)
            vocab_path = os.path.join(pytorch_dump_folder_path, "vocab.json")
            if not os.path.isdir(pytorch_dump_folder_path):
                logger.error(f"--pytorch_dump_folder_path ({pytorch_dump_folder_path}) should be a directory")
                return
            os.makedirs(pytorch_dump_folder_path, exist_ok=True)
            vocab_dict = target_dict.indices

            # fairseq has the <pad> and <s> switched
            vocab_dict["<pad>"] = 0
            vocab_dict["<s>"] = 1
            with open(vocab_path, "w", encoding="utf-8") as vocab_handle:
                json.dump(vocab_dict, vocab_handle)
            tokenizer = Wav2Vec2CTCTokenizer(
                vocab_path,
                unk_token=target_dict.unk_word,
                pad_token=target_dict.pad_word,
                bos_token=target_dict.bos_word,
                eos_token=target_dict.eos_word,
                word_delimiter_token="|",
                do_lower_case=False,
            )
            return_attention_mask = original_config.feat_extract_norm == "layer"
            feature_extractor = Wav2Vec2FeatureExtractor(
                feature_size=1,
                sampling_rate=16000,
                padding_value=0,
                do_normalize=True,
                return_attention_mask=return_attention_mask,
            )
            processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
            processor.save_pretrained(pytorch_dump_folder_path)

        hf_wav2vec = OmniASRForCTC(original_config)

    if is_finetuned:
        model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [checkpoint_path], arg_overrides={"data": "/".join(dict_path.split("/")[:-1])}
        )
    else:
        task_arg = argparse.Namespace(task="audio_pretraining")
        task = fairseq.tasks.setup_task(task_arg)

        model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([checkpoint_path], task=task)

    model = model[0].eval()

    recursively_load_weights(model, hf_wav2vec, not is_finetuned)

    hf_wav2vec.save_pretrained(pytorch_dump_folder_path)

    # upload to hub
    if repo_id:
        logger.info("Pushing model to the Hub ...")
        hf_wav2vec.push_to_hub(repo_id)
        processor.push_to_hub(repo_id)

"""
Reproducible usage
------------------

Setup
```
pip install -e .
pip install omnilingual-asr

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
python src/transformers/models/omniasr/convert_omniasr_to_hf.py \
    --model_card omniASR_CTC_300M_v2 \
    --output_dir omniasr_ctc_300m_v2 \
    --repo_id bezzam/omniasr-ctc-300m-v2
```
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_card", default=None, type=str, help="Name of original model in omnilingual-asr")
    parser.add_argument("--output_dir", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument("--repo_id", default=None, type=str, help="The repository ID for pushing the model to the Hub")
    args = parser.parse_args()

    convert_omniasr_checkpoint(
        args.model_card,
        args.output_dir,
        args.repo_id,
    )
