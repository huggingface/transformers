# coding=utf-8
# Copyright 2025 Meta Platforms, Inc. and the HuggingFace Inc. team. All rights reserved.
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
import gc
import json
import os
import tempfile
import warnings

import torch
from timm.models.eva import checkpoint_filter_fn
from tokenizers import AddedToken, processors

from transformers import (
    GenerationConfig,
    LlamaConfig,
    LlamaTokenizer,
    PreTrainedTokenizerFast,
)
from transformers.convert_slow_tokenizer import TikTokenConverter
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.perception_lm.configuration_perception_lm import (
    PerceptionLMConfig,
)
from transformers.models.perception_lm.image_processing_perception_lm_fast import (
    PerceptionLMImageProcessorFast,
)
from transformers.models.perception_lm.modeling_perception_lm import (
    PerceptionLMForConditionalGeneration,
)
from transformers.models.perception_lm.processing_perception_lm import (
    PerceptionLMProcessor,
)
from transformers.models.perception_lm.video_processing_perception_lm import (
    PerceptionLMVideoProcessor,
)
from transformers.models.timm_wrapper.configuration_timm_wrapper import TimmWrapperConfig


try:
    from transformers import LlamaTokenizerFast
except ImportError as e:
    warnings.warn(e)
    warnings.warn(
        "The converted tokenizer will be the `slow` tokenizer. To use the fast, update your `tokenizers` library and re-run the tokenizer conversion"
    )
    LlamaTokenizerFast = None

"""
Sample usage:

```
python src/transformers/models/perception_lm/convert_perception_lm_weights_to_hf.py \
    --input_dir /path/to/downloaded/perception_lm/model_path  --output_dir /output/path
```

Thereafter, models can be loaded via:

```py
from transformers import LlamaForCausalLM, LlamaTokenizer

model = LlamaForCausalLM.from_pretrained("/output/path")
tokenizer = LlamaTokenizer.from_pretrained("/output/path")
```

Important note: you need to be able to host the whole model in RAM to execute this script (even if the biggest versions
come in several checkpoints they each contain a part of each weight of the model, so we need to load them all in RAM).

If you want your tokenizer to add a bos automatically you should update the tokenizer._tokenizers.post_processor:

```py
from tokenizers import processors
bos = "<|begin_of_text|>"
tokenizer._tokenizers.post_processor = processors.Sequence(
    [
        processors.ByteLevel(trim_offsets=False),
        processors.TemplateProcessing(
            single=f"{bos}:0 $A:0",
            pair=f"{bos}:0 $A:0 {bos}:1 $B:1",
            special_tokens=[
                (bos, tokenizer.encode(bos)),
            ],
        ),
    ]
)
```

"""

BOS_ADDED_TOKEN = AddedToken(
    "<|begin_of_text|>",
    single_word=False,
    lstrip=False,
    rstrip=False,
    normalized=False,
    special=True,
)
EOS_ADDED_TOKEN = AddedToken(
    "<|end_of_text|>",
    single_word=False,
    lstrip=False,
    rstrip=False,
    normalized=False,
    special=True,
)
EOT_ADDED_TOKEN = AddedToken(
    "<|eot_id|>",
    single_word=False,
    lstrip=False,
    rstrip=False,
    normalized=False,
    special=True,
)

DEFAULT_SPECIAL_TOKENS = {
    "perception_lm": [
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|image|>",
        "<|video|>",
        "<|reserved_special_token_2|>",
        "<|reserved_special_token_3|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|reserved_special_token_4|>",
        "<|eot_id|>",  # End of turn
    ]
    + [f"<|reserved_special_token_{i}|>" for i in range(5, 256 - 5)]
}

CHAT_TEMPLATE = (
    "{{- bos_token }}"
    "{%- if messages[0]['role'] == 'system' -%}"
    "    {%- set system_message = messages[0]['content']|trim %}\n"
    "    {%- set messages = messages[1:] %}\n"
    "{%- else %}"
    "    {%- set system_message = 'You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.' %}"
    "{%- endif %}"
    "{{- '<|start_header_id|>system<|end_header_id|>\\n\\n' }}"
    "{{- system_message }}"
    "{{- '<|eot_id|>' }}"
    "{%- for message in messages %}"
    "{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' }}"
    "{%- for content in message['content'] | selectattr('type', 'equalto', 'image') %}"
    "{{ '<|image|>' }}"
    "{%- endfor %}"
    "{%- for content in message['content'] | selectattr('type', 'equalto', 'video') %}"
    "{{ '<|video|>' }}"
    "{%- endfor %}"
    "{%- for content in message['content'] | selectattr('type', 'equalto', 'text') %}"
    "{{- content['text'] | trim }}"
    "{%- endfor %}"
    "{{'<|eot_id|>' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
    "{%- endif %}"
)


def compute_intermediate_size(n, ffn_dim_multiplier=1, multiple_of=256):
    return multiple_of * ((int(ffn_dim_multiplier * int(8 * n / 3)) + multiple_of - 1) // multiple_of)


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f)


def write_weights(state_dict, index_dict, param_count, filename):
    for k, v in state_dict.items():
        index_dict["weight_map"][k] = filename
        param_count += v.numel()
    torch.save(state_dict, filename)
    print(f"Saved {filename}")
    return param_count


def write_model(
    model_path,
    input_base_path,
    params,
    image_token_id,
    safe_serialization=True,
    tokenizer=None,
    num_shards=None,
    push_to_hub=False,
):
    print("Converting the model.")
    num_shards = 1
    model_params = params.get("model", params)
    n_layers = model_params["n_layers"]
    n_heads = model_params["n_heads"]
    dim = model_params["dim"]
    dims_per_head = dim // n_heads
    base = model_params.get("rope_theta", 10000.0)
    inv_freq = 1.0 / (base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head))
    context_length = model_params["max_seqlen"]
    max_position_embeddings = context_length
    tie_word_embeddings = model_params.get("weight_tying", False)
    projector_pooling_ratio = model_params.get("pooling_ratio", 1)

    if model_params.get("n_kv_heads", None) is not None:
        num_key_value_heads = model_params["n_kv_heads"]  # for GQA / MQA
        key_value_dim = dims_per_head * num_key_value_heads
    else:  # compatibility with other checkpoints
        num_key_value_heads = n_heads
        key_value_dim = dim

    # permute for sliced rotary
    def permute(w, n_heads, dim1=dim, dim2=dim):
        return w.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)

    with tempfile.TemporaryDirectory() as tmp_model_path:
        print(f"Fetching all parameters from the checkpoint at {input_base_path}.")
        # Load weights
        if num_shards == 1:
            # Not sharded
            # (The sharded implementation would also work, but this is simpler.)
            loaded = torch.load(
                os.path.join(input_base_path, "consolidated.pth"),
                map_location="cpu",
                weights_only=True,
            )
        else:
            # Sharded
            checkpoint_list = sorted([file for file in os.listdir(input_base_path) if file.endswith(".pth")])
            print("Loading in order:", checkpoint_list)
            loaded = [
                torch.load(
                    os.path.join(input_base_path, file),
                    map_location="cpu",
                    weights_only=True,
                )
                for file in checkpoint_list
            ]
        param_count = 0
        index_dict = {"weight_map": {}}
        for layer_i in range(n_layers):
            filename = f"pytorch_model-{layer_i + 1}-of-{n_layers + 2}.bin"
            assert num_shards == 1, "PerceptionLM does not support sharded weights"
            state_dict = {
                f"model.language_model.layers.{layer_i}.self_attn.q_proj.weight": permute(
                    loaded[f"layers.{layer_i}.attention.wq.weight"], n_heads=n_heads
                ),
                f"model.language_model.layers.{layer_i}.self_attn.k_proj.weight": permute(
                    loaded[f"layers.{layer_i}.attention.wk.weight"],
                    n_heads=num_key_value_heads,
                    dim1=key_value_dim,
                ),
                f"model.language_model.layers.{layer_i}.self_attn.v_proj.weight": loaded[
                    f"layers.{layer_i}.attention.wv.weight"
                ],
                f"model.language_model.layers.{layer_i}.self_attn.o_proj.weight": loaded[
                    f"layers.{layer_i}.attention.wo.weight"
                ],
                f"model.language_model.layers.{layer_i}.mlp.gate_proj.weight": loaded[
                    f"layers.{layer_i}.feed_forward.w1.weight"
                ],
                f"model.language_model.layers.{layer_i}.mlp.down_proj.weight": loaded[
                    f"layers.{layer_i}.feed_forward.w2.weight"
                ],
                f"model.language_model.layers.{layer_i}.mlp.up_proj.weight": loaded[
                    f"layers.{layer_i}.feed_forward.w3.weight"
                ],
                f"model.language_model.layers.{layer_i}.input_layernorm.weight": loaded[
                    f"layers.{layer_i}.attention_norm.weight"
                ],
                f"model.language_model.layers.{layer_i}.post_attention_layernorm.weight": loaded[
                    f"layers.{layer_i}.ffn_norm.weight"
                ],
            }
            state_dict[f"model.language_model.layers.{layer_i}.self_attn.rotary_emb.inv_freq"] = inv_freq
            for k, v in state_dict.items():
                index_dict["weight_map"][k] = filename
                param_count += v.numel()
            torch.save(state_dict, os.path.join(tmp_model_path, filename))
            print(f"Saved {filename}")

        filename = f"pytorch_model-{n_layers + 1}-of-{n_layers + 2}.bin"

        state_dict = {
            "model.language_model.embed_tokens.weight": loaded["tok_embeddings.weight"],
            "model.language_model.norm.weight": loaded["norm.weight"],
            "model.multi_modal_projector.linear_1.weight": loaded["vision_projector.projector.0.weight"],
            "model.multi_modal_projector.linear_2.weight": loaded["vision_projector.projector.2.weight"],
            "model.multi_modal_projector.linear_1.bias": loaded["vision_projector.projector.0.bias"],
            "model.multi_modal_projector.linear_2.bias": loaded["vision_projector.projector.2.bias"],
        }
        if not tie_word_embeddings:
            state_dict["lm_head.weight"] = loaded["output.weight"]
        for k, v in state_dict.items():
            index_dict["weight_map"][k] = filename
            param_count += v.numel()
        torch.save(state_dict, os.path.join(tmp_model_path, filename))
        print(f"Saved {filename}")

        filename = f"pytorch_model-{n_layers + 2}-of-{n_layers + 2}.bin"
        state_dict = {k.replace("vision_model.", ""): v for k, v in loaded.items() if "vision_model" in k}
        vision_params = model_params["vision_model"]
        if vision_params["layers"] == 23 and vision_params["width"] == 1024:
            architecture = "vit_pe_core_large_patch14_336"
        elif vision_params["layers"] == 47 and vision_params["width"] == 1536:
            architecture = "vit_pe_core_gigantic_patch14_448"
        else:
            raise ValueError(
                f"Unsupported PE config: {vision_params['layers']} layers and {vision_params['width']} width"
            )

        vision_config = TimmWrapperConfig.from_pretrained(
            f"timm/{architecture}.fb",
            model_args={
                "embed_dim": vision_params["width"],
                "depth": vision_params["layers"],
                "img_size": (vision_params["image_size"], vision_params["image_size"]),
                "global_pool": "",
                "use_post_transformer_norm": vision_params["use_ln_post"],
                "init_values": vision_params["ls_init_value"],
                "ref_feat_shape": (
                    vision_params["image_size"] // vision_params["patch_size"],
                    vision_params["image_size"] // vision_params["patch_size"],
                ),
            },
        )

        perception_encoder = AutoModel.from_config(vision_config)
        state_dict = checkpoint_filter_fn(state_dict, perception_encoder)
        state_dict = {"model.vision_tower.timm_model." + k: v for k, v in state_dict.items()}
        for k, v in state_dict.items():
            index_dict["weight_map"][k] = filename
            param_count += v.numel()
        torch.save(state_dict, os.path.join(tmp_model_path, filename))
        print(f"Saved {filename}")

        # Write configs
        index_dict["metadata"] = {"total_size": param_count * 2}
        write_json(index_dict, os.path.join(tmp_model_path, "pytorch_model.bin.index.json"))
        ffn_dim_multiplier = model_params.get("ffn_dim_multiplier", 1)
        multiple_of = model_params.get("multiple_of", 256)

        bos_token_id = tokenizer.convert_tokens_to_ids("<|begin_of_text|>")
        eos_token_id = [tokenizer.convert_tokens_to_ids(t) for t in ["<|end_of_text|>", "<|eot_id|>"]]

        use_scaled_rope = model_params["use_scaled_rope"]
        if use_scaled_rope:
            rope_scaling = {
                "factor": model_params["rope_scale_factor"] * 1.0,
                "low_freq_factor": model_params.get("low_freq_factor", 1.0) * 1.0,
                "high_freq_factor": model_params.get("high_freq_factor", 4.0) * 1.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3",
            }
        else:
            rope_scaling = None

        text_config = LlamaConfig(
            hidden_size=dim,
            intermediate_size=compute_intermediate_size(dim, ffn_dim_multiplier, multiple_of),
            num_attention_heads=model_params["n_heads"],
            num_hidden_layers=model_params["n_layers"],
            rms_norm_eps=model_params["norm_eps"],
            num_key_value_heads=num_key_value_heads,
            vocab_size=len(tokenizer),
            rope_theta=base,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
        )

        config = PerceptionLMConfig(
            text_config=text_config.to_dict(),
            vision_config=vision_config.to_dict(),
            projector_pooling_ratio=projector_pooling_ratio,
            vision_use_cls_token=vision_params["use_cls_token"],
            image_token_id=tokenizer.image_token_id,
            video_token_id=tokenizer.video_token_id,
        )

        config.save_pretrained(tmp_model_path)

        generation_config = GenerationConfig(
            do_sample=False,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
        )
        generation_config.save_pretrained(tmp_model_path)

        # Make space so we can load the model properly now.
        del state_dict
        # output_weight = loaded.get("output.weight", None)
        del loaded
        gc.collect()

        print("Loading the checkpoint in a PerceptionLM model.")
        model = PerceptionLMForConditionalGeneration.from_pretrained(
            tmp_model_path, dtype=torch.bfloat16, low_cpu_mem_usage=True
        )
        # if not tie_word_embeddings:
        #     if output_weight is None:
        #         raise ValueError("Output weight/lm_head is not found in the checkpoint.")
        #     model.lm_head.load_state_dict({"weight": output_weight})

        # Avoid saving this as part of the config.
        del model.config._name_or_path
        model.config.dtype = torch.bfloat16

        print("Saving in the Transformers format.")
        if push_to_hub:
            print("Pushing to the hub.")
            model.push_to_hub(
                model_path,
                safe_serialization=safe_serialization,
                private=True,
                use_temp_dir=True,
            )
        else:
            print("Saving to disk.")
            model.save_pretrained(model_path, safe_serialization=safe_serialization)


class Llama3Converter(TikTokenConverter):
    def __init__(
        self,
        vocab_file,
        special_tokens=None,
        context_length=11520,
        **kwargs,
    ):
        super().__init__(vocab_file, additional_special_tokens=special_tokens, **kwargs)
        tokenizer = self.converted()

        self.converted_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            bos_token="<|begin_of_text|>",
            eos_token="<|eot_id|>",
            model_input_names=["input_ids", "attention_mask"],
            model_max_length=context_length,
            clean_up_tokenization_spaces=True,
            extra_special_tokens={
                "image_token": "<|image|>",
                "video_token": "<|video|>",
                "pad_token": "<|end_of_text|>",
            },
        )
        self.converted_tokenizer.image_token_id = self.converted_tokenizer.encode(
            self.converted_tokenizer.image_token, add_special_tokens=False
        )[0]
        self.converted_tokenizer.video_token_id = self.converted_tokenizer.encode(
            self.converted_tokenizer.video_token, add_special_tokens=False
        )[0]
        self.update_post_processor(self.converted_tokenizer)
        # finer special_tokens_map.json
        self.converted_tokenizer._bos_token = BOS_ADDED_TOKEN
        self.converted_tokenizer._eos_token = EOT_ADDED_TOKEN

    # We can't do this while building the tokenizer because we have no easy access to the bos token id
    def update_post_processor(self, tokenizer):
        tokenizer._tokenizer.post_processor = processors.Sequence(
            [
                processors.ByteLevel(trim_offsets=False),
                processors.TemplateProcessing(
                    single="<|begin_of_text|> $A",
                    pair="<|begin_of_text|>:0 $A:0 <|begin_of_text|>:1 $B:1",
                    special_tokens=[
                        (
                            "<|begin_of_text|>",
                            tokenizer.convert_tokens_to_ids("<|begin_of_text|>"),
                        ),
                    ],
                ),
            ]
        )


def write_tokenizer(
    tokenizer_path,
    input_tokenizer_path,
    special_tokens=None,
    params=None,
    push_to_hub=False,
):
    print("Converting the tokenizer.")
    tokenizer_class = LlamaTokenizer if LlamaTokenizerFast is None else LlamaTokenizerFast
    context_length = params["model"]["max_seqlen"]
    tokenizer = Llama3Converter(
        input_tokenizer_path,
        special_tokens,
        context_length,
    ).converted_tokenizer

    tokenizer.image_token_id = tokenizer.encode(tokenizer.image_token, add_special_tokens=False)[0]
    processor_config = {
        "pooling_ratio": params["model"]["pooling_ratio"],
        "patch_size": params["model"]["vision_model"]["patch_size"],
        "processor_class": "PerceptionLMProcessor",
    }
    tile_size = params["model"]["vision_model"]["image_size"]

    image_preprocessor_config = {
        "image_processor_type": "PerceptionLMImageProcessorFast",
        "vision_input_type": params["data"]["vision_input_type"],
        "tile_size": tile_size,
        "max_num_tiles": params["data"]["max_num_tiles"],
        "max_frame_tiles": 1,
        "size": {"height": tile_size, "width": tile_size},
        "do_resize": True,
        "do_rescale": True,
        "do_normalize": True,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
    }
    image_preprocessor = PerceptionLMImageProcessorFast(**image_preprocessor_config)
    video_preprocessor_config = {
        "video_processor_type": "PerceptionLMVideoProcessor",
        "size": {"height": tile_size, "width": tile_size},
    }
    video_preprocessor = PerceptionLMVideoProcessor(**video_preprocessor_config)
    processor = PerceptionLMProcessor(
        image_processor=image_preprocessor,
        video_processor=video_preprocessor,
        tokenizer=tokenizer,
        chat_template=CHAT_TEMPLATE,
        **processor_config,
    )

    if push_to_hub:
        print(f"Pushing a {tokenizer_class.__name__} to the Hub repo - {tokenizer_path}.")
        processor.push_to_hub(tokenizer_path, private=True, use_temp_dir=True)
    else:
        print(f"Saving a {tokenizer_class.__name__} to {tokenizer_path}.")
        processor.save_pretrained(tokenizer_path)
    return tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        help="Location of Llama weights, which contains tokenizer.model and model folders",
    )
    parser.add_argument(
        "--output_dir",
        help="Location to write HF model and tokenizer",
    )
    parser.add_argument(
        "--push_to_hub",
        help="Whether or not to push the model to the hub at `output_dir` instead of saving it locally.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--safe_serialization",
        action="store_true",
        default=True,
        help="Whether or not to save using `safetensors`.",
    )
    parser.add_argument(
        "--num_shards",
        default=None,
        type=int,
        help="The number of individual shards used for the model. Does not have to be the same as the number of consolidated_xx.pth",
    )
    parser.add_argument(
        "--special_tokens",
        default=None,
        type=list[str],
        help="The list of special tokens that should be added to the model.",
    )
    args = parser.parse_args()
    if args.special_tokens is None:
        # no special tokens by default
        args.special_tokens = DEFAULT_SPECIAL_TOKENS.get("perception_lm", [])

    params = read_json(os.path.join(args.input_dir, "params.json"))

    spm_path = os.path.join(args.input_dir, "tokenizer.model")
    tokenizer = write_tokenizer(
        args.output_dir,
        spm_path,
        special_tokens=args.special_tokens,
        params=params,
        push_to_hub=args.push_to_hub,
    )
    write_model(
        model_path=args.output_dir,
        input_base_path=args.input_dir,
        params=params,
        image_token_id=tokenizer.image_token_id,
        safe_serialization=args.safe_serialization,
        tokenizer=tokenizer,
        num_shards=args.num_shards,
        push_to_hub=args.push_to_hub,
    )


if __name__ == "__main__":
    main()
