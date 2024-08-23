# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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
"""Convert Nougat checkpoints using the original `nougat` library. URL:
https://github.com/facebookresearch/nougat/tree/main"""

import argparse

import torch
from huggingface_hub import hf_hub_download
from nougat import NougatModel
from nougat.dataset.rasterize import rasterize_paper
from nougat.utils.checkpoint import get_checkpoint
from PIL import Image

from transformers import (
    DonutSwinConfig,
    DonutSwinModel,
    MBartConfig,
    MBartForCausalLM,
    NougatImageProcessor,
    NougatProcessor,
    NougatTokenizerFast,
    VisionEncoderDecoderModel,
)


def get_configs(model):
    original_config = model.config

    encoder_config = DonutSwinConfig(
        image_size=original_config.input_size,
        patch_size=4,
        depths=original_config.encoder_layer,
        num_heads=[4, 8, 16, 32],
        window_size=original_config.window_size,
        embed_dim=128,
    )
    decoder_config = MBartConfig(
        is_decoder=True,
        is_encoder_decoder=False,
        add_cross_attention=True,
        decoder_layers=original_config.decoder_layer,
        max_position_embeddings=original_config.max_position_embeddings,
        vocab_size=len(
            model.decoder.tokenizer
        ),  # several special tokens are added to the vocab of XLMRobertaTokenizer, see repo on the hub (added_tokens.json)
        scale_embedding=True,
        add_final_layer_norm=True,
        tie_word_embeddings=False,
    )

    return encoder_config, decoder_config


# Copied from transformers.models.donut.convert_donut_to_pytorch.rename_key
def rename_key(name):
    if "encoder.model" in name:
        name = name.replace("encoder.model", "encoder")
    if "decoder.model" in name:
        name = name.replace("decoder.model", "decoder")
    if "patch_embed.proj" in name:
        name = name.replace("patch_embed.proj", "embeddings.patch_embeddings.projection")
    if "patch_embed.norm" in name:
        name = name.replace("patch_embed.norm", "embeddings.norm")
    if name.startswith("encoder"):
        if "layers" in name:
            name = "encoder." + name
        if "attn.proj" in name:
            name = name.replace("attn.proj", "attention.output.dense")
        if "attn" in name and "mask" not in name:
            name = name.replace("attn", "attention.self")
        if "norm1" in name:
            name = name.replace("norm1", "layernorm_before")
        if "norm2" in name:
            name = name.replace("norm2", "layernorm_after")
        if "mlp.fc1" in name:
            name = name.replace("mlp.fc1", "intermediate.dense")
        if "mlp.fc2" in name:
            name = name.replace("mlp.fc2", "output.dense")

        if name == "encoder.norm.weight":
            name = "encoder.layernorm.weight"
        if name == "encoder.norm.bias":
            name = "encoder.layernorm.bias"

    return name


# Copied from transformers.models.donut.convert_donut_to_pytorch.convert_state_dict
def convert_state_dict(orig_state_dict, model):
    for key in orig_state_dict.copy().keys():
        val = orig_state_dict.pop(key)

        if "qkv" in key:
            key_split = key.split(".")
            layer_num = int(key_split[3])
            block_num = int(key_split[5])
            dim = model.encoder.encoder.layers[layer_num].blocks[block_num].attention.self.all_head_size

            if "weight" in key:
                orig_state_dict[
                    f"encoder.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.query.weight"
                ] = val[:dim, :]
                orig_state_dict[
                    f"encoder.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.key.weight"
                ] = val[dim : dim * 2, :]
                orig_state_dict[
                    f"encoder.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.value.weight"
                ] = val[-dim:, :]
            else:
                orig_state_dict[
                    f"encoder.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.query.bias"
                ] = val[:dim]
                orig_state_dict[
                    f"encoder.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.key.bias"
                ] = val[dim : dim * 2]
                orig_state_dict[
                    f"encoder.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.value.bias"
                ] = val[-dim:]
        elif "attn_mask" in key or key in ["encoder.model.norm.weight", "encoder.model.norm.bias"]:
            # HuggingFace implementation doesn't use attn_mask buffer
            # and model doesn't use final LayerNorms for the encoder
            pass
        else:
            orig_state_dict[rename_key(key)] = val

    return orig_state_dict


def convert_nougat_checkpoint(model_tag, pytorch_dump_folder_path=None, push_to_hub=False):
    # load original model
    checkpoint_path = get_checkpoint(None, model_tag)
    original_model = NougatModel.from_pretrained(checkpoint_path)
    original_model.eval()

    # load HuggingFace model
    encoder_config, decoder_config = get_configs(original_model)
    encoder = DonutSwinModel(encoder_config)
    decoder = MBartForCausalLM(decoder_config)
    model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)
    model.eval()

    state_dict = original_model.state_dict()
    new_state_dict = convert_state_dict(state_dict, model)
    model.load_state_dict(new_state_dict)

    # verify results on PDF
    filepath = hf_hub_download(repo_id="ysharma/nougat", filename="input/nougat.pdf", repo_type="space")
    images = rasterize_paper(pdf=filepath, return_pil=True)
    image = Image.open(images[0])

    tokenizer_file = checkpoint_path / "tokenizer.json"
    tokenizer = NougatTokenizerFast(tokenizer_file=str(tokenizer_file))
    tokenizer.pad_token = "<pad>"
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"
    tokenizer.unk_token = "<unk>"
    tokenizer.model_max_length = original_model.config.max_length

    size = {"height": original_model.config.input_size[0], "width": original_model.config.input_size[1]}
    image_processor = NougatImageProcessor(
        do_align_long_axis=original_model.config.align_long_axis,
        size=size,
    )
    processor = NougatProcessor(image_processor=image_processor, tokenizer=tokenizer)

    # verify pixel_values
    pixel_values = processor(image, return_tensors="pt").pixel_values
    original_pixel_values = original_model.encoder.prepare_input(image).unsqueeze(0)

    assert torch.allclose(original_pixel_values, pixel_values)

    # verify patch embeddings
    original_patch_embed = original_model.encoder.model.patch_embed(pixel_values)
    patch_embeddings, _ = model.encoder.embeddings(pixel_values)
    assert torch.allclose(original_patch_embed, patch_embeddings)

    # verify encoder hidden states
    original_last_hidden_state = original_model.encoder(pixel_values)
    last_hidden_state = model.encoder(pixel_values).last_hidden_state
    assert torch.allclose(original_last_hidden_state, last_hidden_state, atol=1e-2)

    # NOTE original model does not use tied weights for embeddings of decoder
    original_embeddings = original_model.decoder.model.model.decoder.embed_tokens
    embeddings = model.decoder.model.decoder.embed_tokens
    assert torch.allclose(original_embeddings.weight, embeddings.weight, atol=1e-3)

    # verify decoder hidden states
    prompt = "hello world"
    decoder_input_ids = original_model.decoder.tokenizer(
        prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids
    decoder_attention_mask = torch.ones_like(decoder_input_ids)
    original_logits = original_model(
        image_tensors=pixel_values, decoder_input_ids=decoder_input_ids, attention_mask=decoder_attention_mask
    ).logits
    logits = model(
        pixel_values,
        decoder_input_ids=decoder_input_ids[:, :-1],
        decoder_attention_mask=decoder_attention_mask[:, :-1],
    ).logits
    assert torch.allclose(original_logits, logits, atol=1e-3)

    # verify generation
    outputs = model.generate(
        pixel_values,
        min_length=1,
        max_length=30,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
        bad_words_ids=[
            [tokenizer.unk_token_id],
        ],
        return_dict_in_generate=True,
        do_sample=False,
    )
    generated = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)[0]

    if model_tag == "0.1.0-base":
        expected_generation = "# Nougat: Neural Optical Understanding for Academic Documents\n\nLukas Blecher\n\nCorrespondence to: lblec"
    elif model_tag == "0.1.0-small":
        expected_generation = (
            "# Nougat: Neural Optical Understanding for Academic Documents\n\nLukas Blecher\n\nCorrespondence to: lble"
        )
    else:
        raise ValueError(f"Unexpected model tag: {model_tag}")

    assert generated == expected_generation
    print("Looks ok!")

    if pytorch_dump_folder_path is not None:
        print(f"Saving model and processor to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        processor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        tag_to_name = {"0.1.0-base": "nougat-base", "0.1.0-small": "nougat-small"}
        model_name = tag_to_name[model_tag]

        model.push_to_hub(f"facebook/{model_name}")
        processor.push_to_hub(f"facebook/{model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_tag",
        default="0.1.0-base",
        required=False,
        type=str,
        choices=["0.1.0-base", "0.1.0-small"],
        help="Tag of the original model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        required=False,
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the converted model and processor to the 🤗 hub.",
    )

    args = parser.parse_args()
    convert_nougat_checkpoint(args.model_tag, args.pytorch_dump_folder_path, args.push_to_hub)
