import argparse

import torch
from datasets import load_dataset
from PIL import Image

from donut import DonutModel
from transformers import MBartConfig, MBartForCausalLM, SwinConfig, SwinModel, VisionEncoderDecoderModel


def get_configs():
    encoder_config = SwinConfig(
        image_size=[1280, 960],
        patch_size=4,
        depths=[2, 2, 14, 2],
        num_heads=[4, 8, 16, 32],
        window_size=10,
        embed_dim=128,
    )
    decoder_config = MBartConfig(
        is_decoder=True,
        is_encoder_decoder=False,
        add_cross_attention=True,
        decoder_layers=4,
        max_position_embeddings=768,
        vocab_size=57580,  # several special tokens are added to the vocab of XLMRobertaTokenizer, see repo on the hub (added_tokens.json)
        scale_embedding=True,
        add_final_layer_norm=True,
    )

    return encoder_config, decoder_config


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
        elif "attn_mask" in key:
            # TODO check attn_mask buffers
            pass
        else:
            orig_state_dict[rename_key(key)] = val

    return orig_state_dict


def convert_swin_checkpoint(checkpoint_path, pytorch_dump_folder_path):
    encoder_config, decoder_config = get_configs()
    encoder = SwinModel(encoder_config)
    decoder = MBartForCausalLM(decoder_config)
    model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)
    model.eval()

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    new_state_dict = convert_state_dict(state_dict, model)
    model.load_state_dict(new_state_dict)

    # load original model
    original_model = DonutModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")

    # verify results on scanned document
    dataset = load_dataset("hf-internal-testing/fixtures_docvqa")
    image = Image.open(dataset["test"][0]["file"]).convert("RGB")

    pixel_values = original_model.encoder.prepare_input(image).unsqueeze(0)

    task_prompt = "<s_docvqa><s_question>{user_input}</s_question><s_answer>"
    question = "When is the coffee break?"
    user_prompt = task_prompt.replace("{user_input}", question)

    last_hidden_state = original_model.encoder(pixel_values)

    print("First values of last_hidden_state:", last_hidden_state[0, :3, :3])

    outputs = model.encoder(pixel_values, output_hidden_states=True)
    print("Shape of last hidden state HuggingFace one:", outputs.last_hidden_state[0, :3, :3])

    # TODO assert outputs
    # assert torch.allclose(timm_outs, hf_outs, atol=1e-3)

    if pytorch_dump_folder_path is not None:
        print(f"Saving model and feature extractor to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        # feature_extractor.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--checkpoint_path",
        default="/Users/nielsrogge/Documents/Donut/pytorch_model.bin",
        type=str,
        help="Path to the original checkpoint you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        required=False,
        type=str,
        help="Path to the output PyTorch model directory.",
    )

    args = parser.parse_args()
    convert_swin_checkpoint(args.checkpoint_path, args.pytorch_dump_folder_path)
