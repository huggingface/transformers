import argparse
import os
import re

import mediapy
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file
from torch import nn

from transformers import (
    AutoModel,
    AutoTokenizer,
    VideoPrismConfig,
    VideoPrismTextConfig,
    VideoPrismVisionConfig,
)
from transformers.models.videoprism.modeling_videoprism import VideoPrismClipModel, VideoPrismVisionModel


torch.set_printoptions(precision=10)

# backbone refers to VideoPrismVisionModel, lvt (original name) refers to VideoPrismClipModel
COOMMON_CONFIG_PARAMS = {
    "backbone_base": {
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_attention_heads": 12,
        "num_frames": 16,
        "num_spatial_layers": 12,
        "num_temporal_layers": 4,
    },
    "backbone_large": {
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "num_attention_heads": 16,
        "num_frames": 8,
        "num_spatial_layers": 24,
        "num_temporal_layers": 4,
    },
    "lvt_base": {
        "vision_config": {
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_attention_heads": 12,
            "num_frames": 16,
            "num_spatial_layers": 12,
            "num_temporal_layers": 4,
            "num_auxiliary_layers": 2,
        },
        "text_config": {
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_attention_heads": 12,
            "num_text_layers": 12,
        },
    },
    "lvt_large": {
        "vision_config": {
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "num_attention_heads": 16,
            "num_frames": 8,
            "num_spatial_layers": 24,
            "num_temporal_layers": 4,
            "num_auxiliary_layers": 2,
        },
        "text_config": {
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "num_attention_heads": 16,
            "num_text_layers": 12,
        },
    },
}

SENTENCES = [
    [262, 266, 768, 267, 1376, 14293, 259],
    [262, 266, 768, 267, 2865, 259],
    [262, 266, 768, 267, 1376, 20682, 259],
    [262, 266, 768, 267, 1376, 289, 10691, 259],
    [262, 266, 768, 267, 4605, 259],
]

ORIGINAL_CHECKPOINTS = {
    "backbone_base": {
        "repo_id": "google/videoprism-base-f16r288",
        "filename": "flax_base_f16r288_repeated.npz",
        "new_checkpoint_name": "videoprism-base-f16r288",
    },
    "backbone_large": {
        "repo_id": "google/videoprism-large-f8r288",
        "filename": "flax_large_f8r288_repeated.npz",
        "new_checkpoint_name": "videoprism-large-f8r288",
    },
    "lvt_base": {
        "repo_id": "google/videoprism-lvt-base-f16r288",
        "filename": "flax_lvt_base_f16r288_repeated.npz",
        "new_checkpoint_name": "videoprism-lvt-base-f16r288",
    },
    "lvt_large": {
        "repo_id": "google/videoprism-lvt-large-f8r288",
        "filename": "flax_lvt_large_f8r288_repeated.npz",
        "new_checkpoint_name": "videoprism-lvt-large-f8r288",
    },
}

EXPECTED_OUTPUTS = {
    "backbone_base": torch.tensor(
        [
            [0.11648951, 0.4568253, 0.19288044],
            [0.28420594, -0.04224018, 0.377879],
            [0.24594213, -0.3914095, -0.30516925],
        ]
    ),
    "backbone_large": torch.tensor(
        [
            [0.39503154, 0.07308281, 0.21407786],
            [0.4963156, -0.02489206, 0.49198192],
            [-0.41461205, 0.24869855, 0.25285226],
        ]
    ),
    "lvt_base": {
        "vision": torch.tensor(
            [
                -0.01940615,
                -0.04830061,
                0.0069022,
                0.02915299,
                -0.05897291,
                0.02168823,
                -0.01471708,
                -0.00971614,
                -0.00220576,
            ]
        ),
        "text": torch.tensor(
            [
                [-0.00802545, 0.00931361, 0.01555958],
                [0.02245245, 0.00010197, -0.01073526],
                [-0.02258418, 0.00133927, -0.01555064],
                [0.01056228, 0.01835608, -0.01539922],
                [-0.00366718, 0.00370416, 0.00800336],
            ]
        ),
    },
    "lvt_large": {
        "vision": torch.tensor(
            [
                -0.00077759,
                0.00582959,
                -0.00158949,
                0.04192347,
                -0.01581791,
                0.02410023,
                -0.00364033,
                -0.02118852,
                0.00181754,
            ]
        ),
        "text": torch.tensor(
            [
                [0.00454123, -0.02623128, -0.00612541],
                [-0.00042687, -0.0018771, 0.01664249],
                [0.02318677, -0.02984732, 0.00270805],
                [-0.02054974, 0.00793169, 0.00964476],
                [-0.00214194, -0.02825877, 0.01981462],
            ]
        ),
    },
}

ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    # Vision Encoder
    r"params(/vision_encoder)?/patch_projection/linear/(bias|kernel)": r"video_model.vision_encoder.spatial_embeddings.patch_embeddings.projection.\2",
    r"params(/vision_encoder)?/(spatial|temporal)_pos_emb/emb_var": r"video_model.vision_encoder.\2_embeddings.position_embeddings",
    r"params(/vision_encoder)?/(spatial|temporal)_encoder/transformers_stack/x_layers/ff_layer/ffn_layer1/linear/(bias|kernel)": r"video_model.vision_encoder.\2_encoder.layer.intermediate.dense.\3",
    r"params(/vision_encoder)?/(spatial|temporal)_encoder/transformers_stack/x_layers/ff_layer/ffn_layer2/linear/(bias|kernel)": r"video_model.vision_encoder.\2_encoder.layer.output.dense.\3",
    r"params(/vision_encoder)?/(spatial|temporal)_encoder/transformers_stack/x_layers/ff_layer/layer_norm/(bias|scale)": r"video_model.vision_encoder.\2_encoder.layer.layernorm_after.\3",
    r"params(/vision_encoder)?/(spatial|temporal)_encoder/transformers_stack/x_layers/layer_norm/(bias|scale)": r"video_model.vision_encoder.\2_encoder.layer.layernorm_before.\3",
    r"params(/vision_encoder)?/(spatial|temporal)_encoder/transformers_stack/x_layers/self_attention/(key|post|query|value)/(b|w)": r"video_model.vision_encoder.\2_encoder.layer.attention.attention.\3.\4",
    r"params(/vision_encoder)?/(spatial|temporal)_ln/(bias|scale)": r"video_model.vision_encoder.layernorm\2.\3",
    # Auxiliary Encoder
    r"params/auxiliary_encoder/transformers_stack/x_layers/ff_layer/ffn_layer1/linear/(bias|kernel)": r"video_model.auxiliary_encoder.layer.intermediate.dense.\1",
    r"params/auxiliary_encoder/transformers_stack/x_layers/ff_layer/layer_norm/(bias|scale)": r"video_model.auxiliary_encoder.layer.layernorm_after.\1",
    r"params/auxiliary_encoder/transformers_stack/x_layers/layer_norm/(bias|scale)": r"video_model.auxiliary_encoder.layer.layernorm_before.\1",
    r"params/auxiliary_encoder/transformers_stack/x_layers/self_attention/(key|post|query|value)/(b|w)": r"video_model.auxiliary_encoder.layer.attention.attention.\1.\2",
    r"params/auxiliary_encoder/transformers_stack/x_layers/ff_layer/ffn_layer2/linear/(bias|kernel)": r"video_model.auxiliary_encoder.layer.output.dense.\1",
    # Attention Pooler
    r"params/contrastive_vision_pooler/pooling_attention/(query|key|value|post)/(b|w)": r"video_model.contrastive_vision_pooler.\1.\2",
    r"params/contrastive_vision_pooler/pooling_attention/per_dim_scale/per_dim_scale": r"video_model.contrastive_vision_pooler.per_dim_scale",
    r"params/contrastive_vision_pooler/pooling_attention_layer_norm/(bias|scale)": r"video_model.contrastive_vision_pooler.layernorm.\1",
    r"params/contrastive_vision_pooler/pooling_attention_query": r"video_model.contrastive_vision_pooler.pooling_attention_query",
    # Text Encoder
    r"params/text_encoder/cls_emb": r"text_model.cls_emb",
    r"params/text_encoder/token_emb/emb_var": r"text_model.token_embeddings.weight",
    r"params/text_encoder/unimodal_ln/(bias|scale)": r"text_model.layernorm.\1",
    r"params/text_encoder/unimodal_transformer/x_layers/ff_layer/ffn_layer1/linear/(bias|kernel)": r"text_model.text_encoder.layer.intermediate.dense.\1",
    r"params/text_encoder/unimodal_transformer/x_layers/ff_layer/ffn_layer2/linear/(bias|kernel)": r"text_model.text_encoder.layer.output.dense.\1",
    r"params/text_encoder/unimodal_transformer/x_layers/ff_layer/layer_norm/(bias|scale)": r"text_model.text_encoder.layer.layernorm_after.\1",
    r"params/text_encoder/unimodal_transformer/x_layers/layer_norm/(bias|scale)": r"text_model.text_encoder.layer.layernorm_before.\1",
    r"params/text_encoder/unimodal_transformer/x_layers/self_attention/(query|key|value|post)/(b|w)": r"text_model.text_encoder.layer.attention.attention.\1.\2",
}


def download_flax_weights(checkpoint_info):
    # Download the weights file
    file = hf_hub_download(repo_id=checkpoint_info["repo_id"], filename=checkpoint_info["filename"])
    state_dict = np.load(file)
    return state_dict


def transform_block_params(key, param, hidden_size):
    if re.fullmatch(
        r"params(/vision_encoder)?/(spatial|temporal|auxiliary|text)_encoder/(transformers_stack|unimodal_transformer)/x_layers/self_attention/(key|query|value)/w",
        key,
    ):
        new_param = param.reshape(hidden_size, -1).T

    elif re.fullmatch(
        r"params(/vision_encoder)?/(spatial|temporal|auxiliary|text)_encoder/(transformers_stack|unimodal_transformer)/x_layers/self_attention/post/w",
        key,
    ):
        new_param = param.reshape(hidden_size, -1)

    elif re.fullmatch(
        r"params(/vision_encoder)?/(spatial|temporal|auxiliary|text)_encoder/(transformers_stack|unimodal_transformer)/x_layers/self_attention/(key|post|query|value)/b",
        key,
    ):
        new_param = param.reshape(-1)

    elif re.fullmatch(
        r"params(/vision_encoder)?/(spatial|temporal|auxiliary|text)_encoder/(transformers_stack|unimodal_transformer)/x_layers/ff_layer/ffn_layer([12])/linear/kernel",
        key,
    ):
        new_param = param.T

    else:
        new_param = param

    return new_param


def transform_remaining_params(key, param, hidden_size):
    # Vision Encoder specific transformations
    if re.fullmatch(r"params(/vision_encoder)?/patch_projection/linear/kernel", key):
        # Hard-coded number of patches
        new_param = param.T.reshape(hidden_size, 1, 18, 18, 3).transpose(0, 4, 1, 2, 3)

    elif re.fullmatch(r"params(/vision_encoder)?/(spatial|temporal)_pos_emb/emb_var", key):
        new_param = np.expand_dims(param, 0)

    # Contrastive Vision Pooler specific transformations
    elif re.fullmatch(r"params/contrastive_vision_pooler/pooling_attention_query", key):
        new_param = param.reshape(1, 1, -1)

    elif re.fullmatch(r"params/contrastive_vision_pooler/pooling_attention/(query|key|value)/w", key):
        new_param = param.reshape(hidden_size, -1).T

    elif re.fullmatch(r"params/contrastive_vision_pooler/pooling_attention/post/w", key):
        new_param = param.reshape(hidden_size, -1)

    elif re.fullmatch(r"params/contrastive_vision_pooler/pooling_attention/(query|key|value|post)/b", key):
        new_param = param.reshape(-1)

    else:
        new_param = param

    return new_param


def convert_params(flax_state_dict, model_name):
    # Convert flax parameters to HF-Pytorch format
    new_state_dict = {}
    if "lvt" in model_name:
        vision_config = COOMMON_CONFIG_PARAMS[model_name]["vision_config"]
        hidden_size = vision_config["hidden_size"]
    else:
        config = COOMMON_CONFIG_PARAMS[model_name]
        hidden_size = config["hidden_size"]

    for key in flax_state_dict:
        for original_pattern, new_pattern in ORIGINAL_TO_CONVERTED_KEY_MAPPING.items():
            if re.fullmatch(original_pattern, key):
                try:
                    new_key = re.sub(original_pattern, new_pattern, key)
                except Exception as e:
                    print(f"Error processing key: {key}")
                    raise e

                # Additional substitutions
                new_key = re.sub(r"\.scale$", ".weight", new_key)
                new_key = re.sub(r"attention\.post", "output.dense", new_key)
                new_key = re.sub(r"contrastive_vision_pooler\.post", "contrastive_vision_pooler.projection", new_key)
                new_key = re.sub(r"\.b$", ".bias", new_key)
                new_key = re.sub(r"\.w$|\.kernel$", ".weight", new_key)
                new_key = re.sub(r"layernormspatial", "layernorm1", new_key)
                new_key = re.sub(r"layernormtemporal", "layernorm2", new_key)
                new_key = re.sub(r"vision_encoder", "backbone", new_key)

                if "lvt" not in model_name:
                    new_key = new_key.replace("video_model.backbone.", "")

                param = flax_state_dict[key]
                if "layer." in new_key and param.ndim > 1:
                    # Split weights and biases layerwise
                    for layer in range(param.shape[0]):
                        layer_key = new_key.replace("layer.", f"layer.{layer}.")
                        new_param = transform_block_params(key, param[layer], hidden_size)
                        new_state_dict[layer_key] = torch.tensor(new_param).contiguous()

                else:
                    # Transformation of non-layerwise parameters
                    new_param = transform_remaining_params(key, param, hidden_size)
                    new_state_dict[new_key] = torch.tensor(new_param).contiguous()

    # Last step is to add the buffer named "scale"
    if "lvt" in model_name:
        dim = int(vision_config["intermediate_size"] / vision_config["num_attention_heads"])
        r_softplus_0 = 1.442695041
        scale = torch.tensor(r_softplus_0 / (dim**0.5))
        softplus = nn.functional.softplus(new_state_dict["video_model.contrastive_vision_pooler.per_dim_scale"])
        new_state_dict["video_model.contrastive_vision_pooler.scale"] = (scale * softplus).contiguous()

    return new_state_dict


def read_and_preprocess_video(  # This function is from the original repo
    filename: str, target_num_frames: int, target_frame_size: tuple[int, int]
):
    """Reads and preprocesses a video."""

    frames = mediapy.read_video(filename)

    # Sample to target number of frames.
    frame_indices = np.linspace(0, len(frames), num=target_num_frames, endpoint=False, dtype=np.int32)
    frames = np.array([frames[i] for i in frame_indices])

    # Resize to target size.
    original_height, original_width = frames.shape[-3:-1]
    target_height, target_width = target_frame_size
    assert original_height * target_width == original_width * target_height, (
        "Currently does not support aspect ratio mismatch."
    )
    frames = mediapy.resize_video(frames, shape=target_frame_size)

    # Normalize pixel values to [0.0, 1.0].
    frames = mediapy.to_float01(frames)

    return frames


def get_tokenizer(checkpoint_name=None):
    TEXT_QUERY_CSV = "playing drums,sitting,playing flute,playing at playground,concert"  # @param {type: "string"}
    PROMPT_TEMPLATE = "a video of {}."

    text_queries = TEXT_QUERY_CSV.split(",")
    text_queries = [PROMPT_TEMPLATE.format(t) for t in text_queries]

    tokenizer = AutoTokenizer.from_pretrained("MHRDYN7/" + checkpoint_name)

    return tokenizer, text_queries


def pad_and_stack(input_ids_list, pad_token_id=0, max_length=None):
    """
    Pads a list of input ID tensors to the same length and stacks them into a single tensor.
    """
    if max_length is None:
        max_length = max(len(ids) for ids in input_ids_list)

    padded_tensors = []
    for i, ids in enumerate(input_ids_list):
        padded = ids + [pad_token_id] * (max_length - len(ids))
        padded_tensors.append(torch.tensor(padded, dtype=torch.long))

    return torch.stack(padded_tensors)


def ids_to_attention_mask(input_ids: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
    return (input_ids != pad_token_id).long()


@torch.no_grad()
def convert_videoprism_checkpoint(
    model_name="lvt_base",
    pytorch_dump_folder_path="checkpoints/",
    convert=False,
    load_model=True,
    from_pretrained=False,
    from_tokenizer=False,
    load_video=True,
    inference=True,
    upload=False,
):
    checkpoint = ORIGINAL_CHECKPOINTS[model_name]

    if "lvt" in model_name:
        vision_config = VideoPrismVisionConfig(**COOMMON_CONFIG_PARAMS[model_name]["vision_config"])
        text_config = VideoPrismTextConfig(**COOMMON_CONFIG_PARAMS[model_name]["text_config"])
    else:
        vision_config = VideoPrismVisionConfig(**COOMMON_CONFIG_PARAMS[model_name])

    checkpoint_name = checkpoint["new_checkpoint_name"]
    checkpoint_path = os.path.join(pytorch_dump_folder_path, f"{checkpoint_name}.safetensors")

    if convert:
        flax_checkpoint = download_flax_weights(checkpoint)
        hf_checkpoint = convert_params(flax_checkpoint, model_name)
        save_file(hf_checkpoint, checkpoint_path, metadata={"format": "safetensors"})

    if load_model:
        if not from_pretrained:
            model_config = vision_config if "lvt" not in model_name else VideoPrismConfig(text_config, vision_config)
            model = (
                VideoPrismVisionModel(model_config) if "lvt" not in model_name else VideoPrismClipModel(model_config)
            )

            model.config._attn_implementation = "eager"
            state_dict = load_file(checkpoint_path)
            model.load_state_dict(state_dict)
        else:
            model = AutoModel.from_pretrained("MHRDYN7/" + checkpoint_name)  # Hard-coded username of the contributer
            model.config._attn_implementation = "eager"
            model_config = model.config

    if load_video:
        VIDEO_FILE_PATH = "./src/transformers/models/videoprism/water_bottle_drumming.mp4"
        NUM_FRAMES = model_config.num_frames if "lvt" not in model_name else vision_config.num_frames
        FRAME_SIZE = 288
        frames = read_and_preprocess_video(
            VIDEO_FILE_PATH,
            target_num_frames=NUM_FRAMES,
            target_frame_size=[FRAME_SIZE, FRAME_SIZE],
        )

        input_vid = torch.tensor(frames).unsqueeze(0).permute(0, 1, 4, 2, 3)

    if inference:
        model.eval()
        if "lvt" not in model_name:
            outputs = model(input_vid)
            logits = outputs.last_hidden_state[0, :3, :3]
            assert torch.allclose(logits, EXPECTED_OUTPUTS[model_name], atol=1e-5), (
                "The converted model logits do not match the expected logits."
            )
            print("Inference successful and logits match expected outputs.")

        else:
            if from_tokenizer:
                tokenizer, text_queries = get_tokenizer(checkpoint_name=checkpoint_name)
                outputs = tokenizer(text_queries, max_length=64, padding="max_length", return_tensors="pt")
                input_ids, mask = outputs["input_ids"], outputs["attention_mask"]
            else:
                input_ids = pad_and_stack(SENTENCES, pad_token_id=0, max_length=64)
                mask = ids_to_attention_mask(input_ids)
            outputs = model(input_vid, input_ids, mask)
            video_logits = outputs.video_embeds[0, :9]
            text_logits = outputs.text_embeds[:, :3]
            assert torch.allclose(video_logits, EXPECTED_OUTPUTS[model_name]["vision"], atol=1e-5), (
                "The converted model video logits do not match the expected logits."
            )
            assert torch.allclose(text_logits, EXPECTED_OUTPUTS[model_name]["text"], atol=1e-5), (
                "The converted model text logits do not match the expected logits."
            )
            print("Inference successful and logits match expected outputs.")

    if upload:
        repo_id = f"MHRDYN7/{checkpoint_name}"
        model.push_to_hub(repo_id)
        print(f"Uploaded the model to the Hugging Face hub at {repo_id}.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="backbone_base",
        type=str,
        choices=ORIGINAL_CHECKPOINTS.keys(),
        help="Name of the model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default="./src/transformers/models/videoprism/checkpoints/",
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument(
        "--convert",
        default=False,
        type=bool,
        help="Whether to convert the original Flax checkpoint to Hugging Face format.",
    )
    parser.add_argument(
        "--load_model",
        default=True,
        type=bool,
        help="Whether to load the converted model for inference.",
    )
    parser.add_argument(
        "--from_pretrained",
        default=True,
        type=bool,
        help="Whether to load the model weights from the Hugging Face hub. Loads local checkpoint (not in cache dir) if False.",
    )
    parser.add_argument(
        "--from_tokenizer",
        default=True,
        type=bool,
        help="Whether to use AutoTokenizer from the Hugging Face hub. Uses custom input_ids if False.",
    )
    parser.add_argument(
        "--load_video",
        default=True,
        type=bool,
        help="Whether to load and preprocess the sample video for inference.",
    )
    parser.add_argument(
        "--inference",
        default=True,
        type=bool,
        help="Whether to run inference on the loaded model and compare outputs to expected outputs.",
    )
    parser.add_argument(
        "--upload",
        default=False,
        type=bool,
        help="Whether to upload the converted model to the Hugging Face hub.",
    )

    args = parser.parse_args()

    convert_videoprism_checkpoint(
        model_name=args.model_name,
        pytorch_dump_folder_path=args.pytorch_dump_folder_path,
        convert=args.convert,
        load_model=args.load_model,
        from_pretrained=args.from_pretrained,
        from_tokenizer=args.from_tokenizer,
        load_video=args.load_video,
        inference=args.inference,
        upload=args.upload,
    )


if __name__ == "__main__":
    main()
