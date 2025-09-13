from collections import OrderedDict
from torch import nn
import mediapy
import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download
from safetensors.torch import load_file, save_file

from transformers import VideoPrismConfig, VideoPrismTokenizer, VideoPrismTokenizerFast
from transformers import T5TokenizerFast
from transformers.models.videoprism.modeling_videoprism import VideoPrismClipModel, VideoPrismFactorizedEncoderModel


def get_checkpoint_info(model_type="backbone", model_size="base"):
    backbone_base = {
        "model_type": "backbone",
        "model_size": "base",
        "id": "f16r288",
        "repo_id": "google/videoprism-base-f16r288",
        "filename": "flax_base_f16r288_repeated.npz",
        "config": {
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_attention_heads": 12,
            "num_frames": 16,
            "num_spatial_layers": 12,
            "num_temporal_layers": 4,
        },
    }
    backbone_large = {
        "model_type": "backbone",
        "model_size": "large",
        "id": "f8r288",
        "repo_id": "google/videoprism-large-f8r288",
        "filename": "flax_large_f8r288_repeated.npz",
        "config": {
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "num_attention_heads": 16,
            "num_frames": 8,
            "num_spatial_layers": 24,
            "num_temporal_layers": 4,
        },
    }
    lvt_base = {
        "model_type": "lvt",
        "model_size": "base",
        "id": "f16r288",
        "repo_id": "google/videoprism-lvt-base-f16r288",
        "filename": "flax_lvt_base_f16r288_repeated.npz",
        "config": {
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_attention_heads": 12,
            "num_frames": 16,
            "num_spatial_layers": 12,
            "num_temporal_layers": 4,
            "num_auxiliary_layers": 2,
            "num_unimodal_layers": 12,
        },
    }
    lvt_large = {
        "model_type": "lvt",
        "model_size": "large",
        "id": "f8r288",
        "repo_id": "google/videoprism-lvt-large-f8r288",
        "filename": "flax_lvt_large_f8r288_repeated.npz",
        "config": {
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "num_attention_heads": 16,
            "num_frames": 8,
            "num_spatial_layers": 24,
            "num_temporal_layers": 4,
            "num_auxiliary_layers": 2,
            "num_unimodal_layers": 12,
        },
    }
    if model_type == "backbone":
        return backbone_base if model_size == "base" else backbone_large

    elif model_type == "lvt":
        return lvt_base if model_size == "base" else lvt_large


# ? download and load the orginal weights
def download_weights(checkpoint_info):
    # Download the weights file
    file = hf_hub_download(repo_id=checkpoint_info["repo_id"], filename=checkpoint_info["filename"])
    state_dict = np.load(file)
    return state_dict


checkpoint_dict = {}


def transform_state_encoder_block(state, checkpoint_info, modes):
    # ? spatial encoder blocks
    new_state = OrderedDict()
    if checkpoint_info["model_type"] == "backbone":
        extra = ""
    elif checkpoint_info["model_type"] == "lvt":
        extra = "/vision_encoder"
    spatial_prefix = f"params{extra}/spatial_encoder/transformers_stack/x_layers"
    temporal_prefix = f"params{extra}/temporal_encoder/transformers_stack/x_layers"
    auxiliary_prefix = "params/auxiliary_encoder/transformers_stack/x_layers"
    unimodal_prefix = "params/text_encoder/unimodal_transformer/x_layers"
    # ?                 params/text_encoder/unimodal_transformer/x_layers/layer_norm/scale
    spatial = (
        "spatial_encoder.layer" if checkpoint_info["model_type"] == "backbone" else "backbone.spatial_encoder.layer"
    )
    temporal = (
        "temporal_encoder.layer" if checkpoint_info["model_type"] == "backbone" else "backbone.temporal_encoder.layer"
    )
    auxiliary = "auxiliary_encoder.layer"
    unimodal = "text_encoder.unimodal_encoder.layer"

    hidden_size = checkpoint_info["config"]["hidden_size"]

    for mode in modes:
        if mode == "spatial":
            prefix = spatial_prefix
            layer = spatial
            num_layers = checkpoint_info["config"]["num_spatial_layers"]
        elif mode == "temporal":
            prefix = temporal_prefix
            layer = temporal
            num_layers = checkpoint_info["config"]["num_temporal_layers"]
        elif mode == "auxiliary":
            prefix = auxiliary_prefix
            layer = auxiliary
            num_layers = checkpoint_info["config"]["num_auxiliary_layers"]
        elif mode == "unimodal":
            prefix = unimodal_prefix
            layer = unimodal
            num_layers = checkpoint_info["config"]["num_unimodal_layers"]

        for i in range(num_layers):
            # ? attention LN
            new_state[f"{layer}.{i}.layernorm_before.weight"] = state[f"{prefix}/layer_norm/scale"][i]  # ? [768]
            new_state[f"{layer}.{i}.layernorm_before.bias"] = state[f"{prefix}/layer_norm/bias"][i]  # ? [768]
            # ? attention
            new_state[f"{layer}.{i}.attention.attention.query.weight"] = (
                state[f"{prefix}/self_attention/query/w"][i].reshape(hidden_size, -1).T
            )  # ? [768, 12, 64] -> [768, 768]
            new_state[f"{layer}.{i}.attention.attention.query.bias"] = state[f"{prefix}/self_attention/query/b"][
                i
            ].reshape(-1)
            new_state[f"{layer}.{i}.attention.attention.key.weight"] = (
                state[f"{prefix}/self_attention/key/w"][i].reshape(hidden_size, -1).T
            )  # ? [768, 12, 64] -> [768, 768]
            new_state[f"{layer}.{i}.attention.attention.key.bias"] = state[f"{prefix}/self_attention/key/b"][
                i
            ].reshape(-1)
            new_state[f"{layer}.{i}.attention.attention.value.weight"] = (
                state[f"{prefix}/self_attention/value/w"][i].reshape(hidden_size, -1).T
            )  # ? [768, 12, 64] -> [768, 768]
            new_state[f"{layer}.{i}.attention.attention.value.bias"] = state[f"{prefix}/self_attention/value/b"][
                i
            ].reshape(-1)
            new_state[f"{layer}.{i}.attention.output.dense.weight"] = state[f"{prefix}/self_attention/post/w"][
                i
            ].reshape(hidden_size, -1)  # ? [768, 12, 64] -> [768, 768]
            new_state[f"{layer}.{i}.attention.output.dense.bias"] = state[f"{prefix}/self_attention/post/b"][
                i
            ].reshape(-1)
            # ? MLP LN
            new_state[f"{layer}.{i}.layernorm_after.weight"] = state[f"{prefix}/ff_layer/layer_norm/scale"][
                i
            ]  # ? [768]
            new_state[f"{layer}.{i}.layernorm_after.bias"] = state[f"{prefix}/ff_layer/layer_norm/bias"][i]  # ? [768]
            # ? MLP
            new_state[f"{layer}.{i}.intermediate.dense.weight"] = state[f"{prefix}/ff_layer/ffn_layer1/linear/kernel"][
                i
            ].T  # ? [768, 3072] -> [3072, 768]
            new_state[f"{layer}.{i}.intermediate.dense.bias"] = state[f"{prefix}/ff_layer/ffn_layer1/linear/bias"][i]
            new_state[f"{layer}.{i}.output.dense.weight"] = state[f"{prefix}/ff_layer/ffn_layer2/linear/kernel"][
                i
            ].T  # ? [768, 3072] -> [3072, 768]
            new_state[f"{layer}.{i}.output.dense.bias"] = state[f"{prefix}/ff_layer/ffn_layer2/linear/bias"][i]
    return new_state


def transform_state(state, checkpoint_info):
    hidden_size = checkpoint_info["config"]["hidden_size"]
    new_state = OrderedDict()
    if checkpoint_info["model_type"] == "backbone":
        extra = ""
        backbone = ""
    elif checkpoint_info["model_type"] == "lvt":
        extra = "/vision_encoder"
        backbone = "backbone."
    # ? patch embeds
    new_state[f"{backbone}spatial_embeddings.patch_embeddings.projection.weight"] = (
        state[f"params{extra}/patch_projection/linear/kernel"]
        .T.reshape(hidden_size, 1, 18, 18, 3)
        .transpose(0, 4, 1, 2, 3)
    )  # ? [972, 768] -> [768, 3, 1, 18, 18]
    new_state[f"{backbone}spatial_embeddings.patch_embeddings.projection.bias"] = state[
        f"params{extra}/patch_projection/linear/bias"
    ]  # ? [768]
    # ? Spatial/temporal pos embeds
    new_state[f"{backbone}spatial_embeddings.spatial_pos_emb"] = np.expand_dims(
        state[f"params{extra}/spatial_pos_emb/emb_var"], axis=0
    )  # ? [256, 768] -> [1, 256, 768]
    new_state[f"{backbone}temporal_embeddings.temporal_pos_emb"] = np.expand_dims(
        state[f"params{extra}/temporal_pos_emb/emb_var"], axis=0
    )  # ? [256, 768] -> [1, 256, 768]
    # ? 'pre' layernorm
    new_state[f"{backbone}layernorm1.weight"] = state[f"params{extra}/spatial_ln/scale"]  # ? all 768
    new_state[f"{backbone}layernorm1.bias"] = state[f"params{extra}/spatial_ln/bias"]
    new_state[f"{backbone}layernorm2.weight"] = state[f"params{extra}/temporal_ln/scale"]
    new_state[f"{backbone}layernorm2.bias"] = state[f"params{extra}/temporal_ln/bias"]

    new_state.update(transform_state_encoder_block(state, checkpoint_info, ["spatial", "temporal"]))

    if checkpoint_info["model_type"] == "backbone":
        checkpoint = {k: torch.tensor(v).contiguous() for k, v in new_state.items()}

        path = f"videoprism_{checkpoint_info['model_size']}_{checkpoint_info['id']}.safetensors"
        save_file(checkpoint, path, metadata={"format": "safetensors"})
        print("file saved")

    elif checkpoint_info["model_type"] == "lvt":
        # ? Auxiliary layers
        new_state.update(transform_state_encoder_block(state, checkpoint_info, ["auxiliary"]))

        pooler_prefix = "params/contrastive_vision_pooler"
        unimodal_prefix = "params/text_encoder"
        pooler_layer = "contrastive_vision_pooler"
        unimodal_layer = "text_encoder"
        # ? attention LN
        new_state[f"{pooler_layer}.layernorm.weight"] = state[
            f"{pooler_prefix}/pooling_attention_layer_norm/scale"
        ]  # ? [768]
        new_state[f"{pooler_layer}.layernorm.bias"] = state[
            f"{pooler_prefix}/pooling_attention_layer_norm/bias"
        ]  # ? [768]
        # ? attention
        new_state[f"{pooler_layer}.pooling_attention_query"] = state[
            f"{pooler_prefix}/pooling_attention_query"
        ].reshape(1, 1, -1)
        new_state[f"{pooler_layer}.per_dim_scale.per_dim_scale"] = state[
            f"{pooler_prefix}/pooling_attention/per_dim_scale/per_dim_scale"
        ]
        new_state[f"{pooler_layer}.query.weight"] = (
            state[f"{pooler_prefix}/pooling_attention/query/w"].reshape(hidden_size, -1).T
        )  # ? [768, 12, 64] -> [768, 768]
        new_state[f"{pooler_layer}.query.bias"] = state[f"{pooler_prefix}/pooling_attention/query/b"].reshape(-1)
        new_state[f"{pooler_layer}.key.weight"] = (
            state[f"{pooler_prefix}/pooling_attention/key/w"].reshape(hidden_size, -1).T
        )  # ? [768, 12, 64] -> [768, 768]
        new_state[f"{pooler_layer}.key.bias"] = state[f"{pooler_prefix}/pooling_attention/key/b"].reshape(-1)
        new_state[f"{pooler_layer}.value.weight"] = (
            state[f"{pooler_prefix}/pooling_attention/value/w"].reshape(hidden_size, -1).T
        )  # ? [768, 12, 64] -> [768, 768]
        new_state[f"{pooler_layer}.value.bias"] = state[f"{pooler_prefix}/pooling_attention/value/b"].reshape(-1)
        new_state[f"{pooler_layer}.projection.weight"] = state[f"{pooler_prefix}/pooling_attention/post/w"].reshape(
            hidden_size, -1
        )  # ? [768, 12, 64] -> [768, 768]
        new_state[f"{pooler_layer}.projection.bias"] = state[f"{pooler_prefix}/pooling_attention/post/b"].reshape(-1)

        # ? text encoder
        new_state[f"{unimodal_layer}.cls_emb"] = state[f"{unimodal_prefix}/cls_emb"]  # ? (1, 1, 768)
        new_state[f"{unimodal_layer}.token_embeddings.weight"] = state[
            f"{unimodal_prefix}/token_emb/emb_var"
        ]  # ? (32000, 768)
        new_state[f"{unimodal_layer}.layernorm.weight"] = state[f"{unimodal_prefix}/unimodal_ln/scale"]  # ? [768]
        new_state[f"{unimodal_layer}.layernorm.bias"] = state[f"{unimodal_prefix}/unimodal_ln/bias"]  # ? [768]
        new_state.update(transform_state_encoder_block(state, checkpoint_info, ["unimodal"]))

        checkpoint = {k: torch.tensor(v).contiguous() for k, v in new_state.items()}
        path = f"videoprism_lvt_{checkpoint_info['model_size']}_{checkpoint_info['id']}.safetensors"

        save_file(checkpoint, path, metadata={"format": "safetensors"})
        print("file saved")

    else:
        raise ValueError(f"Unsupported model type: {checkpoint_info['model_type']}")


def prepare_video():   # ? borrowed from vivit convert_weights, but not helpful here
    file = hf_hub_download(
        repo_id="hf-internal-testing/spaghetti-video", filename="eating_spaghetti_32_frames.npy", repo_type="dataset"
    )
    video = np.load(file)
    return list(video)


def read_and_preprocess_video(  # This function from the original code
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


def pad_and_stack(input_ids_list, pad_token_id=0, max_length=None):
    """
    Pads a list of input ID tensors to the same length and stacks them into a single tensor.

    Args:
        input_ids_list (List[List[int]]): List of token ID sequences.
        pad_token_id (int): Token ID used for padding.
        max_length (int, optional): Desired sequence length. If None, uses max length in input.
        save_dir (str, optional): Directory to save each sentence's original ID list as .pt files.

    Returns:
        torch.Tensor: Padded and stacked tensor of shape [num_sentences, max_length].
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


def prepare_texts():
    tokenizer = VideoPrismTokenizerFast(

    legacy=False,
    vocab_file="./sentencepiece.model",
    unk_token="<unk>",
    pad_token="<pad>",
    eos_token="</s>",
    bos_token="<s>",  # Optional, if your model uses BOS
    )

    TEXT_QUERY_CSV = 'playing drums,sitting,playing flute,playing at playground,concert'  # @param {type: "string"}
    PROMPT_TEMPLATE = 'a video of {}.'

    text_queries = TEXT_QUERY_CSV.split(',')
    text_queries = [PROMPT_TEMPLATE.format(t) for t in text_queries]

    outputs = tokenizer(text_queries, max_length=64, padding="max_length", truncation=True, return_tensors="pt")
    return outputs["input_ids"], outputs["attention_mask"]

def convert(
    model_type="backbone",
    model_size="base",
    convert=False,
    upload=False,
    load_model=True,
    load_video=True,
    inference=True,
):
    # Load the weights
    checkpoint_info = get_checkpoint_info(model_type, model_size)

    if checkpoint_info["model_type"] == "backbone":
        path = f"videoprism_{checkpoint_info['model_size']}_{checkpoint_info['id']}.safetensors"
    elif checkpoint_info["model_type"] == "lvt":
        path = f"videoprism_lvt_{checkpoint_info['model_size']}_{checkpoint_info['id']}.safetensors"

    if convert:
        state_dict = download_weights(checkpoint_info)
        # for k, v in state_dict.items():
        #     shape = v.shape
        #     new_shape = ()
        #     for i in range(len(shape)):
        #         new_shape += (shape[i]-1,)
        #     print(f"Key: {k}, Value shape: {shape}, values: {v[new_shape]} ")
        # print(state_dict["params/text_encoder/token_emb/emb_var"][:5,:5])

        # first = state_dict["params/patch_projection/linear/bias"]
        # transform_state(state_dict, checkpoint_info)

    if upload:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=path,
            repo_id="MHRDYN7/videoprism-base",
            repo_type="model",
        )
        print("uploaded")

    if load_model:
        config = VideoPrismConfig(**checkpoint_info["config"])
        model = VideoPrismFactorizedEncoderModel(config) if checkpoint_info["model_type"] == "backbone" else VideoPrismClipModel(config)

        # try:
        state_dict = load_file(path)
        # except:
        #     hf_hub_download(repo_id="MHRDYN7/videoprism-base", filename=path, local_dir="./")
        #     state_dict = load_file(path)
            # raise ValueError("File not found, please download first")

        # for lvt

        # key_list = list(state_dict.keys())
        # for k in key_list:
        #     # shape = v.shape
        #     # print(f"Key: {k}, Value shape: {shape}")
        #     if k.startswith("backbone") or k.startswith("auxiliary_encoder") or k.startswith("contrastive_vision_pooler"):
        #         state_dict[f"video_model.{k}"] = state_dict.pop(k)

        #     if k.startswith("text_encoder"):
        #         k_new = k.replace("text_encoder", "text_model")
        #         state_dict[f"{k_new}"] = state_dict.pop(k)
        
        # state_dict["video_model.backbone.spatial_embeddings.position_embeddings"] = state_dict.pop("video_model.backbone.spatial_embeddings.spatial_pos_emb")
        # state_dict["video_model.backbone.temporal_embeddings.position_embeddings"] = state_dict.pop("video_model.backbone.temporal_embeddings.temporal_pos_emb")
        
        
        # For video encoder
        # state_dict["spatial_embeddings.position_embeddings"] = state_dict.pop("spatial_embeddings.spatial_pos_emb")
        # state_dict["temporal_embeddings.position_embeddings"] = state_dict.pop("temporal_embeddings.temporal_pos_emb")    
          
        # for scale buffer

        # self.dim = int(config.intermediate_size / config.num_attention_heads)
        # self.per_dim_scale = nn.Parameter(torch.zeros(self.dim))
        # r_softplus_0 = 1.442695041
        # _scale = torch.tensor(r_softplus_0 / (self.dim**0.5))

        # dim = int(checkpoint_info["config"]["intermediate_size"] / checkpoint_info["config"]["num_attention_heads"])
        # r_softplus_0 = 1.442695041

        # scale = torch.tensor(r_softplus_0 / (dim**0.5))
        # softplus = nn.functional.softplus(state_dict["video_model.contrastive_vision_pooler.per_dim_scale.per_dim_scale"])
        # scale = scale * softplus
        # state_dict["video_model.contrastive_vision_pooler.per_dim_scale.scale"] = scale
        
        model.load_state_dict(state_dict)
        print("all good")

    if load_video:
        VIDEO_FILE_PATH = "./src/transformers/models/videoprism/water_bottle_drumming.mp4"
        NUM_FRAMES = checkpoint_info["config"]["num_frames"]  # ? 16 for base, 8 for large
        FRAME_SIZE = 288
        frames = read_and_preprocess_video(
            VIDEO_FILE_PATH,
            target_num_frames=NUM_FRAMES,
            target_frame_size=[FRAME_SIZE, FRAME_SIZE],
        )

        input_vid = torch.tensor(frames).unsqueeze(0).permute(0, 1, 4, 2, 3)  # ? (1, 16, 3, 288, 288)

        # inputs = prepare_video()
        # frame_indices = np.linspace(
        #     0, len(inputs), num=16, endpoint=False, dtype=np.int32
        # )
        # inputs = np.array([inputs[i] for i in frame_indices])
        # inputs = VideoPrismVideoProcessor()(inputs, return_tensors="pt")
        # ? (1, 16, 3, 288, 288) is the needed input shape

    if inference:
        with torch.no_grad():
            if checkpoint_info["model_type"] == "backbone":
                outputs = model(input_vid)
                backbone_base_expected_tensor = torch.tensor(
                    [
                        [0.11648951, 0.4568253, 0.19288044],
                        [0.28420594, -0.04224018, 0.377879],
                        [0.24594213, -0.3914095, -0.30516925],
                    ]
                )
                backbone_large_expected_tensor = torch.tensor(
                    [
                        [0.39503154, 0.07308281, 0.21407786],
                        [0.4963156, -0.02489206, 0.49198192],
                        [-0.41461205, 0.24869855, 0.25285226],
                    ]
                )

                expected_tensor = (
                    backbone_base_expected_tensor if model_size == "base" else backbone_large_expected_tensor
                )
                print(outputs.last_hidden_state.shape)
                print(outputs.last_hidden_state[0, :3, :3])
                assert torch.allclose(outputs.last_hidden_state[0, :3, :3], expected_tensor, atol=1e-5), (
                    "Output does not match expected tensor."
                )
                print("Inference successful, output matches expected tensor.")
                path = f"videoprism_{checkpoint_info['model_size']}_{checkpoint_info['id']}.safetensors"
                print(path)
                save_file(state_dict, path, metadata={"format": "safetensors"})
                print("done")

            elif checkpoint_info["model_type"] == "lvt":
                sentences = [
                    [262, 266, 768, 267, 1376, 14293, 259],
                    [262, 266, 768, 267, 2865, 259],
                    [262, 266, 768, 267, 1376, 20682, 259],
                    [262, 266, 768, 267, 1376, 289, 10691, 259],
                    [262, 266, 768, 267, 4605, 259],
                ]
                # input_ids = pad_and_stack(sentences, pad_token_id=0, max_length=64)
                # mask = ids_to_attention_mask(input_ids)


                # print(input_vid[0, -1, 0, :3, :3])
                input_ids, mask = prepare_texts()
                
                outputs = model(input_vid, input_ids, mask)
                
                lvt_video_base_expected_tensor = torch.tensor(
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
                )
                lvt_video_large_expected_tensor = torch.tensor(
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
                )
                lvt_text_base_expected_tensor = torch.tensor(
                    [
                        [-0.00802545, 0.00931361, 0.01555958],
                        [0.02245245, 0.00010197, -0.01073526],
                        [-0.02258418, 0.00133927, -0.01555064],
                        [0.01056228, 0.01835608, -0.01539922],
                        [-0.00366718, 0.00370416, 0.00800336],
                    ]
                )
                lvt_text_large_expected_tensor = torch.tensor(
                    [
                        [0.00454123, -0.02623128, -0.00612541],
                        [-0.00042687, -0.0018771, 0.01664249],
                        [0.02318677, -0.02984732, 0.00270805],
                        [-0.02054974, 0.00793169, 0.00964476],
                        [-0.00214194, -0.02825877, 0.01981462],
                    ]
                )
                if checkpoint_info["model_size"] == "base":

                    path = f"videoprism_lvt_{checkpoint_info['model_size']}_{checkpoint_info['id']}.safetensors"
                    # save_file(state_dict, path, metadata={"format": "safetensors"})
                    assert torch.allclose(outputs.video_embeds[:, :9], lvt_video_base_expected_tensor, atol=1e-5), (
                        "Video output does not match expected tensor."
                    )
                    assert torch.allclose(outputs.text_embeds[:, :3], lvt_text_base_expected_tensor, atol=1e-5), (
                        "Text output does not match expected tensor."
                    )
                    print("Inference successful, output matches expected tensor.")
                elif checkpoint_info["model_size"] == "large":                    
                    
                    assert torch.allclose(outputs.video_embeds[:, :9], lvt_video_large_expected_tensor, atol=1e-5), (
                        "Video output does not match expected tensor."
                    )
                    print("video ok")                    
                    assert torch.allclose(outputs.text_embeds[:, :3], lvt_text_large_expected_tensor, atol=1e-5), (
                        "Text output does not match expected tensor."
                    )
                    print("Inference successful, output matches expected tensor.")
                    print(path)
                    save_file(state_dict, path, metadata={"format": "safetensors"})
                    print("done")

                # print(outputs[0].shape)
                # print(outputs[0][:, :9])
                # print(outputs[1].shape)
                # print(outputs[1][:, :3])


if __name__ == "__main__":
    convert(
        model_type="lvt",
        model_size="base",
        convert=False,
        upload=False,
        load_model=True,
        load_video=True,
        inference=True,
    )


# fix the tokenizer
# fix pos embed for text
# fix the attn mask so that  
