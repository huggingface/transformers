#!/usr/bin/env python3
import argparse
import json

import torch
from PIL import Image

from huggingface_hub import hf_hub_download
from timm.models import create_model
from transformers import (
    BeitFeatureExtractor,
    Data2VecVisionConfig,
    Data2VecVisionForImageClassification,
    Data2VecVisionModel,
)


def create_rename_keys(config, has_lm_head=False, is_semantic=False, hf_prefix="data2vec."):
    prefix = "backbone." if is_semantic else ""

    rename_keys = []
    for i in range(config.num_hidden_layers):
        # encoder layers: output projection, 2 feedforward neural networks and 2 layernorms
        rename_keys.append(
            (f"{prefix}blocks.{i}.norm1.weight", f"{hf_prefix}encoder.layer.{i}.layernorm_before.weight")
        )
        rename_keys.append((f"{prefix}blocks.{i}.norm1.bias", f"{hf_prefix}encoder.layer.{i}.layernorm_before.bias"))
        rename_keys.append(
            (f"{prefix}blocks.{i}.attn.proj.weight", f"{hf_prefix}encoder.layer.{i}.attention.output.dense.weight")
        )
        rename_keys.append(
            (f"{prefix}blocks.{i}.attn.proj.bias", f"{hf_prefix}encoder.layer.{i}.attention.output.dense.bias")
        )
        rename_keys.append(
            (f"{prefix}blocks.{i}.norm2.weight", f"{hf_prefix}encoder.layer.{i}.layernorm_after.weight")
        )
        rename_keys.append((f"{prefix}blocks.{i}.norm2.bias", f"{hf_prefix}encoder.layer.{i}.layernorm_after.bias"))
        rename_keys.append(
            (f"{prefix}blocks.{i}.mlp.fc1.weight", f"{hf_prefix}encoder.layer.{i}.intermediate.dense.weight")
        )
        rename_keys.append(
            (f"{prefix}blocks.{i}.mlp.fc1.bias", f"{hf_prefix}encoder.layer.{i}.intermediate.dense.bias")
        )
        rename_keys.append((f"{prefix}blocks.{i}.mlp.fc2.weight", f"{hf_prefix}encoder.layer.{i}.output.dense.weight"))
        rename_keys.append((f"{prefix}blocks.{i}.mlp.fc2.bias", f"{hf_prefix}encoder.layer.{i}.output.dense.bias"))

    # projection layer + position embeddings
    rename_keys.extend(
        [
            (f"{prefix}cls_token", f"{hf_prefix}embeddings.cls_token"),
            (f"{prefix}patch_embed.proj.weight", f"{hf_prefix}embeddings.patch_embeddings.projection.weight"),
            (f"{prefix}patch_embed.proj.bias", f"{hf_prefix}embeddings.patch_embeddings.projection.bias"),
        ]
    )

    if has_lm_head:
        # mask token + shared relative position bias + layernorm
        rename_keys.extend(
            [
                ("mask_token", f"{hf_prefix}embeddings.mask_token"),
                (
                    "rel_pos_bias.relative_position_bias_table",
                    f"{hf_prefix}encoder.relative_position_bias.relative_position_bias_table",
                ),
                (
                    "rel_pos_bias.relative_position_index",
                    f"{hf_prefix}encoder.relative_position_bias.relative_position_index",
                ),
                ("norm.weight", "layernorm.weight"),
                ("norm.bias", "layernorm.bias"),
            ]
        )
    elif is_semantic:
        # semantic segmentation classification heads
        rename_keys.extend(
            [
                ("decode_head.conv_seg.weight", "decode_head.classifier.weight"),
                ("decode_head.conv_seg.bias", "decode_head.classifier.bias"),
                ("auxiliary_head.conv_seg.weight", "auxiliary_head.classifier.weight"),
                ("auxiliary_head.conv_seg.bias", "auxiliary_head.classifier.bias"),
            ]
        )
    else:
        # layernorm + classification head
        rename_keys.extend(
            [
                ("fc_norm.weight", f"{hf_prefix}pooler.layernorm.weight"),
                ("fc_norm.bias", f"{hf_prefix}pooler.layernorm.bias"),
                ("head.weight", "classifier.weight"),
                ("head.bias", "classifier.bias"),
            ]
        )

    return rename_keys


def read_in_q_k_v(state_dict, config, has_lm_head=False, is_semantic=False, hf_prefix="data2vec_vision."):
    for i in range(config.num_hidden_layers):
        prefix = "backbone." if is_semantic else ""
        # queries, keys and values
        in_proj_weight = state_dict.pop(f"{prefix}blocks.{i}.attn.qkv.weight")
        q_bias = state_dict.pop(f"{prefix}blocks.{i}.attn.q_bias")
        v_bias = state_dict.pop(f"{prefix}blocks.{i}.attn.v_bias")

        state_dict[f"{hf_prefix}encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[
            : config.hidden_size, :
        ]
        state_dict[f"{hf_prefix}encoder.layer.{i}.attention.attention.query.bias"] = q_bias
        state_dict[f"{hf_prefix}encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[
            config.hidden_size : config.hidden_size * 2, :
        ]
        state_dict[f"{hf_prefix}encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[
            -config.hidden_size :, :
        ]
        state_dict[f"{hf_prefix}encoder.layer.{i}.attention.attention.value.bias"] = v_bias

        # gamma_1 and gamma_2
        # we call them lambda because otherwise they are renamed when using .from_pretrained
        gamma_1 = state_dict.pop(f"{prefix}blocks.{i}.gamma_1")
        gamma_2 = state_dict.pop(f"{prefix}blocks.{i}.gamma_2")

        state_dict[f"{hf_prefix}encoder.layer.{i}.lambda_1"] = gamma_1
        state_dict[f"{hf_prefix}encoder.layer.{i}.lambda_2"] = gamma_2

        # relative_position bias table + index
        if not has_lm_head:
            # each layer has its own relative position bias
            table = state_dict.pop(f"{prefix}blocks.{i}.attn.relative_position_bias_table")
            index = state_dict.pop(f"{prefix}blocks.{i}.attn.relative_position_index")

            state_dict[
                f"{hf_prefix}encoder.layer.{i}.attention.attention.relative_position_bias.relative_position_bias_table"
            ] = table
            state_dict[
                f"{hf_prefix}encoder.layer.{i}.attention.attention.relative_position_bias.relative_position_index"
            ] = index


def get_args():
    parser = argparse.ArgumentParser(
        "Convert Data2VecVision to HF for image classification and pretraining", add_help=False
    )
    parser.add_argument("--hf_checkpoint_name", type=str)
    parser.add_argument("--input_size", default=224, type=int, help="images input size")
    parser.add_argument("--beit_checkpoint", default="", help="beit checkpoint")

    return parser.parse_args()


def load_beit_model(args, is_finetuned, is_large):
    def load_state_dict(model, state_dict, prefix="", ignore_missing="relative_position_index"):
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=""):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs
            )
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        load(model, prefix=prefix)

        warn_missing_keys = []
        ignore_missing_keys = []
        for key in missing_keys:
            keep_flag = True
            for ignore_key in ignore_missing.split("|"):
                if ignore_key in key:
                    keep_flag = False
                    break
            if keep_flag:
                warn_missing_keys.append(key)
            else:
                ignore_missing_keys.append(key)

        missing_keys = warn_missing_keys

        if len(missing_keys) > 0:
            print(
                "Weights of {} not initialized from pretrained model: {}".format(
                    model.__class__.__name__, missing_keys
                )
            )
        if len(unexpected_keys) > 0:
            print("Weights from pretrained model not used in {}: {}".format(model.__class__.__name__, unexpected_keys))
        if len(ignore_missing_keys) > 0:
            print(
                "Ignored weights of {} not initialized from pretrained model: {}".format(
                    model.__class__.__name__, ignore_missing_keys
                )
            )
        if len(error_msgs) > 0:
            print("\n".join(error_msgs))

    model_kwargs = {
        "pretrained": False,
        "use_shared_rel_pos_bias": True,
        "use_abs_pos_emb": False,
        "init_values": 0.1,
    }

    if is_finetuned:
        model_kwargs.update(
            {
                "num_classes": 1000,
                "use_mean_pooling": True,
                "init_scale": 0.001,
                "use_rel_pos_bias": True,
            }
        )

    model = create_model(
        "beit_large_patch16_224" if is_large else "beit_base_patch16_224",
        **model_kwargs,
    )
    patch_size = model.patch_embed.patch_size
    args.window_size = (args.input_size // patch_size[0], args.input_size // patch_size[1])
    checkpoint = torch.load(args.beit_checkpoint, map_location="cpu")

    print(f"Load ckpt from {args.beit_checkpoint}")
    checkpoint_model = None
    for model_key in ("model", "module"):
        if model_key in checkpoint:
            checkpoint_model = checkpoint[model_key]
            print(f"Load state_dict by model_key = {model_key}")
            break

    all_keys = list(checkpoint_model.keys())
    for key in all_keys:
        if "relative_position_index" in key:
            checkpoint_model.pop(key)

        if "relative_position_bias_table" in key:
            rel_pos_bias = checkpoint_model[key]
            src_num_pos, num_attn_heads = rel_pos_bias.size()
            dst_num_pos, _ = model.state_dict()[key].size()
            dst_patch_shape = model.patch_embed.patch_shape
            if dst_patch_shape[0] != dst_patch_shape[1]:
                raise NotImplementedError()

    load_state_dict(model, checkpoint_model, prefix="")

    return model


def main():
    args = get_args()

    is_finetuned = "ft1k" in args.hf_checkpoint_name
    is_large = "large" in args.hf_checkpoint_name

    if is_finetuned:
        # To convert Beit's data2vec_vision to HF you need to copy
        # https://github.com/facebookresearch/data2vec_vision/blob/main/beit/modeling_finetune.py
        # into this folder.
        import modeling_finetune  # noqa: F401
    else:
        # To convert Beit's data2vec_vision to HF you need to copy
        # https://github.com/facebookresearch/data2vec_vision/blob/main/beit/modeling_cyclical.py
        # into this folder
        # IMPORTANT: Note that for now we've only converted the down-stream
        # model and not the full pretrained model. This means for the integration
        # test you need to add a `return x` after the following line:
        # https://github.com/facebookresearch/data2vec_vision/blob/af9a36349aaed59ae66e69b5dabeef2d62fdc5da/beit/modeling_cyclical.py#L197
        # to make the integration test pass.
        import modeling_cyclical  # noqa: F401

    # 1. Create model config
    config = Data2VecVisionConfig()
    if is_finetuned:
        config.use_relative_position_bias = True
        config.use_shared_relative_position_bias = False
        config.use_mean_pooling = True
        config.num_labels = 1000

        repo_id = "datasets/huggingface/label-files"
        filename = "imagenet-1k-id2label.json"
        id2label = json.load(open(hf_hub_download(repo_id, filename), "r"))
        id2label = {int(k): v for k, v in id2label.items()}
        config.id2label = id2label
        config.label2id = {v: k for k, v in id2label.items()}
    else:
        config.use_relative_position_bias = False
        config.use_shared_relative_position_bias = True
        config.use_mean_pooling = False

    if is_large:
        config.hidden_size = 1024
        config.intermediate_size = 4096
        config.num_hidden_layers = 24
        config.num_attention_heads = 16

    # 2. Load Beit model
    orig_model = load_beit_model(args, is_finetuned, is_large)
    orig_model.eval()

    # 3. Forward Beit model
    feature_extractor = BeitFeatureExtractor(size=config.image_size, do_center_crop=False)
    image = Image.open("../../../../tests/fixtures/tests_samples/COCO/000000039769.png")
    encoding = feature_extractor(images=image, return_tensors="pt")
    pixel_values = encoding["pixel_values"]

    orig_args = (pixel_values,) if is_finetuned else (pixel_values, None)
    with torch.no_grad():
        orig_model_output = orig_model(*orig_args)

    # 4. Load HF Data2VecVision model
    if is_finetuned:
        hf_model = Data2VecVisionForImageClassification(config)
        hf_model.eval()
        has_lm_head = False
        hf_prefix = "data2vec_vision."
    else:
        hf_model = Data2VecVisionModel(config)
        hf_model.eval()
        has_lm_head = True
        hf_prefix = ""

    rename_keys = create_rename_keys(config, hf_prefix=hf_prefix, has_lm_head=has_lm_head)
    state_dict = orig_model.state_dict()
    for src, dest in rename_keys:
        val = state_dict.pop(src)
        state_dict[dest] = val

    read_in_q_k_v(state_dict, config, hf_prefix=hf_prefix, has_lm_head=has_lm_head)
    missing_keys, unexpected_keys = hf_model.load_state_dict(state_dict, strict=False)
    print("HF missing", missing_keys)
    print("HF unexpected_keys", unexpected_keys)

    # 5. Forward HF Data2VecVision model
    with torch.no_grad():
        hf_model_output = hf_model(pixel_values)

    hf_output = hf_model_output.logits if is_finetuned else hf_model_output.last_hidden_state

    # 6. Compare
    max_absolute_diff = torch.max(torch.abs(hf_output - orig_model_output)).item()

    print(f"max_absolute_diff = {max_absolute_diff}")
    success = torch.allclose(hf_output, orig_model_output, atol=1e-3)
    print("Do both models output the same tensors?", "🔥" if success else "💩")
    if not success:
        raise Exception("Something went wRoNg")

    # 7. Save
    print(f"Saving to {args.hf_checkpoint_name}")
    hf_model.save_pretrained(args.hf_checkpoint_name)
    feature_extractor.save_pretrained(args.hf_checkpoint_name)


if __name__ == "__main__":
    main()
    # Run the following to convert checkpoints
    #  python ./convert_data2vec_vision_original_pytorch_checkpoint_to_pytorch.py \
    #          --beit_checkpoint ./pretrained_base.pt \
    #          --hf_checkpoint_name "./data2vec-vision-base"
    #  python ./convert_data2vec_vision_original_pytorch_checkpoint_to_pytorch.py \
    #          --beit_checkpoint ./finetuned_base.pt \
    #          --hf_checkpoint_name "./data2vec-vision-base-ft1k"
    #  python ./convert_data2vec_vision_original_pytorch_checkpoint_to_pytorch.py \
    #          --beit_checkpoint ./pretrained_large.pt \
    #          --hf_checkpoint_name "./data2vec-vision-large"
    #  python ./convert_data2vec_vision_original_pytorch_checkpoint_to_pytorch.py \
    #          --beit_checkpoint ./finetuned_large.pt \
    #          --hf_checkpoint_name "./data2vec-vision-large-ft1k"
