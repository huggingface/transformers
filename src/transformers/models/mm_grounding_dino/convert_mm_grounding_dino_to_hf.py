import argparse

import torch

from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.grounding_dino.image_processing_grounding_dino import GroundingDinoImageProcessor
from transformers.models.grounding_dino.processing_grounding_dino import GroundingDinoProcessor
from transformers.models.mm_grounding_dino.configuration_mm_grounding_dino import MMGroundingDinoConfig
from transformers.models.mm_grounding_dino.modeling_mm_grounding_dino import MMGroundingDinoForObjectDetection
from transformers.models.swin.configuration_swin import SwinConfig


try:
    import mmengine
    import requests
    from PIL import Image
except ModuleNotFoundError as e:
    raise ModuleNotFoundError('Install "mmengine", "PIL" and "requests" before running the conversion script.') from e


CHECKPOINT_URL_MAP = {
    "mm_grounding_dino_tiny_o365v1_goldg": "https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg/grounding_dino_swin-t_pretrain_obj365_goldg_20231122_132602-4ea751ce.pth",
    "mm_grounding_dino_tiny_o365v1_goldg_grit": "https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_20231128_200818-169cc352.pth",
    "mm_grounding_dino_tiny_o365v1_goldg_v3det": "https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_v3det/grounding_dino_swin-t_pretrain_obj365_goldg_v3det_20231218_095741-e316e297.pth",
    "mm_grounding_dino_tiny_o365v1_goldg_grit_v3det": "https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth",
    "mm_grounding_dino_base_o365v1_goldg_v3det": "https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-b_pretrain_obj365_goldg_v3det/grounding_dino_swin-b_pretrain_obj365_goldg_v3de-f83eef00.pth",
    "mm_grounding_dino_base_all": "https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-b_pretrain_all/grounding_dino_swin-b_pretrain_all-f9818a7c.pth",
    "mm_grounding_dino_large_o365v2_oiv6_goldg": "https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-l_pretrain_obj365_goldg/grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth",
    "mm_grounding_dino_large_all": "https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-l_pretrain_all/grounding_dino_swin-l_pretrain_all-56d69e78.pth",
    "llmdet_tiny": "https://huggingface.co/fushh7/LLMDet/resolve/main/tiny.pth?download=true",
    "llmdet_base": "https://huggingface.co/fushh7/LLMDet/resolve/main/base.pth?download=true",
    "llmdet_large": "https://huggingface.co/fushh7/LLMDet/resolve/main/large.pth?download=true",
}


# copied from: https://github.com/iSEE-Laboratory/LLMDet/blob/96ec8c82a9d97b170db759e043afd5b81445d0f1/hf_model/mmdet2groundingdino_swint.py#L8C1-L13C13
def correct_unfold_reduction_order(x):
    out_channel, in_channel = x.shape
    x = x.reshape(out_channel, in_channel // 4, 4).transpose(1, 2)
    x = x[:, [0, 2, 1, 3], :]
    x = x.reshape(out_channel, in_channel)
    return x


# copied from: https://github.com/iSEE-Laboratory/LLMDet/blob/96ec8c82a9d97b170db759e043afd5b81445d0f1/hf_model/mmdet2groundingdino_swint.py#L15C1-L20C13
def correct_unfold_norm_order(x):
    in_channel = x.shape[0]
    x = x.reshape(in_channel // 4, 4).transpose(0, 1)
    x = x[[0, 2, 1, 3], :]
    x = x.reshape(in_channel)
    return x


def convert_mm_to_hf_state(mm_state: dict, hf_cfg: MMGroundingDinoConfig) -> dict:
    # key mapping will be a dict of
    # str: str - one to one key to value mapping
    # str: tup[str, tup[str]] - where the first element is str for mapping type
    #      (e.g. to chunk target key into q, k, v values or duplicate to all target keys)
    #      and the second element is a tuple of target keys
    key_mapping = {}

    ################
    # Other params #
    ################

    key_mapping["level_embed"] = "model.level_embed"

    key_mapping["query_embedding.weight"] = "model.query_position_embeddings.weight"

    for param in ["weight", "bias"]:
        key_mapping[f"memory_trans_fc.{param}"] = f"model.enc_output.{param}"
        key_mapping[f"memory_trans_norm.{param}"] = f"model.enc_output_norm.{param}"

    for layer in range(hf_cfg.decoder_layers):
        key_mapping[f"bbox_head.cls_branches.{layer}.bias"] = (
            "duplicate",
            (
                f"class_embed.{layer}.bias",
                f"model.decoder.class_embed.{layer}.bias",
            ),
        )
        for param in ["weight", "bias"]:
            key_mapping[f"bbox_head.reg_branches.{layer}.0.{param}"] = (
                "duplicate",
                (
                    f"bbox_embed.{layer}.layers.0.{param}",
                    f"model.decoder.bbox_embed.{layer}.layers.0.{param}",
                ),
            )
            key_mapping[f"bbox_head.reg_branches.{layer}.2.{param}"] = (
                "duplicate",
                (
                    f"bbox_embed.{layer}.layers.1.{param}",
                    f"model.decoder.bbox_embed.{layer}.layers.1.{param}",
                ),
            )
            key_mapping[f"bbox_head.reg_branches.{layer}.4.{param}"] = (
                "duplicate",
                (
                    f"bbox_embed.{layer}.layers.2.{param}",
                    f"model.decoder.bbox_embed.{layer}.layers.2.{param}",
                ),
            )

    # last branch in original gets mapped to encoder
    enc_idx = hf_cfg.decoder_layers
    key_mapping[f"bbox_head.cls_branches.{enc_idx}.bias"] = "model.encoder_output_class_embed.bias"
    for param in ["weight", "bias"]:
        key_mapping[f"bbox_head.reg_branches.{enc_idx}.0.{param}"] = (
            f"model.encoder_output_bbox_embed.layers.0.{param}"
        )
        key_mapping[f"bbox_head.reg_branches.{enc_idx}.2.{param}"] = (
            f"model.encoder_output_bbox_embed.layers.1.{param}"
        )
        key_mapping[f"bbox_head.reg_branches.{enc_idx}.4.{param}"] = (
            f"model.encoder_output_bbox_embed.layers.2.{param}"
        )

    ###################
    # Vision backbone #
    ###################

    for param in ["weight", "bias"]:
        key_mapping[f"backbone.patch_embed.projection.{param}"] = (
            f"model.backbone.conv_encoder.model.embeddings.patch_embeddings.projection.{param}"
        )
        key_mapping[f"backbone.patch_embed.norm.{param}"] = (
            f"model.backbone.conv_encoder.model.embeddings.norm.{param}"
        )

    for stage, depth in enumerate(hf_cfg.backbone_config.depths):
        for block in range(depth):
            key_mapping[f"backbone.stages.{stage}.blocks.{block}.attn.w_msa.relative_position_bias_table"] = (
                f"model.backbone.conv_encoder.model.encoder.layers.{stage}.blocks.{block}.attention.self.relative_position_bias_table"
            )
            key_mapping[f"backbone.stages.{stage}.blocks.{block}.attn.w_msa.relative_position_index"] = (
                f"model.backbone.conv_encoder.model.encoder.layers.{stage}.blocks.{block}.attention.self.relative_position_index"
            )

            for param in ["weight", "bias"]:
                key_mapping[f"backbone.stages.{stage}.blocks.{block}.norm1.{param}"] = (
                    f"model.backbone.conv_encoder.model.encoder.layers.{stage}.blocks.{block}.layernorm_before.{param}"
                )
                key_mapping[f"backbone.stages.{stage}.blocks.{block}.attn.w_msa.qkv.{param}"] = (
                    "split_qkv",
                    (
                        f"model.backbone.conv_encoder.model.encoder.layers.{stage}.blocks.{block}.attention.self.query.{param}",
                        f"model.backbone.conv_encoder.model.encoder.layers.{stage}.blocks.{block}.attention.self.key.{param}",
                        f"model.backbone.conv_encoder.model.encoder.layers.{stage}.blocks.{block}.attention.self.value.{param}",
                    ),
                )
                key_mapping[f"backbone.stages.{stage}.blocks.{block}.attn.w_msa.proj.{param}"] = (
                    f"model.backbone.conv_encoder.model.encoder.layers.{stage}.blocks.{block}.attention.output.dense.{param}"
                )
                key_mapping[f"backbone.stages.{stage}.blocks.{block}.norm2.{param}"] = (
                    f"model.backbone.conv_encoder.model.encoder.layers.{stage}.blocks.{block}.layernorm_after.{param}"
                )
                key_mapping[f"backbone.stages.{stage}.blocks.{block}.ffn.layers.0.0.{param}"] = (
                    f"model.backbone.conv_encoder.model.encoder.layers.{stage}.blocks.{block}.intermediate.dense.{param}"
                )
                key_mapping[f"backbone.stages.{stage}.blocks.{block}.ffn.layers.1.{param}"] = (
                    f"model.backbone.conv_encoder.model.encoder.layers.{stage}.blocks.{block}.output.dense.{param}"
                )

        is_last_stage = stage == len(hf_cfg.backbone_config.depths) - 1
        if not is_last_stage:
            key_mapping[f"backbone.stages.{stage}.downsample.reduction.weight"] = (
                f"model.backbone.conv_encoder.model.encoder.layers.{stage}.downsample.reduction.weight"
            )
            for param in ["weight", "bias"]:
                key_mapping[f"backbone.stages.{stage}.downsample.norm.{param}"] = (
                    f"model.backbone.conv_encoder.model.encoder.layers.{stage}.downsample.norm.{param}"
                )

    for out_idx in hf_cfg.backbone_config.out_indices:
        for param in ["weight", "bias"]:
            key_mapping[f"backbone.norm{out_idx - 1}.{param}"] = (
                f"model.backbone.conv_encoder.model.hidden_states_norms.stage{out_idx}.{param}"
            )

    num_normal_projs = len(hf_cfg.backbone_config.out_indices)
    for i in range(num_normal_projs):
        for param in ["weight", "bias"]:
            key_mapping[f"neck.convs.{i}.conv.{param}"] = f"model.input_proj_vision.{i}.0.{param}"
            key_mapping[f"neck.convs.{i}.gn.{param}"] = f"model.input_proj_vision.{i}.1.{param}"

    num_extra_projs = hf_cfg.num_feature_levels - num_normal_projs
    for i in range(num_extra_projs):
        for param in ["weight", "bias"]:
            key_mapping[f"neck.extra_convs.{i}.conv.{param}"] = (
                f"model.input_proj_vision.{num_normal_projs + i}.0.{param}"
            )
            key_mapping[f"neck.extra_convs.{i}.gn.{param}"] = (
                f"model.input_proj_vision.{num_normal_projs + i}.1.{param}"
            )

    ##################
    # Language model #
    ##################

    for k in mm_state:
        if k.startswith("language_model") and "position_ids" not in k:
            key_mapping[k] = k.replace("language_model.language_backbone.body.model", "model.text_backbone")

    for param in ["weight", "bias"]:
        key_mapping[f"text_feat_map.{param}"] = f"model.text_projection.{param}"

    ###########
    # Encoder #
    ###########

    for layer in range(hf_cfg.encoder_layers):
        # fusion
        key_mapping[f"encoder.fusion_layers.{layer}.gamma_v"] = (
            f"model.encoder.layers.{layer}.fusion_layer.vision_param"
        )
        key_mapping[f"encoder.fusion_layers.{layer}.gamma_l"] = f"model.encoder.layers.{layer}.fusion_layer.text_param"
        for param in ["weight", "bias"]:
            key_mapping[f"encoder.fusion_layers.{layer}.layer_norm_v.{param}"] = (
                f"model.encoder.layers.{layer}.fusion_layer.layer_norm_vision.{param}"
            )
            key_mapping[f"encoder.fusion_layers.{layer}.attn.v_proj.{param}"] = (
                f"model.encoder.layers.{layer}.fusion_layer.attn.vision_proj.{param}"
            )
            key_mapping[f"encoder.fusion_layers.{layer}.attn.values_v_proj.{param}"] = (
                f"model.encoder.layers.{layer}.fusion_layer.attn.values_vision_proj.{param}"
            )
            key_mapping[f"encoder.fusion_layers.{layer}.attn.out_v_proj.{param}"] = (
                f"model.encoder.layers.{layer}.fusion_layer.attn.out_vision_proj.{param}"
            )

            key_mapping[f"encoder.fusion_layers.{layer}.layer_norm_l.{param}"] = (
                f"model.encoder.layers.{layer}.fusion_layer.layer_norm_text.{param}"
            )
            key_mapping[f"encoder.fusion_layers.{layer}.attn.l_proj.{param}"] = (
                f"model.encoder.layers.{layer}.fusion_layer.attn.text_proj.{param}"
            )
            key_mapping[f"encoder.fusion_layers.{layer}.attn.values_l_proj.{param}"] = (
                f"model.encoder.layers.{layer}.fusion_layer.attn.values_text_proj.{param}"
            )
            key_mapping[f"encoder.fusion_layers.{layer}.attn.out_l_proj.{param}"] = (
                f"model.encoder.layers.{layer}.fusion_layer.attn.out_text_proj.{param}"
            )

        # img deformable self attention
        for proj in ["sampling_offsets", "attention_weights", "value_proj", "output_proj"]:
            for param in ["weight", "bias"]:
                key_mapping[f"encoder.layers.{layer}.self_attn.{proj}.{param}"] = (
                    f"model.encoder.layers.{layer}.deformable_layer.self_attn.{proj}.{param}"
                )
        for param in ["weight", "bias"]:
            key_mapping[f"encoder.layers.{layer}.norms.0.{param}"] = (
                f"model.encoder.layers.{layer}.deformable_layer.self_attn_layer_norm.{param}"
            )
            key_mapping[f"encoder.layers.{layer}.ffn.layers.0.0.{param}"] = (
                f"model.encoder.layers.{layer}.deformable_layer.fc1.{param}"
            )
            key_mapping[f"encoder.layers.{layer}.ffn.layers.1.{param}"] = (
                f"model.encoder.layers.{layer}.deformable_layer.fc2.{param}"
            )
            key_mapping[f"encoder.layers.{layer}.norms.1.{param}"] = (
                f"model.encoder.layers.{layer}.deformable_layer.final_layer_norm.{param}"
            )

        # text self attention
        for param in ["weight", "bias"]:
            key_mapping[f"encoder.text_layers.{layer}.self_attn.attn.in_proj_{param}"] = (
                "split_qkv",
                (
                    f"model.encoder.layers.{layer}.text_enhancer_layer.self_attn.query.{param}",
                    f"model.encoder.layers.{layer}.text_enhancer_layer.self_attn.key.{param}",
                    f"model.encoder.layers.{layer}.text_enhancer_layer.self_attn.value.{param}",
                ),
            )
            key_mapping[f"encoder.text_layers.{layer}.self_attn.attn.out_proj.{param}"] = (
                f"model.encoder.layers.{layer}.text_enhancer_layer.self_attn.out_proj.{param}"
            )
        for param in ["weight", "bias"]:
            key_mapping[f"encoder.text_layers.{layer}.norms.0.{param}"] = (
                f"model.encoder.layers.{layer}.text_enhancer_layer.layer_norm_before.{param}"
            )
            key_mapping[f"encoder.text_layers.{layer}.ffn.layers.0.0.{param}"] = (
                f"model.encoder.layers.{layer}.text_enhancer_layer.fc1.{param}"
            )
            key_mapping[f"encoder.text_layers.{layer}.ffn.layers.1.{param}"] = (
                f"model.encoder.layers.{layer}.text_enhancer_layer.fc2.{param}"
            )
            key_mapping[f"encoder.text_layers.{layer}.norms.1.{param}"] = (
                f"model.encoder.layers.{layer}.text_enhancer_layer.layer_norm_after.{param}"
            )

    ###########
    # Decoder #
    ###########

    for param in ["weight", "bias"]:
        key_mapping[f"decoder.norm.{param}"] = f"model.decoder.layer_norm.{param}"

    for i in [0, 1]:
        for param in ["weight", "bias"]:
            key_mapping[f"decoder.ref_point_head.layers.{i}.{param}"] = (
                f"model.decoder.reference_points_head.layers.{i}.{param}"
            )

    for layer in range(hf_cfg.encoder_layers):
        # query self att
        for param in ["weight", "bias"]:
            key_mapping[f"decoder.layers.{layer}.self_attn.attn.in_proj_{param}"] = (
                "split_qkv",
                (
                    f"model.decoder.layers.{layer}.self_attn.query.{param}",
                    f"model.decoder.layers.{layer}.self_attn.key.{param}",
                    f"model.decoder.layers.{layer}.self_attn.value.{param}",
                ),
            )
            key_mapping[f"decoder.layers.{layer}.self_attn.attn.out_proj.{param}"] = (
                f"model.decoder.layers.{layer}.self_attn.out_proj.{param}"
            )
            key_mapping[f"decoder.layers.{layer}.norms.0.{param}"] = (
                f"model.decoder.layers.{layer}.self_attn_layer_norm.{param}"
            )

        # query-text cross att
        for param in ["weight", "bias"]:
            key_mapping[f"decoder.layers.{layer}.cross_attn_text.attn.in_proj_{param}"] = (
                "split_qkv",
                (
                    f"model.decoder.layers.{layer}.encoder_attn_text.query.{param}",
                    f"model.decoder.layers.{layer}.encoder_attn_text.key.{param}",
                    f"model.decoder.layers.{layer}.encoder_attn_text.value.{param}",
                ),
            )
            key_mapping[f"decoder.layers.{layer}.cross_attn_text.attn.out_proj.{param}"] = (
                f"model.decoder.layers.{layer}.encoder_attn_text.out_proj.{param}"
            )
            key_mapping[f"decoder.layers.{layer}.norms.1.{param}"] = (
                f"model.decoder.layers.{layer}.encoder_attn_text_layer_norm.{param}"
            )

        # query-img deformable cross att
        for proj in ["sampling_offsets", "attention_weights", "value_proj", "output_proj"]:
            for param in ["weight", "bias"]:
                key_mapping[f"decoder.layers.{layer}.cross_attn.{proj}.{param}"] = (
                    f"model.decoder.layers.{layer}.encoder_attn.{proj}.{param}"
                )
        for param in ["weight", "bias"]:
            key_mapping[f"decoder.layers.{layer}.norms.2.{param}"] = (
                f"model.decoder.layers.{layer}.encoder_attn_layer_norm.{param}"
            )

        # ffn
        for param in ["weight", "bias"]:
            key_mapping[f"decoder.layers.{layer}.ffn.layers.0.0.{param}"] = f"model.decoder.layers.{layer}.fc1.{param}"
            key_mapping[f"decoder.layers.{layer}.ffn.layers.1.{param}"] = f"model.decoder.layers.{layer}.fc2.{param}"
            key_mapping[f"decoder.layers.{layer}.norms.3.{param}"] = (
                f"model.decoder.layers.{layer}.final_layer_norm.{param}"
            )

    #################
    # convert model #
    #################

    # do a copy here so it errors out with any missed or extra keys
    hf_state = {}

    # map keys
    for mm_key, hf_key in key_mapping.items():
        # str, str -> one to one
        if isinstance(hf_key, str):
            hf_state[hf_key] = mm_state[mm_key]
        # str, tup[str, tup[str]] -> either chunk into qkv or duplicate
        elif isinstance(hf_key, tuple):
            mapping_type, keys = hf_key
            if mapping_type == "split_qkv":
                q_param, k_param, v_param = mm_state[mm_key].chunk(3)
                q_key, k_key, v_key = keys
                hf_state[q_key] = q_param
                hf_state[k_key] = k_param
                hf_state[v_key] = v_param
            elif mapping_type == "duplicate":
                for key in keys:
                    hf_state[key] = mm_state[mm_key]
            else:
                raise ValueError(f"Unknown mapping type: {mapping_type}")

    # convert downsample params
    for k in hf_state:
        if "downsample.reduction" in k:
            hf_state[k] = correct_unfold_reduction_order(hf_state[k])
        if "downsample.norm" in k:
            hf_state[k] = correct_unfold_norm_order(hf_state[k])

    return hf_state


def convert_mm_to_hf_config(mm_cfg: mmengine.Config) -> MMGroundingDinoConfig:
    hf_backbone_cfg = SwinConfig(
        embed_dim=mm_cfg["model"]["backbone"]["embed_dims"],
        depths=mm_cfg["model"]["backbone"]["depths"],
        num_heads=mm_cfg["model"]["backbone"]["num_heads"],
        window_size=mm_cfg["model"]["backbone"]["window_size"],
        mlp_ratio=mm_cfg["model"]["backbone"]["mlp_ratio"],
        qkv_bias=mm_cfg["model"]["backbone"]["qkv_bias"],
        out_indices=[idx + 1 for idx in mm_cfg["model"]["backbone"]["out_indices"]],
    )
    hf_cfg = MMGroundingDinoConfig(
        backbone_config=hf_backbone_cfg,
        num_queries=mm_cfg["model"]["num_queries"],
        num_feature_levels=mm_cfg["model"]["neck"]["num_outs"],
    )
    return hf_cfg


def convert_mm_to_hf(mm_cfg: mmengine.Config, mm_state: dict) -> tuple[MMGroundingDinoConfig, dict]:
    hf_cfg = convert_mm_to_hf_config(mm_cfg)
    hf_state = convert_mm_to_hf_state(mm_state, hf_cfg)
    return hf_cfg, hf_state


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str,
        choices=list(CHECKPOINT_URL_MAP.keys()),
        help="URL to the original mm grounding dino checkpoint.",
    )
    parser.add_argument("--hub-user-name", type=str, help="User name on the huggingface hub.")
    parser.add_argument("--push-to-hub", action="store_true", help="Whether to push model to hub or not.")
    parser.add_argument("--draw-result", action="store_true", help="Whether to draw and display predictions or not.")
    return parser.parse_args()


def main(args):
    # download weights
    ckpt = torch.hub.load_state_dict_from_url(CHECKPOINT_URL_MAP[args.checkpoint], map_location="cpu")
    mm_cfg = mmengine.Config.fromstring(ckpt["meta"]["cfg"], file_format=".py")
    mm_state = ckpt["state_dict"]

    # convert model
    hf_cfg, hf_state = convert_mm_to_hf(mm_cfg, mm_state)
    hf_model = MMGroundingDinoForObjectDetection(hf_cfg).eval()
    print(hf_model.load_state_dict(hf_state, strict=True))

    # check predictions
    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(image_url, stream=True).raw)
    text = [["cat", "remote"]]

    # set up processor
    img_processor = GroundingDinoImageProcessor()
    txt_processor = BertTokenizer.from_pretrained("bert-base-uncased")
    processor = GroundingDinoProcessor(img_processor, txt_processor)

    # run inference
    inputs = processor(images=image, text=text, return_tensors="pt")
    with torch.no_grad():
        outputs = hf_model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, box_threshold=0.4, text_threshold=0.3, target_sizes=[image.size[::-1]]
    )
    result = results[0]
    for box, score, labels in zip(result["boxes"], result["scores"], result["text_labels"]):
        box = [round(x, 2) for x in box.tolist()]
        print(f"Detected {labels} with confidence {round(score.item(), 3)} at location {box}")

    # draw results if needed
    if args.draw_result:
        try:
            from torchvision.transforms.v2.functional import pil_to_tensor, to_pil_image
            from torchvision.utils import draw_bounding_boxes
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError('Please install "torchvision" if you want to draw the result.') from e

        img_draw = to_pil_image(
            draw_bounding_boxes(
                pil_to_tensor(image),
                result["boxes"],
                [f"{label}: {score:.2f}" for label, score in zip(result["text_labels"], result["scores"])],
            )
        )
        img_draw.show()

    # push to hub if needed
    if args.push_to_hub:
        if getattr(args, "hub_user_name", None) is None:
            raise ValueError('You should provide "--hub-user-name" when pushing to hub.')
        hf_model.push_to_hub(f"{args.hub_user_name}/{args.checkpoint}")
        processor.push_to_hub(f"{args.hub_user_name}/{args.checkpoint}")


if __name__ == "__main__":
    main(parse_args())
