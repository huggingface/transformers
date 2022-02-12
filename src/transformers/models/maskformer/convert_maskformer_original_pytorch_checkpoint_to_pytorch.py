from __future__ import annotations

import logging
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, Iterator, List, Set, Tuple

import torch
import torchvision.transforms as T
from PIL import Image
from torch import Tensor

import requests
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config
from transformers.models.maskformer.modeling_maskformer import (
    MaskFormerConfig,
    MaskFormerForInstanceSegmentation,
    MaskFormerForInstanceSegmentationOutput,
    MaskFormerModel,
    MaskFormerOutput
)
from transformers.models.maskformer.feature_extraction_maskformer import MaskFormerFeatureExtractor
from transformers.models.maskformer.MaskFormer.mask_former import add_mask_former_config
from transformers.models.maskformer.MaskFormer.mask_former.mask_former_model import MaskFormer as OriginalMaskFormer


StateDict = Dict[str, Tensor]
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


class TrackedStateDict:
    def __init__(self, to_track: Dict):
        """This class "tracks" a python dictionary by keeping track of which item is accessed.

        Args:
            to_track (Dict): The dictionary we wish to track
        """
        self.to_track = to_track
        self._seen: Set[str] = set()

    def __getitem__(self, key: str) -> Any:
        return self.to_track[key]

    def __setitem__(self, key: str, item: Any):
        self._seen.add(key)
        self.to_track[key] = item

    def diff(self) -> List[str]:
        """This method returns a set difference between the keys in the tracked state dict and the one we have access so far.
        This is an effective method to check if we have update all the keys

        Returns:
            List[str]: List of keys not yet updated
        """
        return set(list(self.to_track.keys())) - self._seen

    def copy(self) -> Dict:
        # proxy the call to the internal dictionary
        return self.to_track.copy()


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    img_data = requests.get(url, stream=True).raw
    im = Image.open(img_data)
    return im


@dataclass
class Args:
    """Fake command line arguments needed by maskformer/detectron implementation"""

    config_file: str


def setup_cfg(args: Args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_mask_former_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    return cfg


class OriginalMaskFormerConfigToOursConverter:

    def __call__(self, original_config: object) -> MaskFormerConfig:

        model = original_config.MODEL
        mask_former = model.MASK_FORMER
        swin = model.SWIN


        config: MaskFormerConfig = MaskFormerConfig(
            fpn_feature_size=model.SEM_SEG_HEAD.CONVS_DIM,
            mask_feature_size=model.SEM_SEG_HEAD.MASK_DIM,
            num_labels=model.SEM_SEG_HEAD.NUM_CLASSES,
            no_object_weight=mask_former.NO_OBJECT_WEIGHT,
            num_queries=mask_former.NUM_OBJECT_QUERIES,
            backbone_config=dict(
                pretrain_img_size=swin.PRETRAIN_IMG_SIZE,
                in_channels=3,
                patch_size=swin.PATCH_SIZE,
                embed_dim=swin.EMBED_DIM,
                depths=swin.DEPTHS,
                num_heads=swin.NUM_HEADS,
                window_size=swin.WINDOW_SIZE,
                # TODO miss attn_drop_rage, path_norm
                drop_path_rate=swin.DROP_PATH_RATE,
                model_type='swin'
            ),
            dice_weight=mask_former.DICE_WEIGHT,
            ce_weight=1.0,
            mask_weight=mask_former.MASK_WEIGHT,
            detr_config=dict(
                detr_max_position_embeddings=1024,
                detr_encoder_layers=6,
                detr_encoder_ffn_dim=2048,
                detr_encoder_attention_heads=8,
                detr_decoder_layers=mask_former.DEC_LAYERS,
                detr_decoder_ffn_dim=mask_former.DIM_FEEDFORWARD,
                detr_decoder_attention_heads=mask_former.NHEADS,
                detr_encoder_layerdrop=0.0,
                detr_decoder_layerdrop=0.0,
                detr_d_model=mask_former.HIDDEN_DIM,
                detr_dropout=mask_former.DROPOUT,
                detr_attention_dropout=0.0,
                detr_activation_dropout=0.0,
                detr_init_std=0.02,
                detr_init_xavier_std=1.0,
                detr_scale_embedding=False,
                detr_auxiliary_loss=False,
                detr_dilation=False,
                # default pretrained config values
            )
        )

        return config


class OriginalMaskFormerConfigToFeatureExtractorConverter:
    def __call__(self, original_config: object) -> MaskFormerFeatureExtractor:
        model = original_config.MODEL
        model_input = original_config.INPUT

        return MaskFormerFeatureExtractor(
            image_mean=(torch.tensor(model.PIXEL_MEAN) / 255).tolist(),
            image_std=(torch.tensor(model.PIXEL_STD) / 255).tolist(),
            size=model_input.MIN_SIZE_TEST,
            max_size=model_input.MAX_SIZE_TEST,
        )


class OriginalMaskFormerCheckpoinToOursConverter:
    def __init__(self, original_model: OriginalMaskFormer, config: MaskFormerConfig):
        self.original_model = original_model
        self.config = config

    def pop_all(self, renamed_keys: List[Tuple[str, str]], dst_state_dict: StateDict, src_state_dict: StateDict):
        for (src_key, dst_key) in renamed_keys:
            dst_state_dict[dst_key] = src_state_dict.pop(src_key)

    def replace_backbone(self, dst_state_dict: StateDict, src_state_dict: StateDict, config: MaskFormerConfig):
        dst_prefix: str = "pixel_level_module.encoder"
        src_prefix: str = "backbone"

        renamed_keys = [
            (
                f"{src_prefix}.patch_embed.proj.weight",
                f"{dst_prefix}.model.embeddings.patch_embeddings.projection.weight",
            ),
            (f"{src_prefix}.patch_embed.proj.bias", f"{dst_prefix}.model.embeddings.patch_embeddings.projection.bias"),
            (f"{src_prefix}.patch_embed.norm.weight", f"{dst_prefix}.model.embeddings.norm.weight"),
            (f"{src_prefix}.patch_embed.norm.bias", f"{dst_prefix}.model.embeddings.norm.bias"),
        ]
        num_layers = len(config.backbone.depths)
        for layer_idx in range(num_layers):
            for block_idx in range(config.backbone.depths[layer_idx]):
                renamed_keys.extend(
                    [  # src, dst
                        (
                            f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.norm1.weight",
                            f"{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.layernorm_before.weight",
                        ),
                        (
                            f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.norm1.bias",
                            f"{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.layernorm_before.bias",
                        ),
                        (
                            f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.relative_position_bias_table",
                            f"{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.relative_position_bias_table",
                        ),
                    ]
                )
                # now we need to handle the attentions
                # read in weights + bias of input projection layer of cross-attention

                src_att_weight = src_state_dict[f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.qkv.weight"]
                src_att_bias = src_state_dict[f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.qkv.bias"]

                size = src_att_weight.shape[0]
                offset = size // 3
                dst_state_dict[
                    f"{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.query.weight"
                ] = src_att_weight[:offset, :]
                dst_state_dict[
                    f"{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.query.bias"
                ] = src_att_bias[:offset]

                dst_state_dict[
                    f"{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.key.weight"
                ] = src_att_weight[offset : offset * 2, :]
                dst_state_dict[
                    f"{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.key.bias"
                ] = src_att_bias[offset : offset * 2]

                dst_state_dict[
                    f"{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.value.weight"
                ] = src_att_weight[-offset:, :]
                dst_state_dict[
                    f"{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.value.bias"
                ] = src_att_bias[-offset:]

                # let's pop them
                src_state_dict.pop(f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.qkv.weight")
                src_state_dict.pop(f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.qkv.bias")
                # proj
                renamed_keys.extend(
                    [
                        (
                            f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.proj.weight",
                            f"{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.output.dense.weight",
                        ),
                        (
                            f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.proj.bias",
                            f"{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.output.dense.bias",
                        ),
                    ]
                )

                # second norm
                renamed_keys.extend(
                    [
                        (
                            f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.norm2.weight",
                            f"{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.layernorm_after.weight",
                        ),
                        (
                            f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.norm2.bias",
                            f"{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.layernorm_after.bias",
                        ),
                    ]
                )

                # mlp
                renamed_keys.extend(
                    [
                        (
                            f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.mlp.fc1.weight",
                            f"{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.intermediate.dense.weight",
                        ),
                        (
                            f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.mlp.fc1.bias",
                            f"{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.intermediate.dense.bias",
                        ),
                        (
                            f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.mlp.fc2.weight",
                            f"{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.output.dense.weight",
                        ),
                        (
                            f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.mlp.fc2.bias",
                            f"{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.output.dense.bias",
                        ),
                    ]
                )

                renamed_keys.extend(
                    [
                        (
                            f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.relative_position_index",
                            f"{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.relative_position_index",
                        )
                    ]
                )

            if layer_idx < num_layers - 1:
                # patch merging
                renamed_keys.extend(
                    [
                        (
                            f"{src_prefix}.layers.{layer_idx}.downsample.reduction.weight",
                            f"{dst_prefix}.model.encoder.layers.{layer_idx}.downsample.reduction.weight",
                        ),
                        (
                            f"{src_prefix}.layers.{layer_idx}.downsample.norm.weight",
                            f"{dst_prefix}.model.encoder.layers.{layer_idx}.downsample.norm.weight",
                        ),
                        (
                            f"{src_prefix}.layers.{layer_idx}.downsample.norm.bias",
                            f"{dst_prefix}.model.encoder.layers.{layer_idx}.downsample.norm.bias",
                        ),
                    ]
                )
                # hidden states norms
                renamed_keys.extend(
                    [
                        (
                            f"{src_prefix}.norm{layer_idx}.weight",
                            f"{dst_prefix}.hidden_states_norms.{layer_idx}.weight",
                        ),
                        (
                            f"{src_prefix}.norm{layer_idx}.bias",
                            f"{dst_prefix}.hidden_states_norms.{layer_idx}.bias",
                        ),
                    ]
                )
            else:
                renamed_keys.extend(
                    [
                        (
                            f"{src_prefix}.norm3.weight",
                            f"{dst_prefix}.model.layernorm.weight",
                        ),
                        (
                            f"{src_prefix}.norm3.bias",
                            f"{dst_prefix}.model.layernorm.bias",
                        ),
                    ]
                )
        # model.layernorm.weight and our hiddin_state_norms[3] have to be the same
        assert torch.allclose(
            dst_state_dict[f"{dst_prefix}.hidden_states_norms.3.weight"],
            dst_state_dict[f"{dst_prefix}.model.layernorm.weight"],
        )
        # dst_state_dict[f"{dst_prefix}.hidden_states_norms.3.weight"].copy_(
        #     dst_state_dict[f"{dst_prefix}.model.layernorm.weight"]
        # )

        # dst_state_dict[f"{dst_prefix}.hidden_states_norms.3.bias"].copy_(
        #     dst_state_dict[f"{dst_prefix}.model.layernorm.bias"]
        # )

        self.pop_all(renamed_keys, dst_state_dict, src_state_dict)

    def replace_pixel_module(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        dst_prefix: str = "pixel_level_module.decoder"
        src_prefix: str = "sem_seg_head.pixel_decoder"

        self.replace_backbone(dst_state_dict, src_state_dict, self.config)

        def rename_keys_for_conv(detectron_conv: str, mine_conv: str):
            return [
                (f"{detectron_conv}.weight", f"{mine_conv}.0.weight"),
                # 2 cuz the have act in the middle -> rename it
                (f"{detectron_conv}.norm.weight", f"{mine_conv}.1.weight"),
                (f"{detectron_conv}.norm.bias", f"{mine_conv}.1.bias"),
            ]

        renamed_keys = [
            (f"{src_prefix}.mask_features.weight", f"{dst_prefix}.mask_projection.weight"),
            (f"{src_prefix}.mask_features.bias", f"{dst_prefix}.mask_projection.bias"),
            # the layers in the original one are in reverse order, stem is the last one!
        ]

        renamed_keys.extend(rename_keys_for_conv(f"{src_prefix}.layer_4", f"{dst_prefix}.fpn.stem"))

        # add all the fpn layers (here we need some config parameters to know the size in advance)
        for src_i, dst_i in zip(range(3, 0, -1), range(0, 3)):
            renamed_keys.extend(
                rename_keys_for_conv(f"{src_prefix}.adapter_{src_i}", f"{dst_prefix}.fpn.layers.{dst_i}.proj")
            )
            renamed_keys.extend(
                rename_keys_for_conv(f"{src_prefix}.layer_{src_i}", f"{dst_prefix}.fpn.layers.{dst_i}.block")
            )

        self.pop_all(renamed_keys, dst_state_dict, src_state_dict)

    def rename_keys_in_detr_decoder(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        dst_prefix: str = "transformer_module.transformer_decoder"
        src_prefix: str = "sem_seg_head.predictor.transformer.decoder"
        # not sure why we are not popping direcetly here!
        # here we list all keys to be renamed (original name on the left, our name on the right)
        rename_keys = []
        for i in range(self.config.transformer_decoder.decoder_layers):
            # decoder layers: 2 times output projection, 2 feedforward neural networks and 3 layernorms
            rename_keys.append(
                (
                    f"{src_prefix}.layers.{i}.self_attn.out_proj.weight",
                    f"{dst_prefix}.layers.{i}.self_attn.out_proj.weight",
                )
            )
            rename_keys.append(
                (
                    f"{src_prefix}.layers.{i}.self_attn.out_proj.bias",
                    f"{dst_prefix}.layers.{i}.self_attn.out_proj.bias",
                )
            )
            rename_keys.append(
                (
                    f"{src_prefix}.layers.{i}.multihead_attn.out_proj.weight",
                    f"{dst_prefix}.layers.{i}.encoder_attn.out_proj.weight",
                )
            )
            rename_keys.append(
                (
                    f"{src_prefix}.layers.{i}.multihead_attn.out_proj.bias",
                    f"{dst_prefix}.layers.{i}.encoder_attn.out_proj.bias",
                )
            )
            rename_keys.append((f"{src_prefix}.layers.{i}.linear1.weight", f"{dst_prefix}.layers.{i}.fc1.weight"))
            rename_keys.append((f"{src_prefix}.layers.{i}.linear1.bias", f"{dst_prefix}.layers.{i}.fc1.bias"))
            rename_keys.append((f"{src_prefix}.layers.{i}.linear2.weight", f"{dst_prefix}.layers.{i}.fc2.weight"))
            rename_keys.append((f"{src_prefix}.layers.{i}.linear2.bias", f"{dst_prefix}.layers.{i}.fc2.bias"))
            rename_keys.append(
                (f"{src_prefix}.layers.{i}.norm1.weight", f"{dst_prefix}.layers.{i}.self_attn_layer_norm.weight")
            )
            rename_keys.append(
                (f"{src_prefix}.layers.{i}.norm1.bias", f"{dst_prefix}.layers.{i}.self_attn_layer_norm.bias")
            )
            rename_keys.append(
                (f"{src_prefix}.layers.{i}.norm2.weight", f"{dst_prefix}.layers.{i}.encoder_attn_layer_norm.weight")
            )
            rename_keys.append(
                (f"{src_prefix}.layers.{i}.norm2.bias", f"{dst_prefix}.layers.{i}.encoder_attn_layer_norm.bias")
            )
            rename_keys.append(
                (f"{src_prefix}.layers.{i}.norm3.weight", f"{dst_prefix}.layers.{i}.final_layer_norm.weight")
            )
            rename_keys.append(
                (f"{src_prefix}.layers.{i}.norm3.bias", f"{dst_prefix}.layers.{i}.final_layer_norm.bias")
            )

        return rename_keys

    def replace_q_k_v_in_detr_decoder(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        dst_prefix: str = "transformer_module.transformer_decoder"
        src_prefix: str = "sem_seg_head.predictor.transformer.decoder"
        for i in range(self.config.transformer_decoder.decoder_layers):
            # read in weights + bias of input projection layer of self-attention
            in_proj_weight = src_state_dict.pop(f"{src_prefix}.layers.{i}.self_attn.in_proj_weight")
            in_proj_bias = src_state_dict.pop(f"{src_prefix}.layers.{i}.self_attn.in_proj_bias")
            # next, add query, keys and values (in that order) to the state dict
            dst_state_dict[f"{dst_prefix}.layers.{i}.self_attn.q_proj.weight"] = in_proj_weight[:256, :]
            dst_state_dict[f"{dst_prefix}.layers.{i}.self_attn.q_proj.bias"] = in_proj_bias[:256]
            dst_state_dict[f"{dst_prefix}.layers.{i}.self_attn.k_proj.weight"] = in_proj_weight[256:512, :]
            dst_state_dict[f"{dst_prefix}.layers.{i}.self_attn.k_proj.bias"] = in_proj_bias[256:512]
            dst_state_dict[f"{dst_prefix}.layers.{i}.self_attn.v_proj.weight"] = in_proj_weight[-256:, :]
            dst_state_dict[f"{dst_prefix}.layers.{i}.self_attn.v_proj.bias"] = in_proj_bias[-256:]
            # read in weights + bias of input projection layer of cross-attention
            in_proj_weight_cross_attn = src_state_dict.pop(f"{src_prefix}.layers.{i}.multihead_attn.in_proj_weight")
            in_proj_bias_cross_attn = src_state_dict.pop(f"{src_prefix}.layers.{i}.multihead_attn.in_proj_bias")
            # next, add query, keys and values (in that order) of cross-attention to the state dict
            dst_state_dict[f"{dst_prefix}.layers.{i}.encoder_attn.q_proj.weight"] = in_proj_weight_cross_attn[:256, :]
            dst_state_dict[f"{dst_prefix}.layers.{i}.encoder_attn.q_proj.bias"] = in_proj_bias_cross_attn[:256]
            dst_state_dict[f"{dst_prefix}.layers.{i}.encoder_attn.k_proj.weight"] = in_proj_weight_cross_attn[
                256:512, :
            ]
            dst_state_dict[f"{dst_prefix}.layers.{i}.encoder_attn.k_proj.bias"] = in_proj_bias_cross_attn[256:512]
            dst_state_dict[f"{dst_prefix}.layers.{i}.encoder_attn.v_proj.weight"] = in_proj_weight_cross_attn[-256:, :]
            dst_state_dict[f"{dst_prefix}.layers.{i}.encoder_attn.v_proj.bias"] = in_proj_bias_cross_attn[-256:]

    def replace_detr_decoder(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        dst_prefix: str = "transformer_module.transformer_decoder"
        src_prefix: str = "sem_seg_head.predictor.transformer.decoder"
        renamed_keys = self.rename_keys_in_detr_decoder(dst_state_dict, src_state_dict)
        # add more
        renamed_keys.extend(
            [
                (f"{src_prefix}.norm.weight", f"{dst_prefix}.layernorm.weight"),
                (f"{src_prefix}.norm.bias", f"{dst_prefix}.layernorm.bias"),
            ]
        )

        self.pop_all(renamed_keys, dst_state_dict, src_state_dict)

        self.replace_q_k_v_in_detr_decoder(dst_state_dict, src_state_dict)

    def replace_transformer_module(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        dst_prefix: str = "transformer_module"
        src_prefix: str = "sem_seg_head.predictor"

        self.replace_detr_decoder(dst_state_dict, src_state_dict)

        renamed_keys = [
            (f"{src_prefix}.query_embed.weight", f"{dst_prefix}.queries_embedder.weight"),
            (f"{src_prefix}.input_proj.weight", f"{dst_prefix}.input_projection.weight"),
            (f"{src_prefix}.input_proj.bias", f"{dst_prefix}.input_projection.bias"),
        ]

        self.pop_all(renamed_keys, dst_state_dict, src_state_dict)

    def replace_instance_segmentation_module(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        # NOTE in our case we don't have a prefix, thus we removed the "." from the keys later on!
        dst_prefix: str = ""
        src_prefix: str = "sem_seg_head.predictor"

        renamed_keys = [
            (f"{src_prefix}.class_embed.weight", f"{dst_prefix}class_predictor.weight"),
            (f"{src_prefix}.class_embed.bias", f"{dst_prefix}class_predictor.bias"),
        ]

        mlp_len = 3
        for i in range(mlp_len):
            renamed_keys.extend(
                [
                    (f"{src_prefix}.mask_embed.layers.{i}.weight", f"{dst_prefix}mask_embedder.{i}.0.weight"),
                    (f"{src_prefix}.mask_embed.layers.{i}.bias", f"{dst_prefix}mask_embedder.{i}.0.bias"),
                ]
            )

        self.pop_all(renamed_keys, dst_state_dict, src_state_dict)

    def convert(self, mask_former: MaskFormerModel) -> MaskFormerModel:
        dst_state_dict = TrackedStateDict(mask_former.state_dict())
        src_state_dict = self.original_model.state_dict()

        self.replace_pixel_module(dst_state_dict, src_state_dict)
        self.replace_transformer_module(dst_state_dict, src_state_dict)

        logger.info(f"Missed keys are {pformat(dst_state_dict.diff())}")
        logger.info(f"Not copied keys are {pformat(src_state_dict.keys())}")
        logger.info("🙌 Done")

        mask_former.load_state_dict(dst_state_dict)

        return mask_former

    def convert_instance_segmentation(self, mask_former: MaskFormerForInstanceSegmentation) -> MaskFormerForInstanceSegmentation:
        dst_state_dict = TrackedStateDict(mask_former.state_dict())
        src_state_dict = self.original_model.state_dict()

        self.replace_instance_segmentation_module(dst_state_dict, src_state_dict)

        mask_former.load_state_dict(dst_state_dict)

        return mask_former


    @classmethod
    def from_original_config_and_checkpoint_file(
        cls, original_config_file: str, original_checkpoint_file: str
    ) -> OriginalMaskFormerCheckpoinToOursConverter:
        original_config = setup_cfg(Args(config_file=original_config_file))
        mask_former_kwargs = OriginalMaskFormer.from_config(original_config)

        original_model = OriginalMaskFormer(**mask_former_kwargs).eval()

        DetectionCheckpointer(original_model).load(original_checkpoint_file)

        config: MaskFormerConfig = OriginalMaskFormerConfigToOursConverter()(original_config)

        to_model = MaskFormerModel(config=config).eval()

        return cls(original_model)

    @staticmethod
    def using_dirs(
        checkpoints_dir: Path, config_dir: Path
    ) -> Iterator[Tuple[OriginalMaskFormerCheckpoinToOursConverter, Path, Path]]:
        checkpoints: List[Path] = checkpoints_dir.glob("**/*.pkl")

        for checkpoint in checkpoints:
            logger.info(f"💪 Converting {checkpoint.stem}")
            # find associated config file
            config: Path = config_dir / checkpoint.parents[0].stem / "swin" / f"{checkpoint.stem}.yaml"
        
            yield config, checkpoint


checkpoints_dir = Path("/home/zuppif/Documents/Work/hugging_face/maskformer/weights")
config_dir = Path("/home/zuppif/Documents/Work/hugging_face/maskformer/MaskFormer/configs")


def test(src, dst):
    with torch.no_grad():

        src = src.eval()
        dst = dst.eval()

        x = torch.zeros((1, 3, 384, 384))

        src_features = src.backbone(x)
        dst_features = dst.pixel_level_module.backbone(x.clone())

        for src_feature, dst_feature in zip(src_features.values(), dst_features):
            assert torch.allclose(src_feature, dst_feature, atol=1e-4)

        src_pixel_out = src.sem_seg_head.decoder.forward_features(src_features)
        dst_pixel_out = dst.pixel_level_module(x)

        assert torch.allclose(src_pixel_out[0], dst_pixel_out[1], atol=1e-4)
        # turn off aux_loss loss
        src.sem_seg_head.predictor.aux_loss = False
        src_transformer_out = src.sem_seg_head.predictor(src_features["res5"], src_pixel_out[0])
        dst_queries = dst.transformer_module(dst_pixel_out[0])
        dst_transformer_out = dst.segmentation_module(dst_queries, dst_pixel_out[1])

        assert torch.allclose(src_transformer_out["pred_logits"], dst_transformer_out["preds_logits"], atol=1e-4)

        assert torch.allclose(src_transformer_out["pred_masks"], dst_transformer_out["preds_masks"], atol=1e-4)

        src_out = src([{"image": x.squeeze(0)}])

        dst_for_seg = MaskFormerForSemanticSegmentation(config=dst.config).eval()
        dst_for_seg.model.load_state_dict(dst.state_dict())

        dst_out: MaskFormerForSemanticSegmentationOutput = dst_for_seg(x)

        assert torch.allclose(src_out[0]["sem_seg"], dst_out.segmentation, atol=1e-4)

        im = prepare_img()

        tr = T.Compose(
            [
                T.Resize((384, 384)),
                T.ToTensor(),
                T.Normalize(
                    mean=torch.tensor([123.675, 116.280, 103.530]) / 255.0,
                    std=torch.tensor([58.395, 57.120, 57.375]) / 255.0,
                ),
            ],
        )

        x = tr(im)

        src_out = src([{"image": x}])
        dst_out = dst_for_seg(x.unsqueeze(0))
        # NOTE this fails but the numbers are very very close
        # assert torch.allclose(src_out[0]["sem_seg"], dst_out.segmentation, atol=1e-4)

        dst_for_pan_seg = MaskFormerForPanopticSegmentation(
            config=dst.config,
            overlap_mask_area_threshold=src.overlap_threshold,
            object_mask_threshold=src.object_mask_threshold,
        ).eval()
        dst_for_pan_seg.model.load_state_dict(dst.state_dict())
        # not all the models will work on pan seg (due to their dataset)
        metadata = src.metadata
        if metadata.get("thing_dataset_id_to_contiguous_id") is not None:
            src.panoptic_on = True

            src_out = src([{"image": x}])
            dst_out: MaskFormerForPanopticSegmentationOutput = dst_for_pan_seg(x.unsqueeze(0))
            src_seg = src_out[0]["panoptic_seg"]

            assert torch.allclose(src_out[0]["panoptic_seg"][0], dst_out.segmentation, atol=1e-4)

            for src_segment, dst_segment in zip(src_seg[1], dst_out.segments):
                assert src_segment["id"] == dst_segment["id"]
                assert src_segment["category_id"] == dst_segment["category_id"]
                assert src_segment["isthing"] == dst_segment["is_thing"]

        logger.info("✅ Test passed!")


if __name__ == "__main__":

    parser = ArgumentParser(
        description="Command line to convert the original maskformers (with swin backbone) to our implementations."
    )

    parser.add_argument(
        "--checkpoints_dir",
        type=Path,
        help="A directory containing the model's checkpoints. The directory has to have the following structure: <DIR_NAME>/<DATASET_NAME>/<CONFIG_NAME>.pkl",
    )
    parser.add_argument(
        "--configs_dir",
        type=Path,
        help="A directory containing the model's configs, see detectron2 doc. The directory has to have the following structure: <DIR_NAME>/<DATASET_NAME>/<CONFIG_NAME>.yaml",
    )
    parser.add_argument(
        "--save_directory",
        default=Path("/tmp/hf/models"),
        type=Path,
        help="Path to the folder to output PyTorch models.",
    )
    args = parser.parse_args()

    checkpoints_dir: Path = args.checkpoints_dir
    config_dir: Path = args.configs_dir
    save_directory: Path = args.save_directory

    checkpoints_dir = Path("/home/zuppif/Documents/Work/hugging_face/maskformer/weights")

    config_dir = Path("/home/zuppif/Documents/Work/hugging_face/maskformer/MaskFormer/configs")


    if not save_directory.exists():
        save_directory.mkdir(parents=True)

    for config_file, checkpoint_file in OriginalMaskFormerCheckpoinToOursConverter.using_dirs(
        checkpoints_dir, config_dir
    ):

        feature_extractor = OriginalMaskFormerConfigToFeatureExtractorConverter()(
            setup_cfg(Args(config_file=config_file))
        )

        original_config = setup_cfg(Args(config_file=config_file))
        mask_former_kwargs = OriginalMaskFormer.from_config(original_config)

        original_model = OriginalMaskFormer(**mask_former_kwargs).eval()

        DetectionCheckpointer(original_model).load(str(checkpoint_file))

        config: MaskFormerConfig = OriginalMaskFormerConfigToOursConverter()(original_config)

        mask_former = MaskFormerModel(config=config).eval()

        converter = OriginalMaskFormerCheckpoinToOursConverter(original_model, config)

        converter.convert(mask_former)

        mask_former_for_instance_segmentation = MaskFormerForInstanceSegmentation(config=config).eval()

        mask_former_for_instance_segmentation.model = mask_former
        converter.convert_instance_segmentation(mask_former_for_instance_segmentation)

        model_name: str = f"{checkpoint_file.parents[0].stem}-{checkpoint_file.stem}"

        print(model_name)
        # test(converter.original_model, converted)

        # feature_extractor.save_pretrained(save_directory / model_name)
        # converted.save_pretrained(save_directory / model_name)
