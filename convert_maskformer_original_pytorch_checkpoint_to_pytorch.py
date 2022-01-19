from __future__ import annotations
from ast import Tuple
from PIL import Image
import requests
import torch
import torch.nn as nn
from torch import Tensor
from dataclasses import dataclass, field
from pprint import pformat, pprint
from typing import List, Any, Dict, Optional, Set
from src.transformers.models.maskformer.modelling_maskformer import (
    MaskFormerConfig,
    MaskFormerForSemanticSegmentation,
    MaskFormerModel,
)
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from MaskFormer.mask_former import add_mask_former_config, config
from MaskFormer.mask_former.mask_former_model import MaskFormer as OriginalMaskFormer
from detectron2.layers.wrappers import Conv2d as DetectronConv2d
import pytorch_lightning as pl
import torchvision.transforms as T

pl.seed_everything(42)

StateDict = Dict[str, Tensor]


class TrackedStateDict:
    def __init__(self, to_track: Dict):
        """This class "tracks" a python dictionary by keeping track of which item is accessed.

        Args:
            to_track (Dict): A dictionary we wish to track
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
    im = Image.open(requests.get(url, stream=True).raw)
    return im


class Args:
    config_file = "/home/zuppif/Documents/Work/hugging_face/maskformer/MaskFormer/configs/coco-panoptic/swin/maskformer_panoptic_swin_base_IN21k_384_bs64_554k.yaml"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_mask_former_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    return cfg


def convert_detectron_conv2d_conv2d_and_norm(detectron_conv2d: DetectronConv2d, conv2d: nn.Conv2d, norm: nn.Module):
    norm.load_state_dict(detectron_conv2d.norm.state_dict())
    del detectron_conv2d.norm
    conv2d.load_state_dict(detectron_conv2d.state_dict())


def pop_all(renamed_keys: List[Tuple[str, str]], dst_state_dict: StateDict, src_state_dict: StateDict):
    for (src_key, dst_key) in renamed_keys:
        dst_state_dict[dst_key] = src_state_dict.pop(src_key)


def replace_pixel_module(dst_state_dict: StateDict, src_state_dict: StateDict):
    dst_prefix: str = "pixel_level_module.pixel_decoder"
    src_prefix: str = "sem_seg_head.pixel_decoder"

    def replace_backbone():
        dst_prefix: str = "pixel_level_module.backbone"
        src_prefix: str = "backbone"

        renamed_keys = []

        for key in src_state_dict.keys():
            if key.startswith(src_prefix):
                renamed_keys.append((key, key.replace(src_prefix, dst_prefix)))

        return renamed_keys

    def rename_keys_for_conv(detectron_conv: str, mine_conv: str):
        return [
            (f"{detectron_conv}.weight", f"{mine_conv}.0.weight"),
            # 2 cuz the have act in the middle -> rename it
            (f"{detectron_conv}.norm.weight", f"{mine_conv}.1.weight"),
            (f"{detectron_conv}.norm.bias", f"{mine_conv}.1.bias"),
        ]

    renamed_keys = [
        (f"{src_prefix}.mask_features.weight", f"{dst_prefix}.mask_proj.weight"),
        (f"{src_prefix}.mask_features.bias", f"{dst_prefix}.mask_proj.bias"),
        # the layers in the original one are in reverse order, stem is the last one!
    ]

    renamed_keys.extend(replace_backbone())

    renamed_keys.extend(rename_keys_for_conv(f"{src_prefix}.layer_4", f"{dst_prefix}.fpn.stem"))

    # add all the fpn layers (here we need some config parameters to know the size in advance)
    for src_i, dst_i in zip(range(3, 0, -1), range(0, 3)):
        renamed_keys.extend(
            rename_keys_for_conv(f"{src_prefix}.adapter_{src_i}", f"{dst_prefix}.fpn.layers.{dst_i}.proj")
        )
        renamed_keys.extend(
            rename_keys_for_conv(f"{src_prefix}.layer_{src_i}", f"{dst_prefix}.fpn.layers.{dst_i}.block")
        )

    pop_all(renamed_keys, dst_state_dict, src_state_dict)


def rename_keys_in_detr_decoder(dst_state_dict: StateDict, src_state_dict: StateDict):
    dst_prefix: str = "transformer_module.detr_decoder"
    src_prefix: str = "sem_seg_head.predictor.transformer.decoder"
    # not sure why we are not popping direcetly here!
    # here we list all keys to be renamed (original name on the left, our name on the right)
    rename_keys = []
    for i in range(6):
        # decoder layers: 2 times output projection, 2 feedforward neural networks and 3 layernorms
        rename_keys.append(
            (
                f"{src_prefix}.layers.{i}.self_attn.out_proj.weight",
                f"{dst_prefix}.layers.{i}.self_attn.out_proj.weight",
            )
        )
        rename_keys.append(
            (f"{src_prefix}.layers.{i}.self_attn.out_proj.bias", f"{dst_prefix}.layers.{i}.self_attn.out_proj.bias")
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
        rename_keys.append((f"{src_prefix}.layers.{i}.norm3.bias", f"{dst_prefix}.layers.{i}.final_layer_norm.bias"))

    return rename_keys


def replace_q_k_v_in_detr_decoder(dst_state_dict: StateDict, src_state_dict: StateDict):
    dst_prefix: str = "transformer_module.detr_decoder"
    src_prefix: str = "sem_seg_head.predictor.transformer.decoder"
    for i in range(6):
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
        dst_state_dict[f"{dst_prefix}.layers.{i}.encoder_attn.k_proj.weight"] = in_proj_weight_cross_attn[256:512, :]
        dst_state_dict[f"{dst_prefix}.layers.{i}.encoder_attn.k_proj.bias"] = in_proj_bias_cross_attn[256:512]
        dst_state_dict[f"{dst_prefix}.layers.{i}.encoder_attn.v_proj.weight"] = in_proj_weight_cross_attn[-256:, :]
        dst_state_dict[f"{dst_prefix}.layers.{i}.encoder_attn.v_proj.bias"] = in_proj_bias_cross_attn[-256:]


def replace_detr_decoder(dst_state_dict: StateDict, src_state_dict: StateDict):
    dst_prefix: str = "transformer_module.detr_decoder"
    src_prefix: str = "sem_seg_head.predictor.transformer.decoder"
    renamed_keys = rename_keys_in_detr_decoder(dst_state_dict, src_state_dict)
    # add more
    renamed_keys.extend(
        [
            (f"{src_prefix}.norm.weight", f"{dst_prefix}.layernorm.weight"),
            (f"{src_prefix}.norm.bias", f"{dst_prefix}.layernorm.bias"),
        ]
    )

    pop_all(renamed_keys, dst_state_dict, src_state_dict)

    replace_q_k_v_in_detr_decoder(dst_state_dict, src_state_dict)


def replace_transformer_module(dst_state_dict: StateDict, src_state_dict: StateDict):
    dst_prefix: str = "transformer_module"
    src_prefix: str = "sem_seg_head.predictor"

    replace_detr_decoder(dst_state_dict, src_state_dict)

    renamed_keys = [
        (f"{src_prefix}.query_embed.weight", f"{dst_prefix}.queries_embedder.weight"),
        (f"{src_prefix}.input_proj.weight", f"{dst_prefix}.input_proj.weight"),
        (f"{src_prefix}.input_proj.bias", f"{dst_prefix}.input_proj.bias"),
    ]

    pop_all(renamed_keys, dst_state_dict, src_state_dict)


def replace_segmentation_module(dst_state_dict: StateDict, src_state_dict: StateDict):
    dst_prefix: str = "segmentation_module"
    src_prefix: str = "sem_seg_head.predictor"

    renamed_keys = [
        (f"{src_prefix}.class_embed.weight", f"{dst_prefix}.class_predictor.weight"),
        (f"{src_prefix}.class_embed.bias", f"{dst_prefix}.class_predictor.bias"),
    ]

    mlp_len = 3
    for i in range(mlp_len):
        renamed_keys.extend(
            [
                (f"{src_prefix}.mask_embed.layers.{i}.weight", f"{dst_prefix}.mask_embedder.{i}.0.weight"),
                (f"{src_prefix}.mask_embed.layers.{i}.bias", f"{dst_prefix}.mask_embedder.{i}.0.bias"),
            ]
        )

    pop_all(renamed_keys, dst_state_dict, src_state_dict)


def replace_maskformer(dst: nn.Module, src: nn.Module):
    dst_state_dict = TrackedStateDict(dst.state_dict())
    src_state_dict = src.state_dict()

    print(len(src_state_dict.keys()))

    replace_pixel_module(dst_state_dict, src_state_dict)
    replace_transformer_module(dst_state_dict, src_state_dict)
    replace_segmentation_module(dst_state_dict, src_state_dict)

    print(f"missed keys are {pformat(dst_state_dict.diff())}")

    dst.load_state_dict(dst_state_dict)


def very_boring_test():
    with torch.no_grad():
        cfg = setup_cfg(Args())

        mask_former_kwargs = OriginalMaskFormer.from_config(cfg)
        src = OriginalMaskFormer(**mask_former_kwargs).eval()
        config = MaskFormerConfig()
        dst = MaskFormerModel(config=config).eval()

        replace_maskformer(dst, src)

        x = torch.zeros((1, 3, 384, 384))

        src_features = src.backbone(x)
        dst_features = dst.pixel_level_module.backbone(x.clone())

        for src_feature, dst_feature in zip(src_features.values(), dst_features):
            assert torch.allclose(src_feature, dst_feature)

        src_pixel_out = src.sem_seg_head.pixel_decoder.forward_features(src_features)
        dst_pixel_out = dst.pixel_level_module(x)

        assert torch.allclose(src_pixel_out[0], dst_pixel_out[1])
        # turn off aux_loss loss
        src.sem_seg_head.predictor.aux_loss = False
        src_transformer_out = src.sem_seg_head.predictor(src_features["res5"], src_pixel_out[0])

        # print(src.sem_seg_head.predictor)
        dst_queries = dst.transformer_module(dst_pixel_out[0])
        dst_transformer_out = dst.segmentation_module(dst_queries, dst_pixel_out[1])

        # LOOKING at them they seems equal!!!
        print((src_transformer_out["pred_logits"] - dst_transformer_out["pred_logits"]).sum())

        assert torch.allclose(src_transformer_out["pred_logits"], dst_transformer_out["pred_logits"], atol=1e-4)

        assert torch.allclose(src_transformer_out["pred_masks"], dst_transformer_out["pred_masks"], atol=1e-4)

        src_out = src([{"image": x.squeeze(0)}])

        dst_for_seg = MaskFormerForSemanticSegmentation(config=config).eval()
        dst_for_seg.model.load_state_dict(dst.state_dict())

        dst_out = dst_for_seg(x)

        assert torch.allclose(src_out[0]["sem_seg"], dst_out.segmentation, atol=1e-4)


def test_with_img():
    im = prepare_img()

    tr = T.Compose(
        [
            T.Resize((384, 384)),
            T.ToTensor(),
            T.Normalize(mean=[123.675, 116.280, 103.530], std=[58.395, 57.120, 57.375]),
        ],
    )

    x = tr(im)

    cfg = setup_cfg(Args())

    with torch.no_grad():

        mask_former_kwargs = OriginalMaskFormer.from_config(cfg)
        original_mask_former = OriginalMaskFormer(**mask_former_kwargs).eval()
        original_mask_former.sem_seg_head.predictor.aux_loss = False
        config = MaskFormerConfig()

        mask_former = MaskFormerModel(config=config).eval()
        replace_maskformer(mask_former, original_mask_former)

        mask_former_for_seg = MaskFormerForSemanticSegmentation(config=config).eval()

        mask_former_for_seg.model.load_state_dict(mask_former.state_dict())

        original_outs = original_mask_former([{"image": x}])
        outs = mask_former_for_seg(x.unsqueeze(0))

        assert torch.allclose(original_outs[0]["sem_seg"], outs.segmentation, atol=1e-4)

        from torchvision.utils import draw_segmentation_masks
        from torchvision.transforms.functional import to_tensor

        # for cat

        img_with_mask = (to_tensor(im.resize((384, 384))) * 255).type(torch.uint8)

        for mask in outs.segmentation[0] > 0.5:
            img_with_mask = draw_segmentation_masks(img_with_mask, masks=mask, alpha=0.6)

        import numpy as np
        import matplotlib.pyplot as plt

        import torchvision.transforms.functional as F

        plt.rcParams["savefig.bbox"] = "tight"

        def show(imgs):
            if not isinstance(imgs, list):
                imgs = [imgs]
            fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
            for i, img in enumerate(imgs):
                img = img.detach()
                img = F.to_pil_image(img)
                axs[0, i].imshow(np.asarray(img))
                axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

        show(img_with_mask)
        plt.show()


# very_boring_test()
test_with_img()
