import torch
import torch.nn as nn
from torchtyping import TensorType, patch_typeguard
from einops import rearrange

from typing import Callable, Union
import timm
import open_clip
from functools import partial

# ----------------------------- Utils --------------------------------------

# clip.model.LayerNorm = (
#     nn.LayerNorm
# )  # we need to patch this for clip to work with deepspeed
# patch_typeguard()  # needed for torchtyping typechecks to work


class Lambda(torch.nn.Module):
    def __init__(self, fn: Callable):
        super().__init__()
        assert hasattr(fn, "__call__")
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


# ------------------------- Image encoders ----------------------------------


def nfresnet50(
    device: Union[torch.device, str] = None, 
    pretrained: bool = True,
    cache_path: str = None
) -> nn.Module:
    """
    Loads nfresnet50 model, removing the pooling layer and replacing it with
    an adaptive pooling layer.
    """
    encoder = torch.nn.Sequential(
        *list(timm.create_model(
            "nf_resnet50", 
            pretrained=pretrained,
            # checkpoint_path=cache_path
            ).children())[:-1]
    )
    pooling = torch.nn.AdaptiveAvgPool2d((1, 1))
    encoder = torch.nn.Sequential(encoder, pooling)
    if device is not None:
        encoder = encoder.to(device)
    return encoder


def clip_encoder(
    device: Union[torch.device, str] = None,
    name: str = "clip",
    pretrain: bool = False,
    cache_path: str = None
) -> nn.Module:
    """
    Loads clip's image encoder module, discarding the lm component.

    If the variant is a resnet model, we also remove the attention pooling.
    """
    if name in ["clip", "ViT-B/32"]:
        name, pretrained = "ViT-B-32", "openai"
    elif name in ["clip_resnet", "RN50x4"]:
        name, pretrained = "RN50x4", "openai"
    elif name in ["clip_resnet_large", "RN50x16"]:
        name, pretrained = "RN50x16", "openai"
    elif "openclip" in name:
        if "H" in name:
            name, pretrained = "ViT-H-14", "laion2b_s32b_b79k"
        elif "B" in name and "32" in name:
            name, pretrained = "ViT-B-32", "laion2b_s34b_b79k"
        else:
            raise NotImplementedError(f"Encoder {name} not recognized")
    else:
        raise NotImplementedError(f"Encoder {name} not recognized")

    # TODO better internet connection
    if pretrain:    
        encoder = open_clip.create_model(
            name, 
            device=device, 
            precision="fp16" if "cuda" in str(device) else "fp32", 
            pretrained=pretrained,
            cache_dir=cache_path
        ).visual
    else:
        encoder = open_clip.create_model(
            name, 
            device=device, 
            precision="fp16" if "cuda" in str(device) else "fp32", 
            # pretrained=pretrained,
            cache_dir=cache_path
        ).visual
         
    if "RN" in name:
        # remove attention pooling
        encoder.attnpool = Lambda(
            partial(rearrange, pattern="b d h w -> b (h w) d")
        )  # remove attn pooling, just use reshaped features

    if False and hasattr(encoder, "transformer"):  # TODO when do we want to disable pooling?
        def forward(self, x: torch.Tensor):
            x = self.conv1(x)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
            x = torch.cat(
                [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                 x], dim=1)  # shape = [*, grid ** 2 + 1, width]
            x = x + self.positional_embedding.to(x.dtype)

            ## a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
            # x = self.patch_dropout(x)
            x = self.ln_pre(x)

            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = self.ln_post(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            return x
        encoder.forward = partial(forward, encoder)


    if device is not None:
        encoder = encoder.to(device)

    return encoder


def get_image_encoder(
    name: str, 
    device: Union[torch.device, str] = None, 
    pretrained: bool = False,
    cache_path: str = None
) -> torch.nn.Module:
    """
    Loads image encoder module
    """
    if name == "nfresnet50":
        encoder = nfresnet50(device=device, pretrained=pretrained, cache_path=cache_path)
    elif "clip" in name:
        encoder = clip_encoder(device=device, name=name, pretrain=pretrained, cache_path=cache_path)
    else:
        raise ValueError(f"image encoder {name} not recognized")
    return encoder


# ------------------------- Image prefix ----------------------------------

# for models that are fixed to a specific sequence lengths (i.e clip models with no pooling), the sequence lengths are below
ENCODER_SEQ_LENS = {
    "clip_resnet": 49,
    "clip_resnet_large": 144,
    "openclip-H": 257
}

ENCODER_OUT_DIMS = {
    "nfresnet50": 2048,
    "clip": 512,
    "clip_resnet": 2560,
    "clip_resnet_large": 3072,
    "openclip-H": 1024,
}


class ImagePrefix(nn.Module):

    """
    Takes in a batch of images and returns a batch of embeddings of the
    same dimensions as the LM's word embeddings.

    :param config: Neox args
    :param out_dim: output dimension of the embedding
    :param device: device to run the model on
    """

    def __init__(
        self,
        config,
        out_dim: int = 2048,
        device=None,
    ):
        super().__init__()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.config = config
        self.encoder_type = config.encoder_name

        # get image encoder backbone
        self.enc = get_image_encoder(
            config.encoder_name,
            # device=self.device,
            pretrained=config.pretrained_img_encoder,
            cache_path = config.load_clip
        )
        self.encoder_out_dim = ENCODER_OUT_DIMS[
            self.encoder_type
        ]  # out dim for image encoder

        self.out_dim = out_dim  # out dim for lm

        # get the output projection
        proj_out_dim = (
            self.out_dim
        )
        self.proj = nn.Linear(self.encoder_out_dim, proj_out_dim)
        self.dropout = nn.Dropout(config.image_embed_dropout_prob)
        self.use_layernorm = config.use_image_embed_layernorm
        if self.use_layernorm:
            self.ln = nn.LayerNorm(self.out_dim)

    def forward(
        self, x: TensorType["b", "c", "h", "w"]
    ) -> TensorType["b", "seq", "out_dim"]:

        # pass through image encoder
        logits = self.enc(x)

        # remove trailing dimensions of size 1 + pass through linear
        if logits.ndim == 4:
            logits = rearrange(logits, "b d 1 1 -> b d")
        elif logits.ndim == 3:
            assert self.encoder_type in ENCODER_SEQ_LENS
        else:
            assert logits.ndim == 2

        logits = self.proj(logits)

        # reshape to desired output shape
        if (
            self.encoder_type not in ENCODER_SEQ_LENS
        ):  # don't need to reshape those with fixed seq lens / no pooling
            logits = rearrange(
                logits, "b (s d) -> b s d", d=self.out_dim, s=self.out_seq_len
            )

        # pass through dropout and layer norm
        logits = self.dropout(logits)

        if self.use_layernorm:
            logits = self.ln(logits)

        # Added for shape mismatch.
        if logits.ndim == 2:
            logits = logits.unsqueeze(1)

        return logits
