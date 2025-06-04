"""VJEPA 2 model configuration"""

import os
from typing import Union

from ...configuration_utils import PretrainedConfig
from ...modeling_outputs import dataclass


class VJEPA2PredictorConfig(PretrainedConfig):

    patch_size: int = 16
    crop_size: int = 224
    frames_per_clip: int = 16
    tubelet_size: int = 2
    use_sdpa: bool = False
    use_SiLU: bool = False
    wide_SiLU: bool = True
    uniform_power: bool = False
    hidden_size: int = -1
    enc_hidden_size: int = -1
    in_chans: int = 3
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    drop_path_rate: float = 0.0
    mlp_ratio: float = 4.0
    is_causal: bool = False
    layer_norm_eps: float = 1e-6
    qkv_bias: bool = True
    attention_probs_dropout_prob: float = 0.0
    hidden_dropout_prob: float = 0.0
    hidden_act: str = "gelu"
    initializer_range: float = 0.02
    use_rope: bool = False
    use_mask_tokens: bool = False
    num_mask_tokens: int = 2
    zero_init_mask_tokens: bool = True


class VJEPA2Config(PretrainedConfig):
    model_type = "vjepa_vision_model"

    def __init__(
        self,
        model_name="vit_large",
        patch_size=16,
        crop_size=224,
        frames_per_clip=16,
        tubelet_size=2,
        use_sdpa=False,
        use_SiLU=False,
        wide_SiLU=True,
        uniform_power=False,
        hidden_size=-1,
        in_chans=3,
        num_attention_heads=12,
        num_hidden_layers=12,
        drop_path_rate=0.0,
        mlp_ratio=4.0,
        is_causal=False,
        layer_norm_eps=1e-6,
        qkv_bias=True,
        attention_probs_dropout_prob=0.0,
        hidden_dropout_prob=0.0,
        hidden_act="gelu",
        initializer_range=0.02,
        use_rope=False,
        # predictor params
        pred_hidden_size=-1,
        pred_num_attention_heads=12,
        pred_num_hidden_layers=6,
        pred_use_mask_tokens=False,
        pred_num_mask_tokens=2,
        pred_zero_init_mask_tokens=True,
        pred_mlp_ratio=4.0,
        use_predictor_loss=False,
        predictor_loss_window=256,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.model_name = model_name
        self.crop_size = crop_size
        self.img_height = crop_size
        self.img_width = crop_size
        self.frames_per_clip = frames_per_clip
        self.patch_size = patch_size
        self.num_frames = frames_per_clip
        self.tubelet_size = tubelet_size
        self.uniform_power = uniform_power
        self.use_sdpa = use_sdpa
        self.use_SiLU = use_SiLU
        self.wide_SiLU = wide_SiLU
        self.hidden_size = hidden_size
        self.in_chans = in_chans
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.drop_path_rate = drop_path_rate
        self.mlp_ratio = mlp_ratio
        self.is_causal = is_causal
        self.layer_norm_eps = layer_norm_eps
        self.qkv_bias = qkv_bias
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.image_size = crop_size
        self.use_rope = use_rope
        # predictor params
        self.pred_hidden_size = pred_hidden_size
        self.pred_num_attention_heads = pred_num_attention_heads
        self.pred_num_hidden_layers = pred_num_hidden_layers
        self.pred_use_mask_tokens = pred_use_mask_tokens
        self.pred_num_mask_tokens = pred_num_mask_tokens
        self.pred_zero_init_mask_tokens = pred_zero_init_mask_tokens
        self.pred_mlp_ratio = pred_mlp_ratio
        self.use_predictor_loss = use_predictor_loss
        self.predictor_loss_window = predictor_loss_window

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(
            pretrained_model_name_or_path, **kwargs
        )

        # get the vision config dict if we are loading from SigLipConfig
        if config_dict.get("model_type") == "vjepa":
            config_dict = config_dict["vision_config"]

        if (
            "model_type" in config_dict
            and hasattr(cls, "model_type")
            and config_dict["model_type"] != cls.model_type
        ):
            print(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)

    def get_predictor_config(self) -> VJEPA2PredictorConfig:
        return VJEPA2PredictorConfig(
            patch_size=self.patch_size,
            crop_size=self.crop_size,
            frames_per_clip=self.frames_per_clip,
            tubelet_size=self.tubelet_size,
            use_sdpa=self.use_sdpa,
            use_SiLU=self.use_SiLU,
            wide_SiLU=self.wide_SiLU,
            uniform_power=self.uniform_power,
            hidden_size=self.pred_hidden_size,
            enc_hidden_size=self.hidden_size,
            in_chans=self.in_chans,
            num_attention_heads=self.pred_num_attention_heads,
            num_hidden_layers=self.pred_num_hidden_layers,
            drop_path_rate=self.drop_path_rate,
            mlp_ratio=self.pred_mlp_ratio,
            is_causal=self.is_causal,
            layer_norm_eps=self.layer_norm_eps,
            qkv_bias=self.qkv_bias,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            hidden_dropout_prob=self.hidden_dropout_prob,
            hidden_act=self.hidden_act,
            initializer_range=self.initializer_range,
            use_rope=self.use_rope,
            use_mask_tokens=self.pred_use_mask_tokens,
            num_mask_tokens=self.pred_num_mask_tokens,
            zero_init_mask_tokens=self.pred_zero_init_mask_tokens,
        )


__all__ = ["VJEPA2Config"]
