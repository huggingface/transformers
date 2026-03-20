import torch
from torch import nn
import math
from fractions import Fraction
from transformers.models.blip_2.configuration_blip_2 import Blip2QFormerConfig
from transformers.models.blip_2.modeling_blip_2 import Blip2QFormerModel


class InterpolateDownsampler:
    """Spatial downsampling via area interpolation."""
    def __init__(self, config, mode="area"):
        self.orig_image_side = config.vision_config.image_size // config.vision_config.patch_size
        self.new_image_side = int(self.orig_image_side * Fraction(config.downsample_rate))
        self.mode = mode

    def __call__(self, image_features):
        batch_size, _, dim = image_features.size()
        up_shape = [batch_size] + [self.orig_image_side] * 2 + [dim]
        large_image_permuted = image_features.view(up_shape).permute(0,3,1,2)
        small_image_permuted = torch.nn.functional.interpolate(
                large_image_permuted, size=(self.new_image_side, self.new_image_side),
                mode=self.mode,
        )
        final = small_image_permuted.permute(0,2,3,1).flatten(1,2)
        return final


class SpatialOffsetDownsampler:
    """
    Downsampler that samples one position from each 2x2 block across the image.
    Maintains full spatial coverage while creating local continuity.
    """
    def __init__(self, config, offset=0):
        """
        Args:
            config: Model configuration
            offset: Integer offset (0, 1, 2, or 3) for position within each 2x2 block
                   0: top-left, 1: top-right, 2: bottom-left, 3: bottom-right
        """
        self.orig_image_side = config.vision_config.image_size // config.vision_config.patch_size
        self.new_image_side = self.orig_image_side // 2
        self.offset = offset
        self.offsets = [(0, 0), (0, 1), (1, 0), (1, 1)]
        self.offset_h, self.offset_w = self.offsets[offset]

    def __call__(self, image_features):
        batch_size, seq_len, hidden_dim = image_features.shape
        features_2d = image_features.reshape(batch_size, self.orig_image_side, self.orig_image_side, hidden_dim)

        n_blocks = self.new_image_side
        features_blocks = features_2d.reshape(
            batch_size, n_blocks, 2, n_blocks, 2, hidden_dim
        )

        sampled = features_blocks[:, :, self.offset_h, :, self.offset_w, :]
        sampled = sampled.reshape(batch_size, -1, hidden_dim)

        return sampled


class WindowQFormerDownsampler(nn.Module):
    """Window-based QFormer downsampler that processes image patches in windows."""
    def __init__(self, config, spatial_offset=None):
        super().__init__()
        llm_hidden_size = config.text_config.hidden_size
        vision_hidden_size = config.vision_config.hidden_size

        self.dropout = nn.Dropout(config.projector_dropout)

        if spatial_offset is not None:
            self.downsampler = SpatialOffsetDownsampler(config, offset=spatial_offset)
        else:
            self.downsampler = InterpolateDownsampler(config)

        configuration = Blip2QFormerConfig(
            hidden_size=vision_hidden_size,
            num_attention_heads=vision_hidden_size // 64,
            intermediate_size=3072,
            num_hidden_layers=1,
            encoder_hidden_size=vision_hidden_size,
            cross_attention_frequency=1,
            max_position_embeddings=2048,
            use_qformer_text_input=False,
        )
        self.qformer = Blip2QFormerModel(configuration)

        self.image_side = config.vision_config.image_size // config.vision_config.patch_size
        q, w = config.downsample_rate.split("/")
        self.query_side, self.window_side = int(q), int(w)
        self.query_length = self.query_side ** 2
        embed_std = 1 / math.sqrt(vision_hidden_size)
        self.norm = nn.LayerNorm(vision_hidden_size, eps=1e-6)
        self.query = nn.Parameter(torch.randn(1, self.query_length, vision_hidden_size) * embed_std)
        self.image_positions = nn.Parameter(torch.randn(1, self.window_side ** 2, vision_hidden_size) * embed_std)
        self.out_linear = nn.Linear(vision_hidden_size, llm_hidden_size, bias=True)

    def _win(self, x, side, win):
        """
        (B, side*side, C) raster -> (B*n*n, win*win, C) where n=side//win
        windows are raster-ordered, and tokens inside each window are raster-ordered.
        """
        B, _, C = x.shape
        n = side // win
        return (
            x.view(B, side, side, C)
            .view(B, n, win, n, win, C)
            .transpose(2, 3)          # (B, n, n, win, win, C)
            .flatten(0, 2)            # (B*n*n, win, win, C)
            .flatten(1, 2)            # (B*n*n, win*win, C)
        )

    def _unwin(self, xw, n, win):
        """
        (B*n*n, win*win, C) -> (B, (n*win)^2, C) raster
        """
        Bnn, _, C = xw.shape
        assert Bnn % (n * n) == 0
        B = Bnn // (n * n)
        side = n * win
        return (
            xw.view(B, n, n, win, win, C)
            .transpose(2, 3)                 # (B, n, win, n, win, C)
            .contiguous()
            .view(B, side, side, C)
            .flatten(1, 2)
        )

    def forward(self, image_features):
        B, HW, C = image_features.shape
        assert HW == self.image_side * self.image_side
        n = self.image_side // self.window_side
        image_features = self.norm(image_features)
        enc = self._win(image_features, self.image_side, self.window_side)

        downsampled = self.downsampler(image_features)

        new_side = n * self.query_side
        downsampled_w = self._win(downsampled, new_side, self.query_side)

        query_embeds = self.query + downsampled_w
        encoder_embeds = self.dropout(enc + self.image_positions)
        out_w = self.qformer(
            query_embeds=query_embeds,
            encoder_hidden_states=encoder_embeds,
            return_dict=True,
        ).last_hidden_state

        out = self._unwin(out_w, n=n, win=self.query_side)

        out = self.dropout(out)
        return self.out_linear(out)
