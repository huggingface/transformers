import torch
import torch.nn as nn


class ResamplerProjector(nn.Module):
    def __init__(self, config, vision_model_config):
        super().__init__()
        self.hw = vision_model_config.image_size // vision_model_config.patch_size

        self.vision_downsample_ratio = 0.5
        proj_input_size = vision_model_config.hidden_size * int(1 / self.vision_downsample_ratio) ** 2

        self.pre_proj_layernorm = torch.nn.LayerNorm(proj_input_size)

        self.mlp = nn.Sequential(
            nn.Linear(proj_input_size, vision_model_config.hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(vision_model_config.hidden_size, config.hidden_size, bias=False),
        )
        self.mlp.apply(init_weights)
        self.pre_proj_layernorm.apply(init_weights)

    def forward(self, x, *args, **kwargs):
        x = x.reshape(x.shape[0], self.hw, self.hw, -1)
        x = pixel_shuffle(x, scale_factor=self.vision_downsample_ratio)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        x = self.pre_proj_layernorm(x)
        x = self.mlp(x)
        # print(torch.distributed.get_rank(), {name: [param, param.grad] for name, param in self.pre_proj_layernorm.named_parameters()})
        # print(torch.distributed.get_rank(), {name: [param, param.grad] for name, param in self.mlp.named_parameters()})
        return x

def pixel_shuffle(x, scale_factor=0.5):
    n, w, h, c = x.size()
    # N, W, H, C --> N, W, H * scale, C // scale
    x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
    # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
    x = x.permute(0, 2, 1, 3).contiguous()
    # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
    x = x.view(n, int(h * scale_factor), int(w * scale_factor),
               int(c / (scale_factor * scale_factor)))
    x = x.permute(0, 2, 1, 3).contiguous()
    return x

def pixel_shuffle_v2(x, scale_stride=2):
    n, w, h, c = x.size()
    assert w == h
    pl = (scale_stride - (h % scale_stride)) % scale_stride
    x = torch.nn.functional.pad(x, (0, 0, 0, pl, 0, pl), "constant", 0)
    h += pl
    w += pl

    x = x.reshape(n, w // scale_stride, scale_stride, h // scale_stride, scale_stride, c)
    x = x.permute(0, 1, 3, 2, 4, 5) 
    x = x.flatten(3)
    x = x.reshape(n, -1, scale_stride * scale_stride * c)
    return x

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

    if isinstance(m, nn.LayerNorm):
        torch.nn.init.ones_(m.weight)
        torch.nn.init.zeros_(m.bias)
