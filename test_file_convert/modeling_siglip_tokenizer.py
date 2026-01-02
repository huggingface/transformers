

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from torchvision import transforms

from PIL import Image
from einops import rearrange

from .modeling_vit import create_siglip_vit


def create_anyres_preprocess(
    short_size=384, 
    long_size=1152, 
    patch_size=16, 
    random_ratio=None, 
    min_short_size=128, 
    max_aspect_ratio=3., 
    filtering=True
):

    def resize_and_filtering(pil_image):
        pil_image = pil_image.convert('RGB')
        width, height = pil_image.size
        ss, ls = min(width, height), max(width, height)
        aspect_ratio = ls / ss
        if filtering and (ss < min_short_size or aspect_ratio > max_aspect_ratio):
            return None
        target_width, target_height = width, height
        if random_ratio is not None:
            log_ratio = torch.log(torch.tensor(random_ratio))
            sqrt_ratio = torch.exp(0.5 * torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()
            target_width = int(round(target_width * sqrt_ratio))
            target_height = int(round(target_height / sqrt_ratio))
        
        ss = min(target_width, target_height)
        if ss < short_size:
            target_width = target_width * (short_size / ss)
            target_height = target_height * (short_size / ss)
        
        ls = max(target_width, target_height)
        if ls > long_size:
            target_width = target_width * (long_size / ls)
            target_height = target_height * (long_size / ls)
        
        target_width = int(round(target_width / patch_size)) * patch_size
        target_height = int(round(target_height / patch_size)) * patch_size
        pil_image = pil_image.resize((target_width, target_height), resample=Image.BICUBIC)
        
        to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        return to_tensor(pil_image)

    transform = transforms.Lambda(resize_and_filtering)
    return transform


class IBQ(nn.Module):
    def __init__(self, n_e, e_dim, skip_quantization_prob=0.0, quantization_temp=2.0, beta=0.25, sane_index_shape=False, l2_norm=True):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.quantization_temp = quantization_temp
        self.skip_quantization_prob = skip_quantization_prob
        self.beta = beta
        self.sane_index_shape = sane_index_shape
        self.l2_norm = l2_norm

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        if self.l2_norm:
            self.embedding.weight.data = F.normalize(self.embedding.weight.data, p=2, dim=-1)
    
    def forward(self, z, temp=None, rescale_logits=False, return_logits=False, **kwargs):
        assert temp is None or temp == 1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits == False, "Only for interface compatible with Gumbel"
        assert return_logits == False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        assert z.shape[-1] == self.e_dim
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        if self.l2_norm:
            z = F.normalize(z, p=2, dim=-1)
            z_flattened = F.normalize(z_flattened, p=2, dim=-1)
            embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
        else:
            embedding = self.embedding.weight

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(embedding**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, torch.einsum('n d -> d n', embedding))
        
        if self.training:
            logits = -d / self.quantization_temp
            soft_one_hot = F.softmax(logits, dim=1)
            min_encoding_indices = soft_one_hot.max(1, keepdim=True)[1]
            hard_one_hot = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(1, min_encoding_indices, 1.0)
            one_hot = hard_one_hot - soft_one_hot.detach() + soft_one_hot

            z_q = einsum('b n, n d -> b d', one_hot, self.embedding.weight).view(z.shape)
            z_q_2 = einsum('b n, n d -> b d', hard_one_hot, self.embedding.weight).view(z.shape)

            # compute loss for embedding
            commit_loss = torch.mean((z_q - z) ** 2) + torch.mean((z_q_2.detach() - z) ** 2) + self.beta * \
                        torch.mean((z_q_2 - z.detach()) ** 2)
        else:
            min_encoding_indices = torch.argmin(d, dim=1)
            z_q = embedding[min_encoding_indices].view(z.shape)
            commit_loss = None
        
        if self.training and self.skip_quantization_prob > 0.0:
            z_q = torch.where(
                torch.rand_like(z_q[:, 0:1, 0:1, 0:1]).expand_as(z_q) <= self.skip_quantization_prob,
                z, z_q,
            )
        
        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return (z_q, None, min_encoding_indices), commit_loss

    def get_codebook_entry(self, indices, bhwc):
        # shape specifying (batch, height, width, channel)
        # get quantized latent vectors
        z_q = self.embedding(indices)

        if bhwc is not None:
            z_q = z_q.view(bhwc)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


class ResidualBlock(nn.Module):
    def __init__(self, channels, num_groups=32):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding='same')
        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        self.activate = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding='same')
        self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
    
    def forward(self, x):
        res = x
        x = self.norm1(x)
        x = self.activate(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.activate(x)
        x = self.conv2(x)
        return x + res


class VQConvProjector(nn.Module):
    def __init__(
        self, 
        z_channels=1536, 
        codebook_size=16384, 
        codebook_dim=2048, 
        conv_layers=2,
        with_norm=True,
        skip_quant_prob=0.1,
    ):
        super().__init__()
        self.quant_conv = nn.Conv2d(z_channels, codebook_dim, 1)
        self.quantize = IBQ(codebook_size, codebook_dim, skip_quant_prob, sane_index_shape=True)
        self.post_quant_conv = nn.Conv2d(codebook_dim, z_channels, 1)
        block = ResidualBlock
        self.post_conv = nn.Sequential(*[block(z_channels) for _ in range(conv_layers)])
    
    def forward(self, x, h, w):
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        z = self.quant_conv(x)
        (z_q, _, _), codebook_loss = self.quantize(z)
        z = self.post_quant_conv(z_q)
        z = self.post_conv(z)
        z = rearrange(z, 'b c h w -> b (h w) c')
        return z, codebook_loss
    
    def encode(self, x, h, w):
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        z = self.quant_conv(x)
        (_, _, tokens), _ = self.quantize(z)
        return tokens
    
    def decode(self, tokens, bhwc):
        z_q = self.quantize.get_codebook_entry(tokens, bhwc)
        z = self.post_quant_conv(z_q)
        z = self.post_conv(z)        
        return z


class SiglipTokenizer(nn.Module):
    def __init__(
        self, 
        siglip_name, 
        siglip_path, 
        projector_path, 
        z_channels=1536, 
        codebook_size=16384, 
        codebook_dim=2048, 
        with_norm=True
    ):
        super().__init__()
        self.vit = create_siglip_vit(model_name=siglip_name, path=siglip_path)
        self.vqproj = VQConvProjector(
            z_channels=z_channels, 
            codebook_size=codebook_size, 
            codebook_dim=codebook_dim, 
            with_norm=with_norm
        )
        self.vqproj.load_state_dict(torch.load(projector_path, map_location='cpu'), strict=True)

    def encode(self, x):
        features, (h, w), _ = self.vit(x)
        tokens = self.vqproj.encode(features, h, w)
        return tokens
    
    def decode(self, tokens, bhwc):
        return self.vqproj.decode(tokens, bhwc)
