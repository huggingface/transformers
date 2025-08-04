"""Convert DINOv3 checkpoints from the original repository.

URL: https://github.com/facebookresearch/dinov3/tree/main
"""

from .configuration_dinov3_vit import Dinov3Config


def convert_dinov3_to_hf(original_dinov3_state_dict, config: Dinov3Config):
    embed_dim = config.hidden_size
    hf_dinov3_state_dict = {}
    for key in original_dinov3_state_dict.keys():
        val = original_dinov3_state_dict[key]
        if key == "cls_token":
            key = "embeddings.cls_token"
        elif key == "mask_token":
            key = "embeddings.mask_token"
        elif key == "storage_tokens":
            key = "embeddings.register_tokens"
        elif key.startswith("patch_embed.proj"):
            key = key.replace("patch_embed.proj", "embeddings.patch_embeddings.proj")
        elif key.startswith("rope_embed"):
            key = key.replace("rope_embed", "rope_embeddings")
        elif key.startswith("blocks"):
            key = key.replace("blocks", "layer")
        if "ls1." in key:
            key = key.replace("ls1", "layer_scale1")
        if "ls2." in key:
            key = key.replace("ls2", "layer_scale2")
        if "attn." in key:
            key = key.replace("attn.", "attention.")
        if "qkv." in key:
            prefix, suffix = key.split("qkv")
            if "bias_mask" in suffix:
                continue
            elif "bias" in suffix:
                q_e, k_e, v_e = (
                    val[0:embed_dim],
                    val[embed_dim : embed_dim * 2],
                    val[embed_dim * 2 :],
                )
            else:
                q_e, k_e, v_e = (
                    val[0:embed_dim, :],
                    val[embed_dim : embed_dim * 2, :],
                    val[embed_dim * 2 :, :],
                )
            hf_dinov3_state_dict[prefix + "query" + suffix] = q_e
            if not ("bias" in suffix and config.mask_k_bias):
                hf_dinov3_state_dict[prefix + "key" + suffix] = k_e
            hf_dinov3_state_dict[prefix + "value" + suffix] = v_e
        else:
            hf_dinov3_state_dict[key] = val
    return hf_dinov3_state_dict
