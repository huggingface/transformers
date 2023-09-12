from transformers.models.crossformer.configuration_crossformer import CrossformerConfig
from transformers.models.crossformer.modeling_crossformer import CrossFormer


config = CrossformerConfig(
    drop_path_rate=0.1,
    embed_dim=64,
    depths=[1, 1, 8, 6],
    num_heads=[2, 4, 8, 16],
    group_size=[7, 7, 7, 7],
    patch_size=[4, 8, 16, 32],
    merge_size=[[2, 4], [2, 4], [2, 4]],
)
model = CrossFormer(config)

print(model)
