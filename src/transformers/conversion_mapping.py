# FILE to store the default conversion mapping that we use in `transformers`.
#
#
#
#
# Either we keep it here, or we move it to the config, but for newcomers, seeing this is kinda weird no?

from .core_model_loading import ConversionType, Fuse, MergeModuleList, WeightConversion

_checkpoint_conversion_mapping = { "mixtral": {
    "experts.*.(w1|w3).weight$": WeightConversion(
        "experts.gate_up_proj.weight", [ConversionType.MERGE_MODULE_LIST, ConversionType.FUSE]
    ),
    "self_attn.(q|k|v)_proj": WeightConversion("self_attn.qkv_proj", ConversionType.FUSE),
    "experts*.w2.weight": WeightConversion("experts.down_proj.weight", ConversionType.MERGE_MODULE_LIST),
}}
