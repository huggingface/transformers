# FILE to store the default conversion mapping that we use in `transformers`.
#
#
#
#
# Either we keep it here, or we move it to the config, but for newcomers, seeing this is kinda weird no?

from .core_model_loading import Concatenate, WeightConversion


_checkpoint_conversion_mapping = {
    "mixtral": [
        WeightConversion(
            source_keys=["experts.*.w1.weight", "experts.*.w3.weight"],
            target_keys="experts.gate_up_proj.weight",
            operations=[Concatenate(0), Concatenate(0)],
        ),
        WeightConversion("self_attn.(q|k|v)_proj", "self_attn.qkv_proj", Concatenate),
        WeightConversion("experts.*.w2.weight", "experts.down_proj.weight", Concatenate),
        WeightConversion("mlp.w2.weight", "mlp.down_proj.weight"),
    ]
}
