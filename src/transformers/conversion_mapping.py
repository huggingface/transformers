# FILE to store the default conversion mapping that we use in `transformers`.
#
#
#
#
# Either we keep it here, or we move it to the config, but for newcomers, seeing this is kinda weird no?

from .core_model_loading import Concatenate, MergeModuleList, WeightConversion, Fp8Quantize, Shard


_checkpoint_conversion_mapping = {
    "mixtral": [
        WeightConversion(
            source_keys=["experts.*.w1.weight", "experts.*.w3.weight"],
            target_keys="experts.gate_up_proj.weight",
            operations=[
                Shard(
                    0
                ),  # we have a 2 lists, so shard 0 -> slice each list, shard 1 -> slice the tensors in the lists
                MergeModuleList,  # each process has two lists of tensors, we cat each list.
                Concatenate(0),  # each process has 2 tensors, gate and up, we concat them into gate_up
                Fp8Quantize,  # we can imagine quantizing at this point
            ],  # we want the loading to add this shard operation here. Though we can't shard after concats and merge, needs to be first
        ),
        WeightConversion(
            ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"], "self_attn.qkv_proj", Concatenate
        ),
        WeightConversion("self_attn.out_proj.weight", operations=Shard(1)),  # If a user wants to force shard?
        WeightConversion("experts.*.w2.weight", "experts.down_proj.weight", Concatenate),
        WeightConversion("mlp.w2.weight", "mlp.down_proj.weight"),
        # 8-bit quantization of certain weights (just for testing!)
        WeightConversion(
            "experts.gate_up_proj.weight", ["experts.gate_up_proj.weight", "experts.gate_up_proj.scale"], Fp8Quantize
        ),
    ]
}
