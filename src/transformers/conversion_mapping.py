# FILE to store the default conversion mapping that we use in `transformers`.
#
#
#
#
# Either we keep it here, or we move it to the config, but for newcomers, seeing this is kinda weird no?

from .core_model_loading import Concatenate, MergeModulelist, WeightConverter


_checkpoint_conversion_mapping = {
    "mixtral": [
        WeightConverter(
            source_keys=[
                "block_sparse_moe.*.w1.weight",
                "block_sparse_moe.*.w3.weight",
            ],  # you give me a list of 2 keys, I collect a list of tensors
            target_keys="experts.gate_up_proj",  # target key gets the list of two tensors
            operations=[
                MergeModulelist(
                    dim=0
                ),  # each process has two lists of tensors, we cat each list. -> we end up with 2 tensors
                Concatenate(dim=1),  # each process has 2 tensors, gate and up, we concat them into gate_up
            ],  # we want the loading to add this shard operation here. Though we can't shard after concats and merge, needs to be first
        ),
        # TODO: this one is flag dependant!
        WeightConverter(
            ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
            "self_attn.qkv_proj",
            Concatenate(dim=0),  # more like stack?
        ),
        WeightConverter("block_sparse_moe.*.w2.weight", "experts.down_proj.weight"),
    ]
}
