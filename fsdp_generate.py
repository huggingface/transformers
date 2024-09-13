import functools

import torch
import torch.distributed
import torch.distributed.fsdp
import torch.distributed.fsdp.wrap
import transformers
import transformers.models.gpt_neo.modeling_gpt_neo


def main() -> None:
    torch.distributed.init_process_group(world_size=2)
    device = torch.device(torch.distributed.get_rank())
    torch.cuda.set_device(device)

    pretrained_model_name_or_path = "EleutherAI/gpt-neo-125m"
    model = transformers.AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        device_map=device,
        attn_implementation="flash_attention_2",  # I'm using flash-attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
        torch_dtype=torch.bfloat16,
    )
    assert isinstance(model, transformers.models.gpt_neo.modeling_gpt_neo.GPTNeoForCausalLM)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    fsdp_model = torch.distributed.fsdp.FullyShardedDataParallel(
        model,
        auto_wrap_policy=functools.partial(
            torch.distributed.fsdp.wrap.transformer_auto_wrap_policy,
            transformer_layer_cls={
                transformers.models.gpt_neo.modeling_gpt_neo.GPTNeoBlock
            },
        ),
        limit_all_gathers=True,
        use_orig_params=True,  # required to overcome the error "The tensor has a non-zero number of elements, but its data is not allocated yet" ... PreTrainedModel.generate is probably using some torch.compile-wrapped function
    )

    data_by_rank = {  # differently sized causes FSDP to hang
        0: "Hello world!",
        1: "The quick brown fox jumps over the lazy dog."
    }

    batch = tokenizer(
        data_by_rank[torch.distributed.get_rank()],
        return_tensors="pt",
        return_attention_mask=True,
    ).to(device)

    with torch.distributed.fsdp.FullyShardedDataParallel.summon_full_params(fsdp_model):  # required to overcome to the error "'weight' must be 2-D"
        generated_text = fsdp_model.module.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_length=20,
            # synced_gpus=True,  # True is required to use differently sized data with FSDP + generate (current default is False)
        )

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

# TODO: # "sdpa" fails with an off by one error?
# [rank1]: Traceback (most recent call last):
# [rank1]:   File "/home/matthew/transformers/fsdp_generate.py", line 64, in <module>
# [rank1]:     main()
# [rank1]:   File "/home/matthew/transformers/fsdp_generate.py", line 52, in main
# [rank1]:     generated_text = fsdp_model.module.generate(
# [rank1]:   File "/home/matthew/.conda/envs/transformers310/lib/python3.10/site-packages/torch/utils/_contextlib.py", line 116, in decorate_context
# [rank1]:     return func(*args, **kwargs)
# [rank1]:   File "/home/matthew/transformers/src/transformers/generation/utils.py", line 2048, in generate
# [rank1]:     result = self._sample(
# [rank1]:   File "/home/matthew/transformers/src/transformers/generation/utils.py", line 3001, in _sample
# [rank1]:     outputs = self(**model_inputs, return_dict=True)
# [rank1]:   File "/home/matthew/.conda/envs/transformers310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
# [rank1]:     return self._call_impl(*args, **kwargs)
# [rank1]:   File "/home/matthew/.conda/envs/transformers310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
# [rank1]:     return forward_call(*args, **kwargs)
# [rank1]:   File "/home/matthew/transformers/src/transformers/models/gpt_neo/modeling_gpt_neo.py", line 1038, in forward
# [rank1]:     transformer_outputs = self.transformer(
# [rank1]:   File "/home/matthew/.conda/envs/transformers310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
# [rank1]:     return self._call_impl(*args, **kwargs)
# [rank1]:   File "/home/matthew/.conda/envs/transformers310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
# [rank1]:     return forward_call(*args, **kwargs)
# [rank1]:   File "/home/matthew/transformers/src/transformers/models/gpt_neo/modeling_gpt_neo.py", line 801, in forward
# [rank1]:     outputs = block(
# [rank1]:   File "/home/matthew/.conda/envs/transformers310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
# [rank1]:     return self._call_impl(*args, **kwargs)
# [rank1]:   File "/home/matthew/.conda/envs/transformers310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
# [rank1]:     return forward_call(*args, **kwargs)
# [rank1]:   File "/home/matthew/.conda/envs/transformers310/lib/python3.10/site-packages/torch/distributed/fsdp/fully_sharded_data_parallel.py", line 863, in forward
# [rank1]:     output = self._fsdp_wrapped_module(*args, **kwargs)
# [rank1]:   File "/home/matthew/.conda/envs/transformers310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
# [rank1]:     return self._call_impl(*args, **kwargs)
# [rank1]:   File "/home/matthew/.conda/envs/transformers310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
# [rank1]:     return forward_call(*args, **kwargs)
# [rank1]:   File "/home/matthew/transformers/src/transformers/models/gpt_neo/modeling_gpt_neo.py", line 512, in forward
# [rank1]:     attn_outputs = self.attn(
# [rank1]:   File "/home/matthew/.conda/envs/transformers310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
# [rank1]:     return self._call_impl(*args, **kwargs)
# [rank1]:   File "/home/matthew/.conda/envs/transformers310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
# [rank1]:     return forward_call(*args, **kwargs)
# [rank1]:   File "/home/matthew/transformers/src/transformers/models/gpt_neo/modeling_gpt_neo.py", line 462, in forward
# [rank1]:     return self.attention(
# [rank1]:   File "/home/matthew/.conda/envs/transformers310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
# [rank1]:     return self._call_impl(*args, **kwargs)
# [rank1]:   File "/home/matthew/.conda/envs/transformers310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
# [rank1]:     return forward_call(*args, **kwargs)
# [rank1]:   File "/home/matthew/transformers/src/transformers/models/gpt_neo/modeling_gpt_neo.py", line 314, in forward
# [rank1]:     attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
# [rank1]:   File "/home/matthew/transformers/src/transformers/models/gpt_neo/modeling_gpt_neo.py", line 278, in _attn
# [rank1]:     attn_weights = attn_weights + causal_mask
# [rank1]: RuntimeError: The size of tensor a (21) must match the size of tensor b (20) at non-singleton dimension 3