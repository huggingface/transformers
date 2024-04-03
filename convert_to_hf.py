import gc
import json
import os
import shutil

import numpy as np
import torch

from transformers.models.cohere.configuration_cohere import CohereConfig
from transformers.models.cohere.modeling_cohere import CohereForCausalLM
from transformers import AutoTokenizer

def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f)


def convert_np_to_torch(x: np.ndarray):
    return torch.from_numpy(x).contiguous()


def write_model(
    model_path, checkpoint_path, longcontext=True, dtype=torch.float16
):
    os.makedirs(model_path, exist_ok=True)
    tmp_model_path = os.path.join(model_path, "tmp")
    os.makedirs(tmp_model_path, exist_ok=True)

    # download tokenizer.json
    # execute via os.popen gsutil -m cp gs://cohere-prod/encoders/releases/{__version__}/m255k_bos_eos_special_tokens_with_eot_template-ns.json model_path
    os.popen(f"gsutil -m cp gs://cohere-prod/encoders/releases/0.4.4/m255k_bos_eos_special_tokens_with_eot_template-ns.json {model_path}/tokenizer.json")

    # tokenizer_config
    tokenizer_config = {
    "add_bos_token": True,
    "add_eos_token": False,
    "bos_token": "<BOS_TOKEN>",
    "clean_up_tokenization_spaces": False,
    "eos_token": "<|END_OF_TURN_TOKEN|>",
    "legacy": True,
    "model_max_length": 1000000000000000019884624838656,
    "pad_token": "<PAD>",
    "sp_model_kwargs": {},
    "spaces_between_special_tokens": False,
    "tokenizer_class": "CohereTokenizerFast",
    "unk_token": None,
    "use_default_system_prompt": False
    }

    write_json(tokenizer_config, os.path.join(model_path, "tokenizer_config.json"))


    params = read_json(os.path.join(checkpoint_path, "unsharded_weights_metadata.json"))['architecture']

    n_layers = params["n_layers"]
    n_heads = params["n_heads"]
    dim = params["embedding_dim"]
    dims_per_head = dim // n_heads
    base = params.get("rope_base_frequency", 10000.0) if not longcontext else 8e6
    inv_freq = 1.0 / (base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head))
    max_position_embeddings = params["max_positional_embedding_length"]
    ffn_dim = params["ffn_dim"]//2

    vocab_size = 256000

    n_kv_heads = params["n_kv_heads"]

    is_gqa = (n_heads != n_kv_heads)
    use_qk_norm = params.get("use_qk_norm", False)

    print("This is a GQA model." if is_gqa else "This is a MHA model.")

    print(f"Fetching all parameters from the checkpoint at {checkpoint_path}.")
    # Load weights
    param_count = 0
    index_dict = {"weight_map": {}}

    # The biases are 0 anyways so we don't really care about them
    for idx in range(n_layers):
        state_dict = {}
        filename = f"pytorch_model-{idx + 1}-of-{n_layers + 1}.bin"

        state_dict[f'model.layers.{idx}.input_layernorm.bias'] = convert_np_to_torch(np.load(
            f'{checkpoint_path}/parallel_transformer_block-{idx}-layernorm_1-offset.npy'))
        state_dict[f'model.layers.{idx}.input_layernorm.weight'] = convert_np_to_torch(np.load(
            f'{checkpoint_path}/parallel_transformer_block-{idx}-layernorm_1-scale.npy'))

        if use_qk_norm:
            state_dict[f'model.layers.{idx}.self_attn.q_norm.weight'] = convert_np_to_torch(np.load(
            f'{checkpoint_path}/parallel_transformer_block-{idx}-q_norm-scale.npy')).squeeze()
            state_dict[f'model.layers.{idx}.self_attn.k_norm.weight'] = convert_np_to_torch(np.load(
            f'{checkpoint_path}/parallel_transformer_block-{idx}-k_norm-scale.npy')).squeeze()

        if is_gqa:
            q_w = np.load(f'{checkpoint_path}/parallel_transformer_block-{idx}-q_proj-w.npy')
            kv_w = np.load(f'{checkpoint_path}/parallel_transformer_block-{idx}-kv_proj-w.npy')
            state_dict[f'model.layers.{idx}.self_attn.q_proj.weight'] = convert_np_to_torch(q_w.transpose())
            state_dict[f'model.layers.{idx}.self_attn.k_proj.weight'] = convert_np_to_torch(np.split(kv_w, 2, axis=-1)[0].transpose())
            state_dict[f'model.layers.{idx}.self_attn.v_proj.weight'] = convert_np_to_torch(np.split(kv_w, 2, axis=-1)[1].transpose())
        else:
            qkv_w = np.load(f'{checkpoint_path}/parallel_transformer_block-{idx}-c_attn-w.npy')
            qkv_b = np.load(f'{checkpoint_path}/parallel_transformer_block-{idx}-c_attn-b.npy')
            state_dict[f'model.layers.{idx}.self_attn.q_proj.weight'] = convert_np_to_torch(np.split(qkv_w, 3, axis=-1)[0].transpose())
            state_dict[f'model.layers.{idx}.self_attn.q_proj.bias'] = convert_np_to_torch(np.split(qkv_b, 3, axis=-1)[0])
            state_dict[f'model.layers.{idx}.self_attn.k_proj.weight'] = convert_np_to_torch(np.split(qkv_w, 3, axis=-1)[1].transpose())
            state_dict[f'model.layers.{idx}.self_attn.k_proj.bias'] = convert_np_to_torch(np.split(qkv_b, 3, axis=-1)[1])
            state_dict[f'model.layers.{idx}.self_attn.v_proj.weight'] = convert_np_to_torch(np.split(qkv_w, 3, axis=-1)[2].transpose())
            state_dict[f'model.layers.{idx}.self_attn.v_proj.bias'] = convert_np_to_torch(np.split(qkv_b, 3, axis=-1)[2])

        state_dict[f'model.layers.{idx}.self_attn.o_proj.weight'] = convert_np_to_torch(np.load(
            f'{checkpoint_path}/parallel_transformer_block-{idx}-c_proj-w.npy').transpose())
        state_dict[f'model.layers.{idx}.self_attn.o_proj.bias'] = convert_np_to_torch(np.load(
            f'{checkpoint_path}/parallel_transformer_block-{idx}-c_proj-b.npy'))

        w = np.load(
            f'{checkpoint_path}/parallel_transformer_block-{idx}-ffn_1-w.npy').transpose()
        b = np.load(
            f'{checkpoint_path}/parallel_transformer_block-{idx}-ffn_1-b.npy')

        state_dict[f'model.layers.{idx}.mlp.up_proj.weight'] = convert_np_to_torch(np.split(w, 2, axis=0)[0])
        state_dict[f'model.layers.{idx}.mlp.up_proj.bias'] = convert_np_to_torch(np.split(b, 2, axis=0)[0])
        state_dict[f'model.layers.{idx}.mlp.gate_proj.weight'] = convert_np_to_torch(np.split(w, 2, axis=0)[1])
        state_dict[f'model.layers.{idx}.mlp.gate_proj.bias'] = convert_np_to_torch(np.split(b, 2, axis=0)[1])

        state_dict[f'model.layers.{idx}.mlp.down_proj.weight'] = convert_np_to_torch(np.load(
            f'{checkpoint_path}/parallel_transformer_block-{idx}-ffn_2-w.npy').transpose())
        state_dict[f'model.layers.{idx}.mlp.down_proj.bias'] = convert_np_to_torch(np.load(
            f'{checkpoint_path}/parallel_transformer_block-{idx}-ffn_2-b.npy'))

        state_dict[f"model.layers.{idx}.self_attn.rotary_emb.inv_freq"] = inv_freq

        for k, v in state_dict.items():
            index_dict["weight_map"][k] = filename
            param_count += v.numel()
        torch.save(state_dict, os.path.join(tmp_model_path, filename))
        print(f"Saved layer {idx + 1} of {n_layers + 1}.")


    filename = f"pytorch_model-{n_layers + 1}-of-{n_layers + 1}.bin"
    state_dict = {
        "model.embed_tokens.weight": convert_np_to_torch(np.load(
            f'{checkpoint_path}/input_embedding_layer-linear-w.npy')),
        "model.norm.weight": convert_np_to_torch(np.load(f'{checkpoint_path}/layer_norm-layer_norm-scale.npy')),
        "model.norm.bias": convert_np_to_torch(np.load(f'{checkpoint_path}/layer_norm-layer_norm-offset.npy')),
    }

    for k, v in state_dict.items():
        index_dict["weight_map"][k] = filename
        param_count += v.numel()
    torch.save(state_dict, os.path.join(tmp_model_path, filename))

    # Write configs
    index_dict["metadata"] = {"total_size": param_count * 2}
    write_json(index_dict, os.path.join(tmp_model_path, "pytorch_model.bin.index.json"))

    config = CohereConfig(
        hidden_size=dim,
        intermediate_size=ffn_dim,
        num_attention_heads=n_heads,
        num_hidden_layers=n_layers,
        layer_norm_eps=1e-5,
        num_key_value_heads=n_kv_heads,
        vocab_size=vocab_size,
        rope_theta=base,
        max_position_embeddings=max_position_embeddings,
        pad_token_id=0,
        bos_token_id=5,
        eos_token_id=255001,
        logit_scale=params['mup']['logit_scale'] * (params['mup']['base_embedding_dim'] / params['embedding_dim']),
        tie_word_embeddings=True,
        use_qk_norm=use_qk_norm
    )
    config.save_pretrained(tmp_model_path)

    # Make space so we can load the model properly now.
    del state_dict
    gc.collect()

    print("Loading the checkpoint in a Cohere model.")
    model = CohereForCausalLM.from_pretrained(tmp_model_path, torch_dtype=dtype, low_cpu_mem_usage=False)
    # Avoid saving this as part of the config.
    del model.config._name_or_path
    model.config.torch_dtype = dtype
    print("Saving in the Transformers format.")
    model.save_pretrained(model_path, safe_serialization=True)
    shutil.rmtree(tmp_model_path)

def test_model(model_path):
    # tokenizer = TemplatedBPTokenizer(BOS_ENDOFTURN_EOS_255k_PATH, "chat-command-turn_tokens-v2", chat_preamble="")
    tokenizer = AutoTokenizer.from_pretrained("CohereForAI/c4ai-command-r-v01")

    model = CohereForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to("cuda")
    # prompt = (
    # "Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. How many sheep do Toulouse, Charleston, and Seattle have together if Seattle has 20 sheep?"
    # )
    prompt = ("Why is docker used?")
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    # input_ids = torch.tensor([tokenizer.encode_turns([{"role": "User", "message": prompt}])]).to("cuda")
    attention_mask = torch.ones_like(input_ids)
    # print(input_ids)

    gen_tokens = model.generate(
    input_ids,
    attention_mask=attention_mask,
    do_sample=False,
    temperature=0.3,
    max_length=50,
    )
    gen_text = tokenizer.decode(gen_tokens.squeeze().detach().cpu().numpy())
    print(gen_text)



def main():
    # write_model('./7B_gqa', '../7b_gqa/', False, torch.float16)
    test_model('./7B_gqa')
    # write_model('./HF_Final_weight_tie', '../commandR', None, True, True, torch.float16)
    # write_model('./Aya_7B', 'aya_7B_9zoxhfci/unsharded_weights', None, True, False, torch.float16)
    # test_model('./HF_Final_weight_tie')
    print('done')


if __name__=='__main__':
    main()
