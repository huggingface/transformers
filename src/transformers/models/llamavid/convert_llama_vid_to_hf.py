# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse

import torch
from huggingface_hub import hf_hub_download
import os
import json
from transformers.models.llamavid.configuration_llamavid import LLaMAVIDLlavaConfig ,LLaMAVIDLlavaQFormerConfig, LLaMAVIDLlavaVisionConfig

from transformers import (
    AddedToken,
    AutoConfig,
    AutoTokenizer,
    CLIPImageProcessor,
    LLaMAVIDLlavaForConditionalGeneration,
    LLaMAVIDLlavaProcessor,
    BertTokenizer,
)


EPILOG_TXT = """Example:
    python transformers/src/transformers/models/llamavid/convert_llama_vid_hf.py --text_model_id lmsys/vicuna-7b-v1.5 --vision_model_id openai/clip-vit-large-patch14-336 --output_hub_path org/llava-v1.5-7b-conv --old_state_dict_id liuhaotian/llava-v1.5-7b

Example for creating the old state dict file with Python:

    import torch
    from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM

    # load model
    kwargs = {"device_map": "auto", "torch_dtype": torch.float16}
    model = LlavaLlamaForCausalLM.from_pretrained("liuhaotian/llava-v1.5-7b", low_cpu_mem_usage=True, **kwargs)

    # load vision tower
    model.get_vision_tower().load_model()

    # Save state dict
    torch.save(model.state_dict(), "tmp/hf_models/llava-v1.5-7b/model_state_dict.bin")
"""

KEYS_TO_MODIFY_MAPPING = {
    
    "model.": "language_model.model.",
    "lm_head": "language_model.lm_head",
    "model.vision_tower.vision_tower":  "vision_tower."
}


def convert_state_dict_to_hf(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.endswith(".inv_freq"):
            continue
        if key.startswith("model.layers"):
            key =  'language_model.' + key
        if key.startswith("model.vision_tower."):
            key= key.replace("model.vision_tower." ,"")
        if key.startswith("model.vlm_att_encoder"):
            key= key.replace("model.vlm_att_encoder.","vlm_att_encoder." )
        if key.startswith("model.vlm_att_"):
            key= key.replace("model.vlm_att_", "vlm_att_")
        if key.startswith("lm_head"):
            key= key.replace("lm_head","language_model.lm_head")
        if key.startswith("model.embed_tokens"):
            key = key.replace("model.embed_tokens", "language_model.model.embed_tokens")
        if key.startswith("model.mm_projector.0"):
            key=key.replace("model.mm_projector.0" , "multi_modal_projector.linear_1")
        if key.startswith("model.mm_projector.2"):
            key=key.replace("model.mm_projector.2" , "multi_modal_projector.linear_2")
        if key.startswith("model.norm.weight"):
            key =key.replace("model.norm.weight","language_model.model.norm.weight")
        if key.startswith("vlm_att_query"):
            key = key.replace("vlm_att_query" , "query_tokens")
        if key.startswith("vlm_att_encoder.bert"):
            key = key.replace("vlm_att_encoder.bert" , "qformer")
            if "ccrossattention.self" in key :
                key= key.replace("crossattention.self", "crossattention.attention")
            if "attention.self" in key:
                key = key.replace("attention.self", "attention.attention")


        new_state_dict[key] = value
    return new_state_dict


# here we list all keys to be renamed (original name on the left, our name on the right)
def create_rename_keys(config):
    rename_keys = []
    # fmt: off

    # vision encoder

   
    rename_keys.append(("vision_tower.cls_token", "vision_tower.embeddings.class_embedding"))
    rename_keys.append(("vision_tower.pos_embed", "vision_tower.embeddings.position_embedding"))
    rename_keys.append(("vision_tower.patch_embed.proj.weight", "vision_tower.embeddings.patch_embedding.weight"))
    rename_keys.append(("vision_tower.patch_embed.proj.bias", "vision_tower.embeddings.patch_embedding.bias"))
    rename_keys.append(("vlm_att_ln.weight", "vision_tower.post_layernorm.weight"))
    rename_keys.append(("vlm_att_ln.bias", "vision_tower.post_layernorm.bias"))


    for i in range(config.vision_tower_config.num_hidden_layers):
        rename_keys.append((f"vision_tower.blocks.{i}.norm1.weight", f"vision_tower.encoder.layers.{i}.layer_norm1.weight"))
        rename_keys.append((f"vision_tower.blocks.{i}.norm1.bias", f"vision_tower.encoder.layers.{i}.layer_norm1.bias"))
        rename_keys.append((f"vision_tower.blocks.{i}.norm2.weight", f"vision_tower.encoder.layers.{i}.layer_norm2.weight"))
        rename_keys.append((f"vision_tower.blocks.{i}.norm2.bias", f"vision_tower.encoder.layers.{i}.layer_norm2.bias"))
        rename_keys.append((f"vision_tower.blocks.{i}.attn.qkv.weight", f"vision_tower.encoder.layers.{i}.self_attn.qkv.weight"))
        rename_keys.append((f"vision_tower.blocks.{i}.attn.proj.weight", f"vision_tower.encoder.layers.{i}.self_attn.projection.weight",))
        rename_keys.append((f"vision_tower.blocks.{i}.attn.proj.bias", f"vision_tower.encoder.layers.{i}.self_attn.projection.bias"))
        rename_keys.append((f"vision_tower.blocks.{i}.mlp.fc1.weight", f"vision_tower.encoder.layers.{i}.mlp.fc1.weight"))
        rename_keys.append((f"vision_tower.blocks.{i}.mlp.fc1.bias", f"vision_tower.encoder.layers.{i}.mlp.fc1.bias"))
        rename_keys.append((f"vision_tower.blocks.{i}.mlp.fc2.weight", f"vision_tower.encoder.layers.{i}.mlp.fc2.weight"))
        rename_keys.append((f"vision_tower.blocks.{i}.mlp.fc2.bias", f"vision_tower.encoder.layers.{i}.mlp.fc2.bias"))
        # QFormer
    rename_keys.append(("qformer.embeddings.LayerNorm.weight", "qformer.embeddings.layernorm.weight"))
    rename_keys.append(("qformer.embeddings.LayerNorm.bias", "qformer.embeddings.layernorm.bias"))

    

    # fmt: on
    return rename_keys

def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val



def convert_to_hf(state_dict, pretext ):
    new_state_dict ={}
    for k , v in state_dict.items():
        k = pretext + k 
        new_state_dict[k] =  v 
    return new_state_dict


def read_in_q_v_bias(state_dict, config):
    for i in range(config.vision_tower_config.num_hidden_layers):
        # read in original q and v biases
        q_bias = state_dict.pop(f"vision_tower.blocks.{i}.attn.q_bias")
        v_bias = state_dict.pop(f"vision_tower.blocks.{i}.attn.v_bias")

        # next, set bias in the state dict
        qkv_bias = torch.cat((q_bias, torch.zeros_like(v_bias, requires_grad=False), v_bias))
        state_dict[f"vision_tower.encoder.layers.{i}.self_attn.qkv.bias"] = qkv_bias

def convert_llamavid_to_hf(text_model_id, vision_model_id, output_hub_path, old_state_dict_id):
    
    text_model_id = 'C:\\Users\\niles\\OneDrive\\Desktop\\Data\\code\\LLaMA-VID\\work_dirs\\llama-vid-7b-full-336'
    vision_model_id = 'C:\\Users\\niles\\OneDrive\\Desktop\\Data\\code\\LLaMA-VID\\llamavid\\processor\\clip-patch14-336'
    output_hub_path ='Nilesh360/llama-vid-7b-full-336'
    old_state_dict_id = 'C:\\Users\\niles\\OneDrive\\Desktop\\Data\\code\\LLaMA-VID\\work_dirs\\llama-vid-7b-full-336'
    #old_state_dict_id = 'C:\\Users\\niles\\OneDrive\\Desktop\\Data\code\\LLaMA-VID\\model_zoo\\LLM\\vicuna\\7b-V1.5'
    torch.set_default_dtype(torch.float16)
    text_config = AutoConfig.from_pretrained(text_model_id)

    qformer_tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased", truncation_side="left")
    qformer_tokenizer.add_special_tokens({"bos_token": "[DEC]"})


    tokenizer = AutoTokenizer.from_pretrained(text_model_id)
    tokenizer.add_tokens(AddedToken("<image>", special=True, normalized=False), special_tokens=True)
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    
    image_processor = CLIPImageProcessor.from_pretrained(vision_model_id)

    processor = LLaMAVIDLlavaProcessor(tokenizer=tokenizer, image_processor=image_processor, qformer_tokenizer = qformer_tokenizer )
    processor.save_pretrained("llama_vid_model")
    vision_tower_config = LLaMAVIDLlavaVisionConfig(image_size=336, qkv_bias = True).to_dict()
    qformer_config =  LLaMAVIDLlavaQFormerConfig(vocab_size=30523).to_dict()
    
    config = LLaMAVIDLlavaConfig(text_config=text_config,vision_tower_config=vision_tower_config, qformer_config=qformer_config   )
    config.pad_token_id = 32001
    config.save_pretrained("llama_vid_model")

    with torch.device("meta"):
        model = LLaMAVIDLlavaForConditionalGeneration(config)

    # Pad to 64 for performance reasons
    pad_shape = 64

    total_state_dict = {}
    '''  '''
    model_path =  old_state_dict_id + '\\' +  'pytorch_model-0000{}-of-00002.bin'

    for shard in range(1 , 3):
        old_dict= model_path.format(shard , shard)
        state_dict = torch.load(old_dict)
        state_dict = convert_state_dict_to_hf(state_dict)
        total_state_dict.update(state_dict)


    rename_keys = create_rename_keys(config)
    for src, dest in rename_keys:
        rename_key(total_state_dict, src, dest)

    del total_state_dict['qformer.embeddings.position_ids']
    
    param_names = sorted([name for name, _ in model.named_parameters() ] )
    with open('model_keys.txt', 'w') as file:
        for key in param_names:
            file.write("%s\n" % key)

    read_in_q_v_bias(total_state_dict , config)
    param_names_ = sorted([name_ for name_, _ in total_state_dict.items() ] )
    with open('state_dict.txt', 'w') as file:
        for key in param_names_:
            file.write("%s\n" % key)


    model.load_state_dict(total_state_dict, strict=True, assign=True)
 
    pre_expansion_embeddings = model.language_model.model.embed_tokens.weight.data
    mu = torch.mean(pre_expansion_embeddings, dim=0).float()
    n = pre_expansion_embeddings.size()[0]
    sigma = ((pre_expansion_embeddings - mu).T @ (pre_expansion_embeddings - mu)) / n
    dist = torch.distributions.multivariate_normal.MultivariateNormal(mu, covariance_matrix=1e-5 * sigma)

    # We add an image token so we resize the model
    model.resize_token_embeddings(config.text_config.vocab_size + 2, pad_shape)
    model.language_model.model.embed_tokens.weight.data[32000:] = torch.stack(
        tuple((dist.sample() for _ in range(model.language_model.model.embed_tokens.weight.data[32000:].shape[0]))),
        dim=0,
    )
    model.language_model.lm_head.weight.data[32000:] = torch.stack(
        tuple((dist.sample() for _ in range(model.language_model.lm_head.weight.data[32000:].shape[0]))),
        dim=0,
    )
 
    model.save_pretrained("llama_vid_model")
    processor.save_pretrained("llama_vid_model")

    #model.push_to_hub("output_hub_path_vid")
    #processor.push_to_hub("output_hub_path_vid")

    #model.push_to_hub(output_hub_path)
    #processor.push_to_hub(output_hub_path)
    #model.save_pretrained('llama_vid_models.pth')
    #torch.save_(model , "llama_vid_model"

    #torch.save(processor , "processor")
    #model.load_state_dict(torch.load('model.pth'))
    #processor._save_to_state_dict("process.pth")
    print('pushed to hub successfully')



def main():
    
    convert_llamavid_to_hf(text_model_id ='', vision_model_id='', output_hub_path='', old_state_dict_id='')


if __name__ == "__main__":
    main()
