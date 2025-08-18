import json
import os
import re

from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file


IS_TIED = True
SAVE_DIR = "TODO"
SAVE_DIR = "/raid/anton/code/forks/transformers/tmp/ernie_vl"


def convert_file(state_dict):
    converted_state_dict = {}
    for key, tensor in state_dict.items():
        if "lm_head" in key and IS_TIED:
            pass
        elif "mlp" in key:
            if "moe_statics" in key:
                suffix = "moe_statics.e_score_correction_bias"
                converted_key = key.removesuffix(suffix)
                converted_state_dict[converted_key + "text_moe." + suffix] = tensor[0][None, :].contiguous()
                converted_state_dict[converted_key + "vision_moe." + suffix] = tensor[1][None, :].contiguous()
            elif "gate.weight" in key:
                moe_type = "text_moe"
                if "weight_1" in key:
                    moe_type = "vision_moe"
                suffix = "gate.weight"
                converted_key = key.removesuffix("_1")  # vision
                converted_key = converted_key.removesuffix("gate.weight")
                converted_state_dict[converted_key + f"{moe_type}." + suffix] = tensor.T.contiguous()
            elif ".experts" in key:
                moe_type = "text_moe"
                expert_number = int(re.findall(r'\d+', key)[-1])
                if expert_number >= 64:
                    moe_type = "vision_moe"
                    expert_number -= 64
                prefix = re.findall(r'model.layers.\d+.mlp.experts.', key)[0]
                converted_key = re.sub(r"\d+", f"{moe_type}.experts.{expert_number}", key.removeprefix(prefix))
                converted_state_dict[re.sub(".experts", "", prefix) + converted_key] = tensor.contiguous()
            else:
                converted_state_dict[key] = tensor.contiguous()
        else:
            converted_state_dict[key] = tensor.contiguous()
    return converted_state_dict


model_path = 'baidu/ERNIE-4.5-VL-28B-A3B-PT'
index_dict = {"weight_map": {}, "metadata": {"total_size": 0}}
checkpoint_path = snapshot_download(repo_id=model_path, allow_patterns=["*.safetensors*"])
for filename in sorted(os.listdir(checkpoint_path)):
    if filename == "model.safetensors.index.json":
        with open(os.path.join(checkpoint_path, filename), "r") as f:
            original_index = json.load(f)
        index_dict["metadata"] = original_index["metadata"]
    if filename.endswith('.safetensors'):
        input_file = os.path.join(checkpoint_path, filename)
        output_file = os.path.join(SAVE_DIR, filename)

        state_dict = load_file(input_file)
        converted_state_dict = convert_file(state_dict)
        save_file(converted_state_dict, output_file)

        for k, v in converted_state_dict.items():
            index_dict["weight_map"][k] = filename


index_file_path = os.path.join(SAVE_DIR, "model.safetensors.index.json")
with open(index_file_path, "w") as f:
    json.dump(index_dict, f)
