import os
from glob import glob


def check_flash_support_list():
    with open("./docs/source/en/perf_infer_gpu_one.md", "r") as f:
        doctext = f.read()

        doctext = doctext.split("FlashAttention-2 is currently supported for the following architectures:")[1]
        doctext = doctext.split("You can request to add FlashAttention-2 support")[0]

    patterns = glob("./src/transformers/models/**/modeling_*.py")
    patterns_tf = glob("./src/transformers/models/**/modeling_tf_*.py")
    patterns_flax = glob("./src/transformers/models/**/modeling_flax_*.py")
    patterns = list(set(patterns) - set(patterns_tf) - set(patterns_flax))
    archs_supporting_fa2 = []
    for filename in patterns:
        with open(filename, "r") as f:
            text = f.read()

            if "_supports_flash_attn_2 = True" in text:
                model_name = os.path.basename(filename).replace(".py", "").replace("modeling_", "")
                archs_supporting_fa2.append(model_name)

    for arch in archs_supporting_fa2:
        if arch not in doctext:
            raise ValueError(
                f"{arch} should be in listed in the flash attention documentation but is not. Please update the documentation."
            )


def check_sdpa_support_list():
    with open("./docs/source/en/perf_infer_gpu_one.md", "r") as f:
        doctext = f.read()

        doctext = doctext.split(
            "For now, Transformers supports SDPA inference and training for the following architectures:"
        )[1]
        doctext = doctext.split("Note that FlashAttention can only be used for models using the")[0]

    patterns = glob("./src/transformers/models/**/modeling_*.py")
    patterns_tf = glob("./src/transformers/models/**/modeling_tf_*.py")
    patterns_flax = glob("./src/transformers/models/**/modeling_flax_*.py")
    patterns = list(set(patterns) - set(patterns_tf) - set(patterns_flax))
    archs_supporting_sdpa = []
    for filename in patterns:
        with open(filename, "r") as f:
            text = f.read()

            if "_supports_sdpa = True" in text:
                model_name = os.path.basename(filename).replace(".py", "").replace("modeling_", "")
                archs_supporting_sdpa.append(model_name)

    for arch in archs_supporting_sdpa:
        if arch not in doctext:
            raise ValueError(
                f"{arch} should be in listed in the SDPA documentation but is not. Please update the documentation."
            )


if __name__ == "__main__":
    check_flash_support_list()
    check_sdpa_support_list()
