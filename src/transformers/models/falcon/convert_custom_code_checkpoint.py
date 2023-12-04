import json
from argparse import ArgumentParser
from pathlib import Path


"""
This script converts Falcon custom code checkpoints to modern Falcon checkpoints that use code in the Transformers
library. After conversion, performance (especially for generation) should improve and the checkpoint can be loaded
without needing trust_remote_code=True.
"""

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        required=True,
        help="Directory containing a custom code checkpoint to convert to a modern Falcon checkpoint.",
    )
    args = parser.parse_args()

    if not args.checkpoint_dir.is_dir():
        raise ValueError("--checkpoint_dir argument should be a directory!")

    if (
        not (args.checkpoint_dir / "configuration_RW.py").is_file()
        or not (args.checkpoint_dir / "modelling_RW.py").is_file()
    ):
        raise ValueError(
            "The model directory should contain configuration_RW.py and modelling_RW.py files! Are you sure this is a custom code checkpoint?"
        )
    (args.checkpoint_dir / "configuration_RW.py").unlink()
    (args.checkpoint_dir / "modelling_RW.py").unlink()

    config = args.checkpoint_dir / "config.json"
    text = config.read_text()
    text = text.replace("RWForCausalLM", "FalconForCausalLM")
    text = text.replace("RefinedWebModel", "falcon")
    text = text.replace("RefinedWeb", "falcon")
    json_config = json.loads(text)
    del json_config["auto_map"]

    if "n_head" in json_config:
        json_config["num_attention_heads"] = json_config.pop("n_head")
    if "n_layer" in json_config:
        json_config["num_hidden_layers"] = json_config.pop("n_layer")
    if "n_head_kv" in json_config:
        json_config["num_kv_heads"] = json_config.pop("n_head_kv")
        json_config["new_decoder_architecture"] = True
    else:
        json_config["new_decoder_architecture"] = False
    bos_token_id = json_config.get("bos_token_id", 1)
    eos_token_id = json_config.get("eos_token_id", 2)
    config.unlink()
    config.write_text(json.dumps(json_config, indent=2, sort_keys=True))

    tokenizer_config = args.checkpoint_dir / "tokenizer_config.json"
    if tokenizer_config.is_file():
        text = tokenizer_config.read_text()
        json_config = json.loads(text)
        if json_config["tokenizer_class"] == "PreTrainedTokenizerFast":
            json_config["model_input_names"] = ["input_ids", "attention_mask"]
            tokenizer_config.unlink()
            tokenizer_config.write_text(json.dumps(json_config, indent=2, sort_keys=True))

    generation_config_path = args.checkpoint_dir / "generation_config.json"
    generation_dict = {
        "_from_model_config": True,
        "bos_token_id": bos_token_id,
        "eos_token_id": eos_token_id,
        "transformers_version": "4.33.0.dev0",
    }
    generation_config_path.write_text(json.dumps(generation_dict, indent=2, sort_keys=True))
    print("Done! Please double-check that the new checkpoint works as expected.")
