"""

Setup
```
uv pip install hydra-core
```

Conversion commands
```bash
python src/transformers/models/llasa/convert_llasa_to_hf.py -cn llasa_1b
python src/transformers/models/llasa/convert_llasa_to_hf.py -cn llasa_3b
python src/transformers/models/llasa/convert_llasa_to_hf.py -cn llasa_8b
```

"""

import hydra

from transformers import AutoModelForCausalLM, LlasaConfig, LlasaForCausalLM


@hydra.main(version_base=None, config_path="conversion_configs", config_name="llasa_1b")
def conversion(config):
    print(f"Converting {config.model_repo} to Transformers format...")

    # load original model
    model_orig = AutoModelForCausalLM.from_pretrained(config.model_repo)

    # load Llasa model
    model_config = LlasaConfig.from_pretrained(config.llama_model)
    model = LlasaForCausalLM(model_config)

    # copy weights
    model.load_state_dict(model_orig.state_dict())

    # TODO tokenizer
    # tokenizer = AutoTokenizer.from_pretrained(config.model_repo)

    # save converted model
    model.save_pretrained(config.local_model_path)
    if config.remote_repo_id:
        print(f"Pushing model to Hugging Face Hub: {config.remote_repo_id}")
        model.push_to_hub(config.remote_repo_id, private=True)


if __name__ == "__main__":
    conversion()
