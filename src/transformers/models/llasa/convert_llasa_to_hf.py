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
import torch

from transformers import AutoModelForCausalLM, LlasaConfig, LlasaForCausalLM, LlasaTokenizer


@hydra.main(version_base=None, config_path="conversion_configs", config_name="llasa_1b")
def conversion(config):
    print(f"Converting {config.original_model.repo} to Transformers format...")

    # load original model
    model_orig = AutoModelForCausalLM.from_pretrained(config.original_model.repo)

    # load Llasa model from Llama model
    # -- tokenizer
    tokenizer = LlasaTokenizer.from_pretrained_llm(
        config.llm_model,
        model_max_length=config.original_model.max_length,
        codebook_size=config.original_model.codebook_size,
    )
    speech_end_id = tokenizer.convert_tokens_to_ids(tokenizer.llasa_token["speech_generation_end"])

    # -- model config and model itself
    model_config = LlasaConfig.from_pretrained_llm(config.llm_model, **config.original_model)
    
    # -- generation configuration
    model_config.eos_token_id = speech_end_id

    # -- create model
    model = LlasaForCausalLM(model_config)
    if config.remote_repo.dtype == "bfloat16":
        model.to(torch.bfloat16)
        print("Model dtype : ", model.dtype)
    assert model.lm_head.weight.size(0) == len(tokenizer)
    assert model.lm_head.weight.size(0) == model_orig.lm_head.weight.size(0)
    
    # -- copy model weights
    model.load_state_dict(model_orig.state_dict())

    # -- save converted model
    if config.remote_repo.id:
        print(f"Pushing a {model.__class__.__name__} to Hugging Face Hub: {config.remote_repo.id}")
        model.push_to_hub(config.remote_repo.id, private=config.remote_repo.private, use_temp_dir=True)
        print(f"Pushing a {tokenizer.__class__.__name__} to Hugging Face Hub: {config.remote_repo.id}")
        tokenizer.push_to_hub(config.remote_repo.id, private=True, use_temp_dir=True)
    if config.local_model_path:
        model.save_pretrained(config.local_model_path)
        print(f"Model saved locally at: {config.local_model_path}")


if __name__ == "__main__":
    conversion()
