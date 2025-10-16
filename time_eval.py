from transformers import AutoConfig, LlavaConfig


remote_text_config = AutoConfig.from_pretrained("AI4Chem/ChemLLM-7B-Chat" trust_remote_code=True)
local_vision_config = AutoConfig.from_pretrained("google/siglip2-so400m-patch14-384")
config = LlavaConfig(text_config=remote_text_config, vision_config=local_vision_config, image_token_id=92544)
config.save_pretrained("local_llava")


config = LlavaConfig.from_pretrained("local_llava", trust_remote_code=True)
