from transformers.models.idefics2.modeling_idefics2 import AutoModelForCausalLM


# config = Idefics2Config.from_pretrained("HuggingFaceM4/idefics2")

# import pdb; pdb.set_trace()

# config.save_pretrained("dummy_config")

# model = Idefics2Model.from_pretrained("HuggingFaceM4/idefics2", ignore_mismatched_sizes=True)
# model = Idefics2Model.from_pretrained("HuggingFaceM4/idefics2", trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained("HuggingFaceM4/idefics2", trust_remote_code=True)
