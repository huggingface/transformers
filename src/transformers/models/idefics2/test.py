from transformers.models.idefics2.modeling_idefics2 import Idefics2Model, Idefics2Config


# config = Idefics2Config.from_pretrained("HuggingFaceM4/idefics2")

# config.save_pretrained("dummy_config")

model = Idefics2Model.from_pretrained("HuggingFaceM4/idefics2", ignore_mismatched_sizes=True)
