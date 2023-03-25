from transformers import Pix2StructConfig, Pix2StructTextConfig, Pix2StructForConditionalGeneration

config = Pix2StructConfig(text_config=dict(tie_word_embeddings=False), tie_word_embeddings=False)

model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-base", config=config)