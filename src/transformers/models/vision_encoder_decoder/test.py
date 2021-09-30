from transformers import BertConfig, BertLMHeadModel, VisionEncoderDecoderModel, ViTConfig, ViTModel


encoder_config = ViTConfig(image_size=384)
encoder = ViTModel(encoder_config)
decoder_config = BertConfig.from_pretrained(
    "bert-large-uncased", is_decoder=True, add_cross_attention=True, encoder_hidden_size=768, tie_word_embeddings=False
)
decoder = BertLMHeadModel(decoder_config)

model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)

for name, param in model.named_parameters():
    print(name, param.shape)
