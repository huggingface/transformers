from transformers import TrOCRConfig, TrOCRForCausalLM, VisionEncoderDecoderModel, ViTConfig, ViTModel


encoder_config = ViTConfig(image_size=384)
encoder = ViTModel(encoder_config)
decoder_config = TrOCRConfig()
decoder = TrOCRForCausalLM(decoder_config)

model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)

for name, param in model.named_parameters():
    print(name, param.shape)
