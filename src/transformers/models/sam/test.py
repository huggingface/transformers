from transformers import (
    SamVisionAutoBackboneConfig,
    SamPromptEncoderConfig,
    SamMaskDecoderConfig,
    SamModel,
    SamConfig,
    TinyVitConfig,
    SamModel,
    SamVisionConfig,
)

backbone_config = TinyVitConfig()
vision_config = SamVisionAutoBackboneConfig(backbone_config=backbone_config)
# vision_config = SamVisionConfig()
prompt_encoder_config = SamPromptEncoderConfig()
mask_decoder_config = SamMaskDecoderConfig()

config = SamConfig(vision_config, prompt_encoder_config, mask_decoder_config)

model = SamModel(config=config)