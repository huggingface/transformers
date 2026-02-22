# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from transformers.models.omnivinci.configuration_omnivinci import OmniVinciConfig
from transformers.models.omnivinci.modeling_omnivinci import OmniVinciForCausalLM
from transformers.models.omnivinci.processing_omnivinci import OmniVinciProcessor


@torch.inference_mode()
def main() -> None:
    model_path = "/fs/nexus-projects/JSALT_workshop/lasha/Dev/comni"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    config = OmniVinciConfig.from_pretrained(model_path)
    config._name_or_path = str(model_path)
    config.load_audio_in_video = True
    config.num_video_frames = 128
    config.audio_chunk_length = "max_3600"

    model = OmniVinciForCausalLM.from_pretrained(
        model_path,
        config=config,
        dtype=dtype,
        device_map="auto",
    ).eval()
    processor = OmniVinciProcessor.from_pretrained(
        model_path, config=model.config, padding_side="left", use_fast=False
    )

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": "nvidia.mp4"},
                {
                    "type": "text",
                    "text": "Assess the video, followed by a detailed description of it's video and audio contents.",
                },
            ],
        }
    ]

    conversation = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor([conversation])

    inputs.input_ids = inputs.input_ids.to(model.device)
    inputs.attention_mask = inputs.attention_mask.to(model.device)

    # Build multimodal prefill embeddings so generation cache positions match the full multimodal prompt length.
    inputs_embeds, _, attention_mask = model._embed(
        inputs.input_ids,
        getattr(inputs, "media", None),
        getattr(inputs, "media_config", None),
        None,
        inputs.attention_mask,
    )

    output_ids = model.generate(
        input_ids=inputs.input_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=1024,
        do_sample=False,
    )

    generated_ids = output_ids[:, inputs.input_ids.shape[1] :]
    print(processor.batch_decode(generated_ids, skip_special_tokens=True)[0])


if __name__ == "__main__":
    main()
