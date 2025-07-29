"""
This is a script to run 6eoo inference using the transfomers lib. 

You can use to vibe check model outputs. 

I see outputs like:
- "The image depicts a vibrant street scene at the entrance to a Chinatown, likely in an urban area..."

(which is correct given: https://www.ilankelman.org/stopsigns/australia.jpg)

I've tested this script with both:
- The raw tif exported checkpoint `gsutil -m cp -r gs://cohere-command/experimental_models/c3-sweep-6eoog65n-e0ry-fp16/tif_export`
- The checkpoint I uploaded to a private HF repo `julianmack/command-vision-test01`

Change the model_id below to switch between them.
"""
import os
import sys
import argparse

# There is an install of HF transformers in the env. Force import to use 
# the cohere-transformers version
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
from transformers import AutoProcessor, AutoModelForCausalLM, AutoImageProcessor
import torch
from transformers.models.cohere2_vision.processing_cohere2_vision import (
    Cohere2VisionProcessor,
)
from transformers.models.cohere2_vision.image_processing_cohere2_vision import (
    Cohere2VisionImageProcessor,
)
from transformers import Cohere2VisionForConditionalGeneration

from PIL import Image
from pathlib import Path
CHAT_TEMPLATE = """{%- for message in messages -%}
    <|START_OF_TURN_TOKEN|>{{ message.role | replace("user", "<|USER_TOKEN|>") | replace("assistant", "<|CHATBOT_TOKEN|><|START_RESPONSE|>") | replace("system", "<|SYSTEM_TOKEN|>") }}
    {%- if message.content is defined -%}
        {%- if message.content is string -%}
{{ message.content }}
        {%- else -%}
            {%- for item in message.content -%}
                {%- if item.type == 'image' -%}
<image>
                {%- elif item.type == 'text' -%}
{{ item.text }}
                {%- endif -%}
            {%- endfor -%}
        {%- endif -%}
    {%- elif message.message is defined -%}
        {%- if message.message is string -%}
{{ message.message }}
        {%- else -%}
            {%- for item in message.message -%}
                {%- if item.type == 'image' -%}
<image>
                {%- elif item.type == 'text' -%}
{{ item.text }}
                {%- endif -%}
            {%- endfor -%}
        {%- endif -%}
    {%- endif -%}
    {%- if message.role == "assistant" -%}
<|END_RESPONSE|>
    {%- endif -%}
<|END_OF_TURN_TOKEN|>
{%- endfor -%}
{%- if add_generation_prompt -%}
<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>
{%- endif -%}"""

def build_content_turns():
    jpg_local_path = '/tmp/australia.jpg'
    if not Path("/tmp/australia.jpg").exists():
        os.system("wget https://www.ilankelman.org/stopsigns/australia.jpg -O /tmp/australia.jpg")
    
    pil_img = Image.open(jpg_local_path).convert("RGB")
    turn1 = {
            "role": "user",
            "content": [
                {"type": "text", "text": "And here’s a local image:"},
                {"type": "image", "image": pil_img},
            ],
        }
    turn2 = {
            "role": "user",
            "content": [{"type": "text", "text": "What did you see in the last image?"}],
        }
    
    return [turn1, turn2]


if __name__ == "__main__":
    """
    To use local weights:
        mkdir -p /root/models/c3-sweep-6eoog65n-e0ry-fp16-tif_export
        gsutil -m cp -r gs://cohere-command/experimental_models/c3-sweep-6eoog65n-e0ry-fp16/tif_export /root/models/c3-sweep-6eoog65n-e0ry-fp16-tif_export
    """
    parser = argparse.ArgumentParser(description="Run 6eoo inference script")
    parser.add_argument(
        "--model_id",
        type=str,
        default="CohereLabs/command-a-vision-07-2025",
        help="Path to the model or Hugging Face model ID",
    )
    parser.add_argument(
        "--small", action="store_true",
        help="Use a smaller model for testing purposes. Overrides model_id.",
    )
    args = parser.parse_args()
    model_id = args.model_id if not args.small else "../ckpts/tmp_tif_export_7b/poseidon"

    processor = AutoProcessor.from_pretrained(model_id, use_auth_token=True)
    img_processor = AutoImageProcessor.from_pretrained(model_id, use_auth_token=True)

    assert isinstance(processor, Cohere2VisionProcessor)
    assert isinstance(processor.image_processor, Cohere2VisionImageProcessor)

    print("processor loaded")
    if args.small:
        processor.chat_template = CHAT_TEMPLATE
    content_turns = build_content_turns()

    inputs = processor.apply_chat_template(
        content_turns,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    print("inputs prepared")

    model, loading_info = Cohere2VisionForConditionalGeneration.from_pretrained(
        model_id, device_map="auto", output_loading_info=True,
    )
    missing, unexpected = loading_info["missing_keys"], loading_info["unexpected_keys"]
    try:
        assert not unexpected, 'All keys in source checkpoint should be used'
        assert not missing, 'No keys should be missing'
    except AssertionError as e:
        breakpoint()
        raise e
    
    inputs = inputs.to(model.device)

    print("running generation...")
    gen_tokens = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=True,
        temperature=0.3,
        use_cache=False,
    )

    print(
        processor.tokenizer.decode(
            gen_tokens[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
    )
