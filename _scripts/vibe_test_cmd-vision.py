"""
This is a script to run command-a-vision inference using the transfomers lib. 

You can use to vibe check model outputs. 

I see outputs like:
- "The image depicts a vibrant street scene at the entrance to a Chinatown, likely in an urban area..."

(which is correct given: https://www.ilankelman.org/stopsigns/australia.jpg)
"""
import os
import argparse

from transformers import AutoProcessor, AutoImageProcessor
from transformers.models.cohere2_vision.processing_cohere2_vision import (
    Cohere2VisionProcessor,
)
try:
    from transformers.models.cohere2_vision.image_processing_cohere2_vision_fast import (
        Cohere2VisionImageProcessorFast,
    )
except ModuleNotFoundError:
    from transformers.models.cohere2_vision.image_processing_cohere2_vision import Cohere2VisionImageProcessor as Cohere2VisionImageProcessorFast
    
from transformers import Cohere2VisionForConditionalGeneration

from PIL import Image
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="Run 6eoo inference script")
    parser.add_argument(
        "--model_id",
        type=str,
        default="CohereLabs/command-a-vision-07-2025",
        help="Path to the model or Hugging Face model ID",
    )
    args = parser.parse_args()
    model_id = args.model_id
    processor = AutoProcessor.from_pretrained(model_id, use_auth_token=True)
    img_processor = AutoImageProcessor.from_pretrained(model_id, use_auth_token=True)

    assert isinstance(processor, Cohere2VisionProcessor)
    assert isinstance(processor.image_processor, Cohere2VisionImageProcessorFast)

    print("processor loaded")
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
        model_id, device_map="auto", output_loading_info=True, torch_dtype="auto",
    )
    missing, unexpected = loading_info["missing_keys"], loading_info["unexpected_keys"]
    
    assert not unexpected, 'All keys in source checkpoint should be used'
    assert not missing, 'No keys should be missing'
    
    inputs = inputs.to(model.device)

    print("running generation...")
    gen_tokens = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=True,
        temperature=0.3,
        top_p=0.75,
    )

    print(
        processor.tokenizer.decode(
            gen_tokens[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
    )
