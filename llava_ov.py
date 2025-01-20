from transformers import (
    AutoVideoProcessor,
    AutoImageProcessor,
    LlavaOnevisionVideoProcessor,
    LlavaOnevisionImageProcessor,
    LlavaNextVideoVideoProcessor,
    InstructBlipVideoVideoProcessor,
    VideoLlavaVideoProcessor,
    Qwen2VLVideoProcessor,
)

classes = [
    LlavaOnevisionVideoProcessor,
    LlavaNextVideoVideoProcessor,
    VideoLlavaVideoProcessor,
    Qwen2VLVideoProcessor
]

ckpts = [
    "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
    "llava-hf/LLaVA-NeXT-Video-7B-hf",
    "LanguageBind/Video-LLaVA-7B-hf",
    "Qwen/Qwen2-VL-2B-Instruct",
]

video_processor = InstructBlipVideoVideoProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
video_processor.save_pretrained("tmp")
video_processor = InstructBlipVideoVideoProcessor.from_pretrained("tmp")

for processor_class, ckpt in zip(classes, ckpts):

    # Can load video processors from auto class and it fails if we load non-video model
    video_processor = AutoVideoProcessor.from_pretrained(ckpt)
    try:
        video_processor = AutoVideoProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    except:
        pass
    else:
        raise ValueError("video processor should not load the image models")


    # Can save and load the video processor back
    video_processor.save_pretrained("tmp")
    video_processor = processor_class.from_pretrained("tmp/video_preprocessor_config.json")
    video_processor = processor_class.from_pretrained("tmp")
    video_processor = AutoVideoProcessor.from_pretrained("tmp/video_preprocessor_config.json")


    # Check backward compatibility and load with old naming
    video_processor = LlavaOnevisionImageProcessor.from_pretrained(ckpt)
    # video_processor = LlavaOnevisionImageProcessor.from_pretrained("tmp")


    # Can load video processor from non-auto class and set new class params for processing
    video_processor = processor_class.from_pretrained(ckpt, do_normalize=False, foo=False)
    assert video_processor.do_normalize is False


# Image processing works as it was working so nothing broke
AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")


