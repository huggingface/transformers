# SPDX-License-Identifier: Apache-2.0
"""
This example shows how to use vLLM for running offline inference with
the correct prompt format on vision language models for text generation.

For most models, the prompt format should follow corresponding examples
on HuggingFace model repository.
"""
import os
import random

from transformers import AutoTokenizer
from transformers.image_utils import load_image

from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset
from vllm.utils import FlexibleArgumentParser

# NOTE: The default `max_num_seqs` and `max_model_len` may result in OOM on
# lower-end GPUs.
# Unless specified, these settings have been tested to work on a single L4.

os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# Aria
def run_aria(question: str, modality: str):
    assert modality == "image"
    model_name = "rhymes-ai/Aria"

    # NOTE: Need L40 (or equivalent) to avoid OOM
    llm = LLM(model=model_name,
              max_model_len=4096,
              max_num_seqs=2,
              dtype="bfloat16",
              disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache)

    prompt = (f"<|im_start|>user\n<fim_prefix><|img|><fim_suffix>{question}"
              "<|im_end|>\n<|im_start|>assistant\n")

    stop_token_ids = [93532, 93653, 944, 93421, 1019, 93653, 93519]
    return llm, prompt, stop_token_ids


# BLIP-2
def run_blip2(question: str, modality: str):
    assert modality == "image"

    # BLIP-2 prompt format is inaccurate on HuggingFace model repository.
    # See https://huggingface.co/Salesforce/blip2-opt-2.7b/discussions/15#64ff02f3f8cf9e4f5b038262 #noqa
    prompt = f"Question: {question} Answer:"
    llm = LLM(model="Salesforce/blip2-opt-2.7b",
              disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache)
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# Chameleon
def run_chameleon(question: str, modality: str):
    assert modality == "image"

    prompt = f"{question}<image>"
    llm = LLM(model="facebook/chameleon-7b",
              max_model_len=4096,
              max_num_seqs=2,
              disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache)
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# Deepseek-VL2
def run_deepseek_vl2(question: str, modality: str):
    assert modality == "image"

    model_name = "deepseek-ai/deepseek-vl2-tiny"

    llm = LLM(model=model_name,
              max_model_len=4096,
              max_num_seqs=2,
              disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache,
              hf_overrides={"architectures": ["DeepseekVLV2ForCausalLM"]})

    prompt = f"<|User|>: <image>\n{question}\n\n<|Assistant|>:"
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# Fuyu
def run_fuyu(question: str, modality: str):
    assert modality == "image"

    prompt = f"{question}\n"
    llm = LLM(model="adept/fuyu-8b",
              max_model_len=2048,
              max_num_seqs=2,
              disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache)
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# GLM-4v
def run_glm4v(question: str, modality: str):
    assert modality == "image"
    model_name = "THUDM/glm-4v-9b"

    llm = LLM(model=model_name,
              max_model_len=2048,
              max_num_seqs=2,
              trust_remote_code=True,
              enforce_eager=True,
              hf_overrides={"architectures": ["GLM4VForCausalLM"]},
              disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache)

    prompt = f"<|user|>\n<|begin_of_image|><|endoftext|><|end_of_image|>\
        {question}<|assistant|>"

    stop_token_ids = [151329, 151336, 151338]
    return llm, prompt, stop_token_ids


# H2OVL-Mississippi
def run_h2ovl(question: str, modality: str):
    assert modality == "image"

    model_name = "h2oai/h2ovl-mississippi-2b"

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=8192,
        disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)
    messages = [{'role': 'user', 'content': f"<image>\n{question}"}]
    prompt = tokenizer.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)

    # Stop tokens for H2OVL-Mississippi
    # https://huggingface.co/h2oai/h2ovl-mississippi-2b
    stop_token_ids = [tokenizer.eos_token_id]
    return llm, prompt, stop_token_ids


# Idefics3-8B-Llama3
def run_idefics3(question: str, modality: str):
    assert modality == "image"
    # model_name = "HuggingFaceM4/Idefics3-8B-Llama3"
    model_name = "HuggingFaceTB/SmolVLM-256M-Instruct"

    llm = LLM(
        model=model_name,
        max_model_len=5000,
        max_num_batched_tokens=1700,
        disable_mm_preprocessor_cache=True,
        gpu_memory_utilization=0.6,
        dtype="bfloat16",
        enforce_eager=True,
        max_num_seqs=1700,
        model_impl="transformers",
        limit_mm_per_prompt={"image": 2},
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        # if you are running out of memory, you can reduce the "longest_edge".
        # see: https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3#model-optimizations
        # mm_processor_kwargs={
        #     "size": {
        #         "longest_edge": 3 * 364
        #     },
        # },
    )
    prompt = (
        f"<|begin_of_text|>User:<image><image>{question}<end_of_utterance>\nAssistant:"
    )
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# InternVL
def run_internvl(question: str, modality: str):
    assert modality == "image"

    model_name = "OpenGVLab/InternVL2-2B"

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=4096,
        disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)
    messages = [{'role': 'user', 'content': f"<image>\n{question}"}]
    prompt = tokenizer.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)

    # Stop tokens for InternVL
    # models variants may have different stop tokens
    # please refer to the model card for the correct "stop words":
    # https://huggingface.co/OpenGVLab/InternVL2-2B/blob/main/conversation.py
    stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
    return llm, prompt, stop_token_ids


# LLaVA-1.5
def run_llava(question: str, modality: str):
    assert modality == "image"

    prompt = f"USER: [IMG][IMG]\n{question}\nASSISTANT:"
    #prompt = f"USER: <image><image>\n{question}\nASSISTANT:"

    llm = LLM(
        model='mistral-community/pixtral-12b',
        #model="llava-hf/llava-1.5-7b-hf",
        max_model_len=4000,
        max_num_batched_tokens=1500,
        disable_mm_preprocessor_cache=True, # args.disable_mm_preprocessor_cache,
        gpu_memory_utilization=0.5,
        dtype="float16",
        enforce_eager=True,
        max_num_seqs=1500,
        model_impl="transformers",
        # hf_overrides={"architectures": ["TransformersMultimodalModel"]},
        # tensor_parallel_size=1,
        limit_mm_per_prompt={"image": 2},
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        # mm_processor_kwargs={"size": {"height": 336, "width": 400}},
    )
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# LLaVA-1.6/LLaVA-NeXT
def run_llava_next(question: str, modality: str):
    assert modality == "image"

    prompt = f"[INST] <image>\n{question} [/INST]"
    llm = LLM(model="llava-hf/llava-v1.6-mistral-7b-hf",
              max_model_len=8192,
              disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache)
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# LlaVA-NeXT-Video
# Currently only support for video input
def run_llava_next_video(question: str, modality: str):
    assert modality == "video"

    prompt = f"USER: <video>\n{question} ASSISTANT:"
    llm = LLM(model="llava-hf/LLaVA-NeXT-Video-7B-hf",
              max_model_len=8192,
              disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache)
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# LLaVA-OneVision
def run_llava_onevision(question: str, modality: str):

    if modality == "video":
        prompt = f"<|im_start|>user <video>\n{question}<|im_end|> \
        <|im_start|>assistant\n"

    elif modality == "image":
        prompt = f"<|im_start|>user <image>\n{question}<|im_end|> \
        <|im_start|>assistant\n"

    llm = LLM(model="llava-hf/llava-onevision-qwen2-7b-ov-hf",
              max_model_len=16384,
              disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache)
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# Mantis
def run_mantis(question: str, modality: str):
    assert modality == "image"

    llama3_template = '<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'  # noqa: E501
    prompt = llama3_template.format(f"{question}\n<image>")

    llm = LLM(
        model="TIGER-Lab/Mantis-8B-siglip-llama3",
        max_model_len=4096,
        hf_overrides={"architectures": ["MantisForConditionalGeneration"]},
        disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache,
    )
    stop_token_ids = [128009]
    return llm, prompt, stop_token_ids


# MiniCPM-V
def run_minicpmv_base(question: str, modality: str, model_name):
    assert modality in ["image", "video"]
    # If you want to use `MiniCPM-o-2_6` with audio inputs, check `audio_language.py` # noqa

    # 2.0
    # The official repo doesn't work yet, so we need to use a fork for now
    # For more details, please see: See: https://github.com/vllm-project/vllm/pull/4087#issuecomment-2250397630 # noqa
    # model_name = "HwwwH/MiniCPM-V-2"

    # 2.5
    # model_name = "openbmb/MiniCPM-Llama3-V-2_5"

    # 2.6
    # model_name = "openbmb/MiniCPM-V-2_6"
    # o2.6

    # modality supports
    # 2.0: image
    # 2.5: image
    # 2.6: image, video
    # o2.6: image, video, audio
    # model_name = "openbmb/MiniCPM-o-2_6"
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)
    llm = LLM(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=2,
        trust_remote_code=True,
        disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache,
    )
    # NOTE The stop_token_ids are different for various versions of MiniCPM-V
    # 2.0
    # stop_token_ids = [tokenizer.eos_id]

    # 2.5
    # stop_token_ids = [tokenizer.eos_id, tokenizer.eot_id]

    # 2.6 / o2.6
    stop_tokens = ['<|im_end|>', '<|endoftext|>']
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]

    modality_placeholder = {
        "image": "(<image>./</image>)",
        "video": "(<video>./</video>)",
    }

    messages = [{
        'role': 'user',
        'content': f'{modality_placeholder[modality]}\n{question}'
    }]
    prompt = tokenizer.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)
    return llm, prompt, stop_token_ids


def run_minicpmo(question: str, modality: str):
    return run_minicpmv_base(question, modality, "openbmb/MiniCPM-o-2_6")


def run_minicpmv(question: str, modality: str):
    return run_minicpmv_base(question, modality, "openbmb/MiniCPM-V-2_6")


# LLama 3.2
def run_mllama(question: str, modality: str):
    assert modality == "image"

    model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"

    # Note: The default setting of max_num_seqs (256) and
    # max_model_len (131072) for this model may cause OOM.
    # You may lower either to run this example on lower-end GPUs.

    # The configuration below has been confirmed to launch on a single L40 GPU.
    llm = LLM(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=16,
        disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    messages = [{
        "role":
        "user",
        "content": [{
            "type": "image"
        }, {
            "type": "text",
            "text": f"{question}"
        }]
    }]
    prompt = tokenizer.apply_chat_template(messages,
                                           add_generation_prompt=True,
                                           tokenize=False)
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# Molmo
def run_molmo(question, modality):
    assert modality == "image"

    model_name = "allenai/Molmo-7B-D-0924"

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        dtype="bfloat16",
        disable_mm_preprocessor_cache=True,
        gpu_memory_utilization=0.6,
        enforce_eager=True,
        # limit_mm_per_prompt={"image": 2},
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
    )

    prompt = question
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# NVLM-D
def run_nvlm_d(question: str, modality: str):
    assert modality == "image"

    model_name = "nvidia/NVLM-D-72B"

    # Adjust this as necessary to fit in GPU
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=4096,
        # tensor_parallel_size=4,
        disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)
    messages = [{'role': 'user', 'content': f"<image>\n{question}"}]
    prompt = tokenizer.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# PaliGemma
def run_paligemma(question: str, modality: str):
    assert modality == "image"

    # PaliGemma has special prompt format for VQA
    prompt = "caption en"
    llm = LLM(model="google/paligemma-3b-mix-224",
              disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache)
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# PaliGemma 2
def run_paligemma2(question: str, modality: str):
    assert modality == "image"

    # PaliGemma 2 has special prompt format for VQA
    prompt = "caption en"
    llm = LLM(model="google/paligemma2-3b-ft-docci-448",
              disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache)
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# Phi-3-Vision
def run_phi3v(question: str, modality: str):
    assert modality == "image"

    prompt = f"<|user|>\n<|image_1|>\n{question}<|end|>\n<|assistant|>\n"

    # num_crops is an override kwarg to the multimodal image processor;
    # For some models, e.g., Phi-3.5-vision-instruct, it is recommended
    # to use 16 for single frame scenarios, and 4 for multi-frame.
    #
    # Generally speaking, a larger value for num_crops results in more
    # tokens per image instance, because it may scale the image more in
    # the image preprocessing. Some references in the model docs and the
    # formula for image tokens after the preprocessing
    # transform can be found below.
    #
    # https://huggingface.co/microsoft/Phi-3.5-vision-instruct#loading-the-model-locally
    # https://huggingface.co/microsoft/Phi-3.5-vision-instruct/blob/main/processing_phi3_v.py#L194
    llm = LLM(
        model="microsoft/Phi-3.5-vision-instruct",
        trust_remote_code=True,
        max_model_len=4096,
        max_num_seqs=2,
        # Note - mm_processor_kwargs can also be passed to generate/chat calls
        mm_processor_kwargs={"num_crops": 16},
        disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache,
    )
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# Pixtral HF-format
def run_pixtral_hf(question: str, modality: str):
    assert modality == "image"

    model_name = "mistral-community/pixtral-12b"

    # NOTE: Need L40 (or equivalent) to avoid OOM
    llm = LLM(
        model=model_name,
        max_model_len=2000,
        max_num_batched_tokens=2000,
        max_num_seqs=1,
        disable_mm_preprocessor_cache=True, # args.disable_mm_preprocessor_cache,
        gpu_memory_utilization=0.5,
        dtype="float16",
        enforce_eager=True,
        # limit_mm_per_prompt={"image": 2},
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
    )

    prompt = f"<s>[INST]{question}\n[IMG][/INST]"
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# Qwen
def run_qwen_vl(question: str, modality: str):
    assert modality == "image"

    llm = LLM(
        model="Qwen/Qwen-VL",
        trust_remote_code=True,
        max_model_len=1024,
        max_num_seqs=2,
        hf_overrides={"architectures": ["QwenVLForConditionalGeneration"]},
        disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache,
    )

    prompt = f"{question}Picture 1: <img></img>\n"
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# Qwen2-VL
def run_qwen2_vl(question: str, modality: str):

    model_name = "Qwen/Qwen2-VL-2B-Instruct"

    llm = LLM(
        model=model_name,
        # Note - mm_processor_kwargs can also be passed to generate/chat calls
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
        },
        max_model_len=4000,
        max_num_batched_tokens=3000,
        disable_mm_preprocessor_cache=True, # args.disable_mm_preprocessor_cache,
        gpu_memory_utilization=0.8,
        dtype="bfloat16",
        enforce_eager=True,
        max_num_seqs=3000,
        model_impl="transformers",
        limit_mm_per_prompt={"image": 2},
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
    )

    if modality == "image":
        placeholder = "<|image_pad|>"
    elif modality == "video":
        placeholder = "<|video_pad|>"

    prompt = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
              f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|> and <|vision_start|>{placeholder}<|vision_end|>"
              f"{question}<|im_end|>\n"
              "<|im_start|>assistant\n")
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# Qwen2.5-VL
def run_qwen2_5_vl(question: str, modality: str):

    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"

    llm = LLM(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=5,
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
        },
        disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache,
    )

    if modality == "image":
        placeholder = "<|image_pad|>"
    elif modality == "video":
        placeholder = "<|video_pad|>"

    prompt = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
              f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
              f"{question}<|im_end|>\n"
              "<|im_start|>assistant\n")
    stop_token_ids = None
    return llm, prompt, stop_token_ids


model_example_map = {
    "aria": run_aria,
    "blip-2": run_blip2,
    "chameleon": run_chameleon,
    "deepseek_vl_v2": run_deepseek_vl2,
    "fuyu": run_fuyu,
    "glm4v": run_glm4v,
    "h2ovl_chat": run_h2ovl,
    "idefics3": run_idefics3,
    "internvl_chat": run_internvl,
    "llava": run_llava,
    "llava-next": run_llava_next,
    "llava-next-video": run_llava_next_video,
    "llava-onevision": run_llava_onevision,
    "mantis": run_mantis,
    "minicpmo": run_minicpmo,
    "minicpmv": run_minicpmv,
    "mllama": run_mllama,
    "molmo": run_molmo,
    "NVLM_D": run_nvlm_d,
    "paligemma": run_paligemma,
    "paligemma2": run_paligemma2,
    "phi3_v": run_phi3v,
    "pixtral_hf": run_pixtral_hf,
    "qwen_vl": run_qwen_vl,
    "qwen2_vl": run_qwen2_vl,
    "qwen2_5_vl": run_qwen2_5_vl,
}


def get_multi_modal_input(args):
    """
    return {
        "data": image or video,
        "question": question,
    }
    """
    if args.modality == "image":
        # Input image and question
        image = ImageAsset("cherry_blossom") \
            .pil_image.convert("RGB")
        img_question = "What is the content of this image?"

        return {
            "data": image,
            "question": img_question,
        }

    if args.modality == "video":
        # Input video and question
        video = VideoAsset(name="sample_demo_1.mp4",
                           num_frames=args.num_frames).np_ndarrays
        vid_question = "Why is this video funny?"

        return {
            "data": video,
            "question": vid_question,
        }

    msg = f"Modality {args.modality} is not supported."
    raise ValueError(msg)


def apply_image_repeat(image_repeat_prob, num_prompts, data, prompt, modality):
    """Repeats images with provided probability of "image_repeat_prob". 
    Used to simulate hit/miss for the MM preprocessor cache.
    """
    assert (image_repeat_prob <= 1.0 and image_repeat_prob >= 0)
    no_yes = [0, 1]
    probs = [1.0 - image_repeat_prob, image_repeat_prob]

    inputs = []
    cur_image = data
    for i in range(num_prompts):
        if image_repeat_prob is not None:
            res = random.choices(no_yes, probs)[0]
            if res == 0:
                # No repeat => Modify one pixel
                cur_image = cur_image.copy()
                new_val = (i // 256 // 256, i // 256, i % 256)
                cur_image.putpixel((0, 0), new_val)

        inputs.append({
            "prompt": prompt,
            "multi_modal_data": {
                modality: cur_image
            }
        })

    return inputs


def main(args):
    model = args.model_type
    if model not in model_example_map:
        raise ValueError(f"Model type {model} is not supported.")

    modality = args.modality
    mm_input = get_multi_modal_input(args)
    data = mm_input["data"]
    images = [load_image(f"https://picsum.photos/id/23{i}/500/700") for i in range(args.num_prompts)]
    question = mm_input["question"]

    llm, prompt, stop_token_ids = model_example_map[model](question, modality)

    # We set temperature to 0.2 so that outputs can be different
    # even when all prompts are identical when running batch inference.
    sampling_params = SamplingParams(temperature=0.2,
                                     max_tokens=10,
                                     stop_token_ids=stop_token_ids)

    assert args.num_prompts > 0
    if args.num_prompts == 1:
        # Single inference
        inputs = {
            "prompt": prompt,
            "multi_modal_data": {modality: [images[0], images[0]]},
        }

    else:
        # Batch inference
        if args.image_repeat_prob is not None:
            # Repeat images with specified probability of "image_repeat_prob"
            inputs = apply_image_repeat(args.image_repeat_prob,
                                        args.num_prompts, data, prompt,
                                        modality)
        else:
            # Use the same image for all prompts
            inputs = [{
                "prompt": f"{prompt}",
                "multi_modal_data": {modality: [images[0], images[0]]},
            } for i in range(args.num_prompts)]

    print(inputs)
    if args.time_generate:
        import time
        start_time = time.time()
        outputs = llm.generate(inputs, sampling_params=sampling_params)
        elapsed_time = time.time() - start_time
        print("-- generate time = {}".format(elapsed_time))

    else:
        outputs = llm.generate(inputs, sampling_params=sampling_params)

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description='Demo on using vLLM for offline inference with '
        'vision language models for text generation')
    parser.add_argument('--model-type',
                        '-m',
                        type=str,
                        default="llava",
                        choices=model_example_map.keys(),
                        help='Huggingface "model_type".')
    parser.add_argument('--num-prompts',
                        type=int,
                        default=2,
                        help='Number of prompts to run.')
    parser.add_argument('--modality',
                        type=str,
                        default="image",
                        choices=['image', 'video'],
                        help='Modality of the input.')
    parser.add_argument('--num-frames',
                        type=int,
                        default=16,
                        help='Number of frames to extract from the video.')

    parser.add_argument(
        '--image-repeat-prob',
        type=float,
        default=None,
        help='Simulates the hit-ratio for multi-modal preprocessor cache'
        ' (if enabled)')

    parser.add_argument(
        '--disable-mm-preprocessor-cache',
        action='store_true',
        help='If True, disables caching of multi-modal preprocessor/mapper.')

    parser.add_argument(
        '--time-generate',
        action='store_true',
        help='If True, then print the total generate() call time')

    args = parser.parse_args()
    main(args)
