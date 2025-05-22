# SPDX-License-Identifier: Apache-2.0
"""
This example shows how to use vLLM for running offline inference with
the correct prompt format on vision language models for text generation.

For most models, the prompt format should follow corresponding examples
on HuggingFace model repository.
"""
import os
import random

from transformers import AutoTokenizer, AutoProcessor
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

def run_aria(question: str, modality: str):
    assert modality == "image"
    model_name = "rhymes-ai/Aria"

    # NOTE: Need L40 (or equivalent) to avoid OOM
    llm = LLM(
            model=model_name,
            max_model_len=2000,
            max_num_batched_tokens=1700,
            disable_mm_preprocessor_cache=True,
            gpu_memory_utilization=0.9,
            dtype="bfloat16",
            enforce_eager=True,
            max_num_seqs=1700,
            model_impl="transformers",
            limit_mm_per_prompt={"image": 2},
            enable_prefix_caching=False,
            enable_chunked_prefill=False
        )

    prompt = (f"<|im_start|>user\n<fim_prefix><|img|><fim_suffix>{question}"
              "<|im_end|>\n<|im_start|>assistant\n")

    stop_token_ids = [93532, 93653, 944, 93421, 1019, 93653, 93519]
    return llm, prompt, stop_token_ids

def run_aya_vision(question: str, modality: str):
    assert modality == "image"

    model_name = "CohereLabs/aya-vision-8b"
    llm = LLM(
            model=model_name,
            max_model_len=2500,
            max_num_batched_tokens=2000,
            disable_mm_preprocessor_cache=True,
            gpu_memory_utilization=0.7,
            dtype="float16",
            enforce_eager=True,
            max_num_seqs=2000,
            model_impl="transformers",
            limit_mm_per_prompt={"image": 1},
            enable_prefix_caching=False,
            enable_chunked_prefill=False
        )

    processor = AutoProcessor.from_pretrained(model_name)
    messages = [
        {'role': 'user', 'content': [
            {"type": "text", "text": question},
            {"type": "image"}
        ]}
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt = prompt[-200:] # crop out a super long system prompt, so that the input is processed at once
    return llm, prompt, None

def run_chameleon(question: str, modality: str):
    assert modality == "image"

    prompt = f"<image>{question}"
    model_name = "facebook/chameleon-7b"
    llm = LLM(
            model=model_name,
            max_model_len=1200,
            max_num_batched_tokens=1200,
            disable_mm_preprocessor_cache=True,
            gpu_memory_utilization=0.7,
            dtype="bfloat16",
            enforce_eager=True,
            max_num_seqs=1200,
            model_impl="transformers",
            limit_mm_per_prompt={"image": 1},
            enable_prefix_caching=False,
            enable_chunked_prefill=False
        )
    stop_token_ids = None
    return llm, prompt, stop_token_ids

def run_emu(question: str, modality: str):
    assert modality == "image"

    model_name = "BAAI/Emu3-Chat-hf"
    llm = LLM(
            model=model_name,
            max_model_len=17000,
            max_num_batched_tokens=2500,
            disable_mm_preprocessor_cache=True,
            gpu_memory_utilization=0.5,
            dtype="float16",
            max_num_seqs=2500,
            model_impl="transformers",
            limit_mm_per_prompt={"image": 1},
            enable_prefix_caching=False,
            enable_chunked_prefill=False
        )

    processor = AutoProcessor.from_pretrained(model_name)
    messages = [
        {'role': 'user', 'content': [
            {"type": "text", "text": question},
            {"type": "image"}
        ]}
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return llm, prompt, None

def run_gemma3(question: str, modality: str):
    assert modality == "image"

    model_name = "google/gemma-3-4b-it"
    llm = LLM(
            model=model_name,
            max_model_len=3000,
            max_num_batched_tokens=2500,
            disable_mm_preprocessor_cache=True,
            gpu_memory_utilization=0.5,
            dtype="bfloat16",
            max_num_seqs=2500,
            model_impl="transformers",
            limit_mm_per_prompt={"image": 1},
            enable_prefix_caching=False,
            enable_chunked_prefill=False
        )

    processor = AutoProcessor.from_pretrained(model_name)
    messages = [
        {'role': 'user', 'content': [
            {"type": "text", "text": question},
            {"type": "image"}
        ]}
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return llm, prompt, None

def run_fuyu(question: str, modality: str):
    assert modality == "image"

    prompt = f"{question}\n"
    llm = LLM(
        model="adept/fuyu-8b",
        max_model_len=1500,
        max_num_batched_tokens=1500,
        disable_mm_preprocessor_cache=True,
        gpu_memory_utilization=0.7,
        dtype="bfloat16",
        enforce_eager=True,
        max_num_seqs=1500,
        model_impl="transformers",
        limit_mm_per_prompt={"image": 1},
        enable_prefix_caching=False,
        enable_chunked_prefill=False
    )
    stop_token_ids = None
    return llm, prompt, stop_token_ids

def run_got_ocr(question: str, modality: str): # EXPLICITLY NOT SUPPORTED YET
    assert modality == "image"

    model_name = "stepfun-ai/GOT-OCR-2.0-hf"
    llm = LLM(
            model=model_name,
            max_model_len=2200,
            max_num_batched_tokens=1500,
            disable_mm_preprocessor_cache=True,
            gpu_memory_utilization=0.9,
            dtype="bfloat16",
            enforce_eager=True,
            max_num_seqs=1500,
            model_impl="transformers",
            limit_mm_per_prompt={"image": 1},
            enable_prefix_caching=False,
            enable_chunked_prefill=False
        )

    processor = AutoProcessor.from_pretrained(model_name)
    messages = [
        {'role': 'user', 'content': [
            {"type": "text", "text": question},
            {"type": "image"}
        ]}
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return llm, prompt, None

def run_idefics3(question: str, modality: str):
    assert modality == "image"
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
        f"<|begin_of_text|>User:<image>{question}<end_of_utterance>\nAssistant:"
    )
    return llm, prompt, None

def run_internvl(question: str, modality: str):
    assert modality == "image"
    model_name = "OpenGVLab/InternVL3-2B-hf"

    llm = LLM(
        model=model_name,
        max_model_len=4000,
        max_num_batched_tokens=3500,
        disable_mm_preprocessor_cache=True,
        gpu_memory_utilization=0.7,
        dtype="bfloat16",
        enforce_eager=True,
        max_num_seqs=3500,
        model_impl="transformers",
        limit_mm_per_prompt={"image": 1},
        enable_prefix_caching=False,
        enable_chunked_prefill=False
    )

    processor = AutoProcessor.from_pretrained(model_name)
    messages = [
        {'role': 'user', 'content': [
            {"type": "text", "text": question},
            {"type": "image"}
        ]}
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return llm, prompt, None

def run_llava(question: str, modality: str):
    assert modality == "image"
    prompt = f"USER: <image>\n{question}\nASSISTANT:"

    llm = LLM(
        model="llava-hf/llava-1.5-7b-hf",
        max_model_len=4000,
        max_num_batched_tokens=1500,
        disable_mm_preprocessor_cache=True,
        gpu_memory_utilization=0.5,
        dtype="float16",
        enforce_eager=True,
        max_num_seqs=1500,
        model_impl="transformers",
        limit_mm_per_prompt={"image": 2},
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
    )
    return llm, prompt, None

def run_pixtral(question: str, modality: str):
    assert modality == "image"
    prompt = f"USER: [IMG]\n{question}\nASSISTANT:"

    llm = LLM(
        model='mistral-community/pixtral-12b',
        max_model_len=4500,
        max_num_batched_tokens=1500,
        disable_mm_preprocessor_cache=True,
        gpu_memory_utilization=0.5,
        dtype="float16",
        enforce_eager=True,
        max_num_seqs=1500,
        model_impl="transformers",
        limit_mm_per_prompt={"image": 2},
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        # mm_processor_kwargs={"size": {"height": 336, "width": 400}},
    )
    return llm, prompt, None

def run_llava_next(question: str, modality: str):
    assert modality == "image"
    prompt = f"[INST] <image>\n{question} [/INST]"
    llm = LLM(
        model="llava-hf/llava-v1.6-mistral-7b-hf",
        max_model_len=8192,
        max_num_batched_tokens=5000,
        disable_mm_preprocessor_cache=True,
        gpu_memory_utilization=0.5,
        dtype="float16",
        enforce_eager=True,
        max_num_seqs=5000,
        model_impl="transformers",
        limit_mm_per_prompt={"image": 2},
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
    )
    return llm, prompt, None

def run_llava_onevision(question: str, modality: str):
    assert modality == "image"

    prompt = f"<|im_start|>user <image>\n{question}<|im_end|> \
    <|im_start|>assistant\n"
    llm = LLM(
        model="llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
        max_model_len=8192,
        max_num_batched_tokens=5000,
        disable_mm_preprocessor_cache=True,
        gpu_memory_utilization=0.5,
        dtype="float16",
        enforce_eager=True,
        max_num_seqs=5000,
        model_impl="transformers",
        limit_mm_per_prompt={"image": 2},
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
    )
    return llm, prompt, None

def run_mistral3(question: str, modality: str):
    assert modality == "image"

    model_name = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    llm = LLM(
            model=model_name,
            max_model_len=2000,
            max_num_batched_tokens=1500,
            disable_mm_preprocessor_cache=True,
            gpu_memory_utilization=0.9,
            dtype="bfloat16",
            max_num_seqs=1500,
            model_impl="transformers",
            limit_mm_per_prompt={"image": 1},
            enable_prefix_caching=False,
            enable_chunked_prefill=False
        )

    processor = AutoProcessor.from_pretrained(model_name)
    messages = [
        {'role': 'user', 'content': [
            {"type": "text", "text": question},
            {"type": "image"}
        ]}
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return llm, prompt, None

def run_mllama(question: str, modality: str):
    assert modality == "image"

    model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    llm = LLM(
        model=model_name,
        max_model_len=8192,
        max_num_batched_tokens=5000,
        disable_mm_preprocessor_cache=True,
        gpu_memory_utilization=0.5,
        dtype="float16",
        enforce_eager=True,
        max_num_seqs=5000,
        model_impl="transformers",
        limit_mm_per_prompt={"image": 2},
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
    )

    processor = AutoProcessor.from_pretrained(model_name)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": f"{question}"}
        ]
    }]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return llm, prompt, None

def run_paligemma(question: str, modality: str):
    assert modality == "image"

    prompt = "caption en"
    llm = LLM(
        model="google/paligemma-3b-pt-224",
        max_model_len=5000,
        max_num_batched_tokens=3000,
        disable_mm_preprocessor_cache=True,
        gpu_memory_utilization=0.7,
        dtype="float16",
        enforce_eager=True,
        max_num_seqs=3000,
        model_impl="transformers",
        limit_mm_per_prompt={"image": 2},
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
    )
    return llm, prompt, None

def run_paligemma2(question: str, modality: str):
    assert modality == "image"

    prompt = "caption en"
    llm = LLM(
        model="google/paligemma2-3b-pt-224",
        max_model_len=8192,
        max_num_batched_tokens=5000,
        disable_mm_preprocessor_cache=True,
        gpu_memory_utilization=0.5,
        dtype="float16",
        enforce_eager=True,
        max_num_seqs=5000,
        model_impl="transformers",
        limit_mm_per_prompt={"image": 2},
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
    )
    return llm, prompt, None

def run_qwen2_vl(question: str, modality: str):
    assert modality == "image"

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
        disable_mm_preprocessor_cache=True,
        gpu_memory_utilization=0.8,
        dtype="bfloat16",
        enforce_eager=True,
        max_num_seqs=3000,
        model_impl="transformers",
        limit_mm_per_prompt={"image": 2},
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
    )

    prompt = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
              f"<|im_start|>user\n<|vision_start|><|image_pad|>|vision_end|>"
              f"{question}<|im_end|>\n"
              "<|im_start|>assistant\n")
    return llm, prompt, None

def run_qwen2_5_vl(question: str, modality: str):
    assert modality == "image"
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    llm = LLM(
        model=model_name,
        max_model_len=3000,
        max_num_batched_tokens=3000,
        disable_mm_preprocessor_cache=True,
        gpu_memory_utilization=0.8,
        dtype="bfloat16",
        enforce_eager=True,
        max_num_seqs=3000,
        model_impl="transformers",
        limit_mm_per_prompt={"image": 1},
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
        },
    )

    prompt = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
              f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
              f"{question}<|im_end|>\n"
              "<|im_start|>assistant\n")
    return llm, prompt, None

def run_vipllava(question: str, modality: str):
    assert modality == "image"

    model_name = "llava-hf/vip-llava-7b-hf"
    llm = LLM(
            model=model_name,
            max_model_len=2200,
            max_num_batched_tokens=1500,
            disable_mm_preprocessor_cache=True,
            gpu_memory_utilization=0.6,
            dtype="bfloat16",
            enforce_eager=True,
            max_num_seqs=1500,
            model_impl="transformers",
            limit_mm_per_prompt={"image": 1},
            enable_prefix_caching=False,
            enable_chunked_prefill=False
        )

    processor = AutoProcessor.from_pretrained(model_name)
    messages = [
        {'role': 'user', 'content': [
            {"type": "text", "text": question},
            {"type": "image"}
        ]}
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return llm, prompt, None


model_example_map = {
    "aria": run_aria,
    "aya_vision": run_aya_vision,
    "chameleon": run_chameleon, # NOTE: DONE but needs to add suppress token in hub generation config
    "emu3": run_emu,
    "gemma3": run_gemma3, # TODO: doesn't work even with text-only input, needs to somehow pass token type ids
    "fuyu": run_fuyu, # Almist there, needs new attn interface for Persimmon LM backend
    "got_ocr": run_got_ocr, # More complex as it needs to add boxes/etc. Might support later
    "idefics3": run_idefics3,
    "internvl_chat": run_internvl,
    "llava": run_llava,
    "pixtral": run_pixtral,
    "llava_next": run_llava_next,
    "llava_onevision": run_llava_onevision,
    "mllama": run_mllama, # Cross attn not yet supported
    "mistral3": run_mistral3,
    "paligemma": run_paligemma,
    "paligemma2": run_paligemma2,
    "qwen2_vl": run_qwen2_vl,
    "qwen2_5_vl": run_qwen2_5_vl,
    "vipllava": run_vipllava,
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
    # images = [load_image(f"https://picsum.photos/id/23{i}/{i+4}00/{i+6}00") for i in range(args.num_prompts + args.num_images)]
    images = [load_image(f"/raid/raushan/image.png") for _ in range(args.num_prompts + args.num_images)]
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
            "multi_modal_data": {modality: [images[i] for i in range(args.num_images)]},
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
                "multi_modal_data": {modality: [images[i+j] for j in range(args.num_images)]},
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
    parser.add_argument('--num-images',
                        type=int,
                        default=1
                    )

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

