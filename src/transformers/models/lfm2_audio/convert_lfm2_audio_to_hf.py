from transformers import Lfm2AudioProcessor, ParakeetFeatureExtractor, AutoTokenizer
import argparse

CHECKPOINT = "LiquidAI/LFM2.5-Audio-1.5B"
AUDIO_1 = "/teamspace/studios/this_studio/transformers/src/transformers/models/lfm2_audio/asr.wav"


CHAT_TEMPLATE = """{%- set ns = namespace(system_prompt="") -%}
{%- if messages and messages[0]["role"] == "system" -%}
    {%- set sys = messages[0]["content"] -%}
    {%- if sys is string -%}
        {%- set ns.system_prompt = sys -%}
    {%- else -%}
        {%- for part in sys -%}
            {%- if part is mapping and part.get("type") == "text" -%}
                {%- set ns.system_prompt = ns.system_prompt + part.get("text", "") -%}
            {%- endif -%}
        {%- endfor -%}
    {%- endif -%}
    {%- set messages = messages[1:] -%}
{%- endif -%}

{%- if ns.system_prompt -%}
{{- "<|im_start|>system\n" + ns.system_prompt + "<|im_end|>\n" -}}
{%- endif -%}

{%- for message in messages -%}
{{- "<|im_start|>" + message["role"] + "\n" -}}
{%- if message["role"] == "user" -%}
{# keep user body empty so audio paths/content are not serialized #}
{%- else -%}
    {%- set content = message["content"] -%}
    {%- if content is string -%}
        {{- content -}}
    {%- else -%}
        {%- for part in content -%}
            {%- if part is mapping and part.get("type") == "text" -%}
                {{- part.get("text", "") -}}
            {%- endif -%}
        {%- endfor -%}
    {%- endif -%}
{%- endif -%}
{{- "<|im_end|>\n" -}}
{%- endfor -%}

{%- if add_generation_prompt -%}
{{- "<|im_start|>assistant\n" -}}
{%- endif -%}"""

def prepare_asr_inputs():
    sr = 16000
    window_size = 0.025
    win_length = int(sr * window_size) # 400
    window_stride = 0.01
    hop_length = int(sr * window_stride) # 160
    feature_extractor = ParakeetFeatureExtractor(
        feature_size=128,
        win_length=win_length,
        hop_length=hop_length,
        nfft=512,
        sampling_rate=sr,
    )
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    processor = Lfm2AudioProcessor(feature_extractor, tokenizer)

    conversation = [
        {'role': 'system', 'content': [{'type': 'text', 'text': 'Perform ASR.'}]},
        {"role": "user", "content": [
            {"type": "audio", "audio": "/teamspace/studios/this_studio/transformers/src/transformers/models/lfm2_audio/asr.wav"},
        ]}
    ]
    processor.chat_template = CHAT_TEMPLATE
    inputs = processor.apply_chat_template(conversation, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)

    return inputs, processor


def main(asr_inputs: bool, check_inputs: bool):

    if asr_inputs:
        inputs, processor = prepare_asr_inputs()
        print(inputs)

    if check_inputs:
        input_prompt = processor.tokenizer.decode(inputs['input_ids'][0])
        assert input_prompt == '<|startoftext|><|im_start|>system\nPerform ASR.<|im_end|>\n<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n', "Input IDs do not match expected format."


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--asr_inputs",
        default=True,
        type=bool,
        help="Whether to prepare ASR inputs for the model.",
    )
    parser.add_argument(
        "--check_inputs",
        default=True,
        type=bool,
        help="Whether to check ASR inputs for the model.",
    )
    args = parser.parse_args()
    main(args.asr_inputs, args.check_inputs)